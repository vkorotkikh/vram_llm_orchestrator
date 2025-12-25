"""
Model analyzer for GGUF files.

Reads GGUF metadata to determine:
- Number of layers
- Size of each layer/tensor
- Optimal GPU distribution based on actual tensor sizes
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os


# GGUF format constants
GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian
GGUF_VERSION = 3

# Tensor type sizes (bytes per element)
GGUF_TENSOR_TYPE_SIZES = {
    0: 4,    # F32
    1: 2,    # F16
    2: 1,    # Q4_0
    3: 1,    # Q4_1
    6: 1,    # Q5_0
    7: 1,    # Q5_1
    8: 1,    # Q8_0
    9: 1,    # Q8_1
    10: 1,   # Q2_K
    11: 1,   # Q3_K
    12: 1,   # Q4_K
    13: 1,   # Q5_K
    14: 1,   # Q6_K
    15: 1,   # Q8_K
    16: 1,   # IQ2_XXS
    17: 1,   # IQ2_XS
    18: 1,   # IQ3_XXS
    19: 1,   # IQ1_S
    20: 1,   # IQ4_NL
    21: 1,   # IQ3_S
    22: 1,   # IQ2_S
    23: 1,   # IQ4_XS
}


@dataclass
class TensorInfo:
    """Information about a single tensor in the model."""
    name: str
    size_bytes: int
    layer_index: Optional[int]  # None if not a layer tensor (e.g., embeddings)
    

@dataclass
class ModelAnalysis:
    """Analysis results for a GGUF model."""
    model_path: str
    total_size_bytes: int
    n_layers: int
    layer_sizes: list[int]  # Size in bytes for each layer
    embedding_size: int  # Size of embedding tensors (always on CPU or GPU 0)
    output_size: int  # Size of output tensors
    avg_layer_size: int
    
    def get_layer_distribution(
        self,
        gpu_budgets_mib: list[int],
        reserve_mib: int = 512,
    ) -> tuple[list[float], list[int]]:
        """
        Calculate optimal layer distribution across GPUs.
        
        Returns:
            tensor_split: Normalized proportions for each GPU
            layers_per_gpu: Number of layers assigned to each GPU
        """
        n_gpus = len(gpu_budgets_mib)
        if n_gpus == 0:
            return [], []
        
        # Convert budgets to bytes, accounting for reserve
        budgets_bytes = [(b - reserve_mib) * 1024 * 1024 for b in gpu_budgets_mib]
        
        # GPU 0 also holds embeddings and output layer, reduce its budget
        overhead_gpu0 = self.embedding_size + self.output_size
        budgets_bytes[0] = max(0, budgets_bytes[0] - overhead_gpu0)
        
        # Greedily assign layers to GPUs
        layers_per_gpu = [0] * n_gpus
        remaining_budget = list(budgets_bytes)
        
        current_gpu = 0
        for layer_idx, layer_size in enumerate(self.layer_sizes):
            # Find a GPU with enough budget
            assigned = False
            for attempt in range(n_gpus):
                gpu = (current_gpu + attempt) % n_gpus
                if remaining_budget[gpu] >= layer_size:
                    layers_per_gpu[gpu] += 1
                    remaining_budget[gpu] -= layer_size
                    current_gpu = gpu
                    assigned = True
                    break
            
            if not assigned:
                # No GPU has enough space - assign to GPU with most remaining
                gpu = remaining_budget.index(max(remaining_budget))
                layers_per_gpu[gpu] += 1
                remaining_budget[gpu] -= layer_size
        
        # Calculate tensor_split proportions based on actual assigned layers
        total_layers = sum(layers_per_gpu)
        if total_layers == 0:
            return [1.0 / n_gpus] * n_gpus, [0] * n_gpus
        
        tensor_split = [l / total_layers for l in layers_per_gpu]
        
        return tensor_split, layers_per_gpu


def _read_string(f) -> str:
    """Read a GGUF string (length-prefixed)."""
    length = struct.unpack('<Q', f.read(8))[0]
    return f.read(length).decode('utf-8', errors='replace')


def _skip_metadata_value(f, value_type: int) -> None:
    """Skip a metadata value based on its type."""
    if value_type == 0:  # UINT8
        f.read(1)
    elif value_type == 1:  # INT8
        f.read(1)
    elif value_type == 2:  # UINT16
        f.read(2)
    elif value_type == 3:  # INT16
        f.read(2)
    elif value_type == 4:  # UINT32
        f.read(4)
    elif value_type == 5:  # INT32
        f.read(4)
    elif value_type == 6:  # FLOAT32
        f.read(4)
    elif value_type == 7:  # BOOL
        f.read(1)
    elif value_type == 8:  # STRING
        _read_string(f)
    elif value_type == 9:  # ARRAY
        arr_type = struct.unpack('<I', f.read(4))[0]
        arr_len = struct.unpack('<Q', f.read(8))[0]
        for _ in range(arr_len):
            _skip_metadata_value(f, arr_type)
    elif value_type == 10:  # UINT64
        f.read(8)
    elif value_type == 11:  # INT64
        f.read(8)
    elif value_type == 12:  # FLOAT64
        f.read(8)


def _read_metadata_uint32(f, value_type: int) -> Optional[int]:
    """Read a uint32 metadata value, or skip if wrong type."""
    if value_type == 4:  # UINT32
        return struct.unpack('<I', f.read(4))[0]
    _skip_metadata_value(f, value_type)
    return None


def analyze_gguf_model(model_path: str) -> ModelAnalysis:
    """
    Analyze a GGUF model file to determine layer sizes.
    
    For split models (part 1 of N), this analyzes the first file.
    """
    path = Path(model_path).expanduser().resolve()
    
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    
    # Collect tensor information
    tensors: list[TensorInfo] = []
    n_layers = 0
    
    with open(path, 'rb') as f:
        # Read header
        magic = struct.unpack('<I', f.read(4))[0]
        if magic != GGUF_MAGIC:
            raise ValueError(f"Invalid GGUF magic: {hex(magic)}")
        
        version = struct.unpack('<I', f.read(4))[0]
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        metadata_kv_count = struct.unpack('<Q', f.read(8))[0]
        
        # Read metadata to find n_layers
        for _ in range(metadata_kv_count):
            key = _read_string(f)
            value_type = struct.unpack('<I', f.read(4))[0]
            
            if key.endswith('.block_count') or key.endswith('.n_layer'):
                val = _read_metadata_uint32(f, value_type)
                if val is not None:
                    n_layers = max(n_layers, val)
            else:
                _skip_metadata_value(f, value_type)
        
        # Read tensor info
        for _ in range(tensor_count):
            name = _read_string(f)
            n_dims = struct.unpack('<I', f.read(4))[0]
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
            tensor_type = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]
            
            # Calculate tensor size
            n_elements = 1
            for d in dims:
                n_elements *= d
            
            # Get bytes per element (approximate for quantized types)
            bytes_per_elem = GGUF_TENSOR_TYPE_SIZES.get(tensor_type, 1)
            
            # For quantized types, adjust for block size
            if tensor_type in [2, 3]:  # Q4_0, Q4_1
                size_bytes = (n_elements // 32) * 18  # 32 elements per block
            elif tensor_type in [6, 7]:  # Q5_0, Q5_1
                size_bytes = (n_elements // 32) * 22
            elif tensor_type in [8, 9]:  # Q8_0, Q8_1
                size_bytes = (n_elements // 32) * 34
            else:
                size_bytes = n_elements * bytes_per_elem
            
            # Determine layer index from name
            layer_idx = None
            if '.layers.' in name or 'blk.' in name:
                # Extract layer number from name like "blk.0.attn" or "model.layers.0.self_attn"
                parts = name.replace('.layers.', '.blk.').split('.')
                for i, p in enumerate(parts):
                    if p == 'blk' and i + 1 < len(parts):
                        try:
                            layer_idx = int(parts[i + 1])
                        except ValueError:
                            pass
                        break
                    elif p.isdigit():
                        layer_idx = int(p)
                        break
            
            tensors.append(TensorInfo(name=name, size_bytes=size_bytes, layer_index=layer_idx))
    
    # Aggregate by layer
    layer_sizes = [0] * n_layers if n_layers > 0 else []
    embedding_size = 0
    output_size = 0
    
    for t in tensors:
        if t.layer_index is not None and 0 <= t.layer_index < len(layer_sizes):
            layer_sizes[t.layer_index] += t.size_bytes
        elif 'embed' in t.name.lower() or 'token_embd' in t.name.lower():
            embedding_size += t.size_bytes
        elif 'output' in t.name.lower() or 'lm_head' in t.name.lower():
            output_size += t.size_bytes
    
    total_size = sum(t.size_bytes for t in tensors)
    avg_layer_size = sum(layer_sizes) // len(layer_sizes) if layer_sizes else 0
    
    return ModelAnalysis(
        model_path=str(path),
        total_size_bytes=total_size,
        n_layers=n_layers,
        layer_sizes=layer_sizes,
        embedding_size=embedding_size,
        output_size=output_size,
        avg_layer_size=avg_layer_size,
    )


def print_model_analysis(analysis: ModelAnalysis) -> None:
    """Print a human-readable analysis summary."""
    print(f"\n[Model Analysis] {analysis.model_path}")
    print(f"  Total size: {analysis.total_size_bytes / (1024**3):.2f} GB")
    print(f"  Layers: {analysis.n_layers}")
    print(f"  Avg layer size: {analysis.avg_layer_size / (1024**2):.2f} MB")
    print(f"  Embedding size: {analysis.embedding_size / (1024**2):.2f} MB")
    print(f"  Output size: {analysis.output_size / (1024**2):.2f} MB")
    
    if analysis.layer_sizes:
        min_layer = min(analysis.layer_sizes) / (1024**2)
        max_layer = max(analysis.layer_sizes) / (1024**2)
        print(f"  Layer size range: {min_layer:.2f} - {max_layer:.2f} MB")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m vram_llm.model_analyzer <model.gguf>")
        sys.exit(1)
    
    analysis = analyze_gguf_model(sys.argv[1])
    print_model_analysis(analysis)
    
    # Example: calculate distribution for 3 GPUs with different budgets
    gpu_budgets = [24000, 24000, 12000]  # MiB
    tensor_split, layers_per_gpu = analysis.get_layer_distribution(gpu_budgets)
    print(f"\n  Suggested tensor_split for GPUs {gpu_budgets}: {tensor_split}")
    print(f"  Layers per GPU: {layers_per_gpu}")

