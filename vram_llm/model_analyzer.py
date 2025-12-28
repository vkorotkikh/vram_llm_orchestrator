"""
Model analyzer for GGUF files.

Reads GGUF metadata to determine:
- Number of layers
- Size of each layer/tensor
- Optimal GPU distribution based on actual tensor sizes
"""
from __future__ import annotations

import logging
import re
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
class TorchGPUStats:
    """Lightweight snapshot of a CUDA device via torch.cuda."""
    index: int
    name: str
    total_mib: int
    free_mib: int
    used_mib: int
    

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

    def get_torch_based_distribution(
        self,
        reserve_mib: int = 512,
        devices: Optional[list[int]] = None,
    ) -> tuple[list[float], list[int], list[TorchGPUStats]]:
        """
        Calculate layer distribution using torch.cuda for live GPU free memory.

        Returns empty lists if torch is unavailable or no CUDA GPUs are visible.
        """
        gpus = _list_torch_gpus(devices)
        if not gpus:
            return [], [], []

        gpu_budgets = [g.free_mib for g in gpus]
        tensor_split, layers_per_gpu = self.get_layer_distribution(
            gpu_budgets_mib=gpu_budgets,
            reserve_mib=reserve_mib,
        )
        return tensor_split, layers_per_gpu, gpus


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


def _bytes_to_mib(v: int) -> int:
    return int(v // (1024 * 1024))


def _resolve_multipart_paths(path: Path) -> list[Path]:
    """
    If the model is split (e.g., foo-00001-of-00002.gguf), return all parts.
    Otherwise return [path].
    """
    if not path.name.lower().endswith(".gguf"):
        raise ValueError(f"Expected a GGUF file, got: {path}")

    m = re.match(r"(.+)-(\d+)-of-(\d+)\.gguf$", path.name, re.IGNORECASE)
    if not m:
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        return [path]

    prefix, idx_str, total_str = m.group(1), m.group(2), m.group(3)
    total_parts = int(total_str)
    width = len(idx_str)

    parts: list[Path] = []
    for i in range(1, total_parts + 1):
        candidate = path.with_name(f"{prefix}-{i:0{width}d}-of-{total_str}.gguf")
        if not candidate.exists():
            raise FileNotFoundError(f"Missing GGUF part {i} of {total_parts}: {candidate}")
        parts.append(candidate)
    return parts


def _layer_logger() -> logging.Logger:
    """File logger dedicated to layer-size aggregation and header details."""
    default_path = Path(__file__).resolve().parent / "layer_sizes.log"
    log_path = Path(os.environ.get("VRAM_LLM_LAYER_LOG", str(default_path)))

    logger = logging.getLogger("vram_llm.layer_sizes")
    if not logger.handlers:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.info("Layer size log started: %s", log_path)
    return logger


def _list_torch_gpus(devices: Optional[list[int]] = None) -> list[TorchGPUStats]:
    """List GPUs using torch.cuda (per-process free memory)."""
    try:
        import torch
    except Exception:
        return []

    try:
        if not torch.cuda.is_available():
            return []
        count = torch.cuda.device_count()
    except Exception:
        return []

    indices = list(range(count))
    if devices is not None:
        indices = [i for i in devices if 0 <= i < count]

    gpus: list[TorchGPUStats] = []
    for idx in indices:
        try:
            with torch.cuda.device(idx):
                free_b, total_b = torch.cuda.mem_get_info()
            props = torch.cuda.get_device_properties(idx)
        except Exception:
            continue

        total_mib = _bytes_to_mib(int(total_b))
        free_mib = _bytes_to_mib(int(free_b))
        used_mib = max(0, total_mib - free_mib)

        gpus.append(
            TorchGPUStats(
                index=idx,
                name=str(props.name),
                total_mib=total_mib,
                free_mib=free_mib,
                used_mib=used_mib,
            )
        )
    return gpus

def calculate_layers_per_gpu(
    model_path: str,
    gpu_budgets_mib: list[int],
    layer_sizes: list[int],
) -> tuple[list[float], list[int]]:
    """
    Calculate optimal layer distribution across GPUs.
    """
    n_gpus = len(gpu_budgets_mib)
    if n_gpus == 0:
        return [], []
    
    # Convert budgets to bytes, accounting for reserve
    budgets_bytes = [(b - reserve_mib) * 1024 * 1024 for b in gpu_budgets_mib]
    
    # identify the largest gpu by vram size and make it gpu 0
    largest_gpu = max(gpu_budgets_mib, key=lambda x: x)
    gpu_budgets_mib[0] = largest_gpu
    gpu_budgets_mib.remove(largest_gpu) 
    
    # GPU 0 also holds embeddings and output layer, reduce its budget
    overhead_gpu0 = embedding_size + output_size
    budgets_bytes[0] = max(0, budgets_bytes[0] - overhead_gpu0)
    
    # Greedily assign layers to GPUs
    layers_per_gpu = [0] * n_gpus
    remaining_budget = list(budgets_bytes)
    
    current_gpu = 0
    
    for layer_idx, layer_size in enumerate(layer_sizes):
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
            
    return layers_per_gpu
                    
def analyze_gguf_model(model_path: str) -> ModelAnalysis:
    """
    Analyze a GGUF model file to determine layer sizes.
    
    For split models (part N of M), this analyzes all parts and aggregates tensors.
    """
    path = Path(model_path).expanduser().resolve()
    paths = _resolve_multipart_paths(path)
    logger = _layer_logger()
    log_path = getattr(logger.handlers[0], "baseFilename", "layer_sizes.log") if logger.handlers else "layer_sizes.log"
    logger.info("Analyzing GGUF parts: %s", ", ".join(p.name for p in paths))
    print(f"[vram-llm] analyzing {len(paths)} GGUF part(s); detailed log -> {log_path}")
    
    # Collect tensor information
    tensors: list[TensorInfo] = []
    n_layers_meta = 0
    
    for part_path in paths:
        with open(part_path, 'rb') as f:
            # Read header to understand what this shard contains.
            magic = struct.unpack('<I', f.read(4))[0]
            if magic != GGUF_MAGIC:
                raise ValueError(f"Invalid GGUF magic in {part_path}: {hex(magic)}")
            
            version = struct.unpack('<I', f.read(4))[0]
            tensor_count = struct.unpack('<Q', f.read(8))[0]
            metadata_kv_count = struct.unpack('<Q', f.read(8))[0]
            print(f"[vram-llm] {part_path.name}: version={version} tensors={tensor_count} metadata_kv={metadata_kv_count}")
            logger.info(
                "Part %s: version=%s tensor_count=%s metadata_kv=%s",
                part_path.name,
                version,
                tensor_count,
                metadata_kv_count,
            )
            
            # Read metadata to find n_layers
            for _ in range(metadata_kv_count):
                key = _read_string(f)
                value_type = struct.unpack('<I', f.read(4))[0]
                
                if key.endswith('.block_count') or key.endswith('.n_layer'):
                    val = _read_metadata_uint32(f, value_type)
                    if val is not None:
                        n_layers_meta = max(n_layers_meta, val)
                else:
                    _skip_metadata_value(f, value_type)
            
            # Read tensor info
            for _ in range(tensor_count):
                name = _read_string(f)
                n_dims = struct.unpack('<I', f.read(4))[0]
                dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
                tensor_type = struct.unpack('<I', f.read(4))[0]
                _offset = struct.unpack('<Q', f.read(8))[0]
                
                # Calculate tensor size in bytes; for quantized tensors we approximate
                # using per-block sizes. This is what we later aggregate to per-layer MB/GB.
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
                
                # Log tensor-level details to the layer log for inspection
                size_mb = size_bytes / (1024 * 1024)
                logger.info(
                    "Tensor %s | layer=%s | type=%s | dims=%s | size=%.2f MB",
                    name,
                    layer_idx if layer_idx is not None else "-",
                    tensor_type,
                    dims,
                    size_mb,
                )
                tensors.append(TensorInfo(name=name, size_bytes=size_bytes, layer_index=layer_idx))
    
    # Derive n_layers from metadata or fallback to max layer index seen
    max_layer_seen = max((t.layer_index for t in tensors if t.layer_index is not None), default=-1)
    n_layers = max(n_layers_meta, max_layer_seen + 1)
    
    # Aggregate by layer after we know layer count
    # layer_sizes[i] holds the total bytes for all tensors that belong to layer i
    layer_sizes = [0] * n_layers if n_layers > 0 else []
    layer_counts = [0] * n_layers if n_layers > 0 else []
    embedding_size = 0
    output_size = 0
    
    for t in tensors:
        if t.layer_index is not None and 0 <= t.layer_index < len(layer_sizes):
            layer_sizes[t.layer_index] += t.size_bytes
            layer_counts[t.layer_index] += 1
        elif 'embed' in t.name.lower() or 'token_embd' in t.name.lower():
            embedding_size += t.size_bytes
        elif 'output' in t.name.lower() or 'lm_head' in t.name.lower():
            output_size += t.size_bytes

    # Log per-layer totals to make it easy to eyeball allocation candidates
    for idx, size_bytes in enumerate(layer_sizes):
        size_mb = size_bytes / (1024 * 1024)
        logger.info("Layer %d total: %.2f MB (%d tensors)", idx, size_mb, layer_counts[idx])
    logger.info("Embeddings total: %.2f MB", embedding_size / (1024 * 1024))
    logger.info("Output head total: %.2f MB", output_size / (1024 * 1024))

    # Calculate total size and average layer size
    total_size = sum(t.size_bytes for t in tensors)
    avg_layer_size = sum(layer_sizes) // len(layer_sizes) if layer_sizes else 0
    
    print(f"[vram-llm] total size: {total_size / (1024**3):.2f} GB")

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
    
    reserve_mib = int(os.environ.get("VRAM_LLM_RESERVE_MIB", "512"))
    analysis = analyze_gguf_model(sys.argv[1])
    print_model_analysis(analysis)
    
    # If torch is available, show a split based on live CUDA free memory.
    torch_split, torch_layers_per_gpu, torch_gpus = analysis.get_torch_based_distribution(
        reserve_mib=reserve_mib
    )
    if torch_gpus:
        print(f"\n  PyTorch CUDA snapshot (reserve {reserve_mib} MiB per GPU):")
        for g, layers in zip(torch_gpus, torch_layers_per_gpu):
            print(f"    GPU {g.index} ({g.name}): free {g.free_mib} / total {g.total_mib} MiB -> {layers} layers")
        print(f"  tensor_split: {[round(t, 3) for t in torch_split]}")
    else:
        # Fallback example: fixed budgets to illustrate usage.
        gpu_budgets = [24000, 24000, 12000]  # MiB
        tensor_split, layers_per_gpu = analysis.get_layer_distribution(
            gpu_budgets,
            reserve_mib=reserve_mib,
        )
        print(f"\n  (Torch not available; using example budgets {gpu_budgets})")
        print(f"  Suggested tensor_split: {tensor_split}")
        print(f"  Layers per GPU: {layers_per_gpu}")
