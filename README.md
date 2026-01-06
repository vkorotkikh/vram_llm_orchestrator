# VRAM LLM Orchestrator (POC)

Backend-only service that loads GGUF models via **llama.cpp** (through **llama-cpp-python**) and **automatically balances model layers across all visible CUDA GPUs** based on *free* VRAM (not total VRAM).

## Key Features

- **VRAM-Aware Allocation**: Query free VRAM via NVML and allocate layers proportionally
- **GPU Reordering**: Automatically reorder `CUDA_VISIBLE_DEVICES` so the GPU with highest free VRAM becomes GPU 0
- **Smart Layer Allocation**: Analyze actual GGUF layer sizes for precise GPU distribution
- **Multi-GPU Support**: Uses llama.cpp `split_mode=layer` to distribute layers across GPUs
- **Memory Optimization**: Options to reduce system RAM usage during model loading

> This is a proof-of-concept skeleton meant to be extended (auth, multi-model, metrics, structured logging, etc.).

---

## Requirements

- Python 3.10+
- NVIDIA driver + CUDA runtime (for CUDA inference)
- `llama-cpp-python` built/installed **with CUDA enabled**
- PyTorch with CUDA (optional, for accurate GPU memory detection)

## Install

```bash
# Create conda environment (recommended)
conda create -n vram_llm python=3.11
conda activate vram_llm

# Install dependencies
pip install -r requirements.txt

# Install llama-cpp-python with CUDA support
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

# Optional: Install PyTorch for accurate GPU memory detection
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

---

## Commands

### 1. Serve (API Server)

Start the HTTP API server with a loaded model:

```bash
vram-llm serve --model /path/to/model.gguf --host 0.0.0.0 --port 8080
```

#### Basic Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | required | Path to GGUF model file |
| `--host` | 127.0.0.1 | Server host |
| `--port` | 8080 | Server port |
| `--n-ctx` | 4096 | Context size |
| `--n-batch` | 512 | Batch size |
| `--n-gpu-layers` | -1 | Layers to offload (-1 = all) |

#### GPU Allocation Options

| Flag | Default | Description |
|------|---------|-------------|
| `--reserve-mib` | 1024 | VRAM reserve per GPU (MiB) |
| `--min-budget-mib` | 1024 | Minimum usable VRAM per GPU |
| `--device-order` | free_desc | GPU ordering: `free_desc`, `total_desc`, `index` |
| `--devices` | all | Comma-separated GPU indices (e.g., `0,2`) |
| `--split-mode` | layer | Split mode: `layer`, `row`, `none` |

#### Smart Allocation Options (NEW)

| Flag | Description |
|------|-------------|
| `--smart-split` | Analyze model to compute layer-size-aware tensor split |
| `--tensor-split "0.35,0.45,0.20"` | Manual tensor split proportions |

#### Memory Management Options (NEW)

| Flag | Description |
|------|-------------|
| `--no-mmap` | Disable memory-mapped loading (reduces system RAM usage) |
| `--mlock` | Lock model in RAM (prevent swapping) |

#### Examples

```bash
# Basic serve with auto allocation
vram-llm serve --model ~/models/llama-7b.gguf

# Smart allocation (analyzes layer sizes)
vram-llm serve --model ~/models/large-model.gguf --smart-split --no-mmap

# Manual tensor split for 3 GPUs
vram-llm serve --model ~/models/model.gguf --tensor-split "0.35,0.45,0.20"

# Reduced context for large models
vram-llm serve --model ~/models/49b-model.gguf --smart-split --no-mmap --n-ctx 2048
```

---

### 2. Analyze (NEW)

Analyze a GGUF model and show recommended GPU allocation:

```bash
vram-llm analyze --model /path/to/model.gguf
```

This command:
- Reads GGUF metadata to determine layer sizes
- Detects available GPUs and their free VRAM
- Calculates optimal layer-to-GPU allocation
- Shows the recommended `--tensor-split` values

#### Example Output

```
[Model Analysis] /path/to/model.gguf
  Total size: 49.35 GB
  Layers: 80
  Avg layer size: 605.04 MB
  Layer size range: 70.16 - 867.06 MB

[vram-llm] GPUs sorted by free VRAM (GPU0 = highest):
  GPU 0 (CUDA:1): NVIDIA GeForce RTX 3090 Ti - 23,287 / 24,563 MiB
  GPU 1 (CUDA:2): NVIDIA GeForce RTX 3090 Ti - 23,287 / 24,563 MiB
  GPU 2 (CUDA:0): NVIDIA GeForce RTX 4080 Laptop GPU - 11,047 / 12,281 MiB

======================================================================
LAYER-TO-GPU ALLOCATION PLAN
======================================================================
  GPU 0:  26 layers (#0-25)   | 21,810 / 23,287 MiB (93.7%)
  GPU 1:  47 layers (#26-72)  | 21,587 / 23,287 MiB (92.7%)
  GPU 2:   7 layers (#73-79)  |  7,134 / 11,047 MiB (64.6%)

TENSOR SPLIT: --tensor-split "0.33,0.59,0.09"
```

---

### 3. Plan

Print the computed GPU allocation plan without loading a model:

```bash
vram-llm plan --reserve-mib 1024
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/models` | GET | List loaded models |
| `/v1/chat/completions` | POST | Chat completions (OpenAI-compatible) |
| `/v1/completions` | POST | Text completions (OpenAI-compatible) |

### Example: Chat Completion

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role":"user","content":"Write a haiku about multi-GPU inference."}],
    "max_tokens": 128,
    "temperature": 0.7
  }'
```

### Example: Python Client

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "Explain quantum computing."}],
    max_tokens=256
)
print(response.choices[0].message.content)
```

---

## Architecture

### GPU Allocation Flow

1. **Detection**: Query NVML for all CUDA GPUs and their free VRAM
2. **Reordering**: Sort GPUs by free VRAM (highest first = GPU 0)
3. **Analysis** (with `--smart-split`): Parse GGUF to get actual layer sizes
4. **Allocation**: Assign layers sequentially, filling GPU 0, then GPU 1, etc.
5. **Loading**: Pass `tensor_split` proportions to llama.cpp

### Key Files

| File | Purpose |
|------|---------|
| `cli.py` | Command-line interface and argument parsing |
| `model_analyzer.py` | GGUF parsing, layer size analysis, allocation planning |
| `llama_cpp_engine.py` | llama-cpp-python wrapper with tensor_split handling |
| `allocator.py` | NVML-based VRAM detection and basic allocation |
| `gpu.py` | GPU listing and CUDA_VISIBLE_DEVICES parsing |
| `api.py` | FastAPI endpoints |

---

## Notes

### Layer Split Mode

`split_mode=layer` splits layers (and KV cache) across GPUs. This avoids the "small GPU becomes the KV cache sink" problem that can happen in row-split if the main GPU is small.

### Mixed GPU Configurations

If you are mixing a small "display GPU" with larger compute GPUs, the default GPU reordering (by free VRAM) helps keep the heaviest allocations off that smaller device.

### Memory Usage

- **With mmap (default)**: Model is memory-mapped from disk. Uses significant system RAM.
- **With `--no-mmap`**: Model loaded directly to GPU. Lower system RAM usage but slower cold start.

### Layer Size Variance

Transformer models have varying layer sizes:
- **Early layers** (embeddings, first attention blocks): Often 20-30% larger
- **Middle layers**: Consistent size
- **Output layer**: Similar size to embeddings

The `--smart-split` option accounts for this by analyzing actual layer sizes.

---

## Optional: llama-fit-params (Experimental)

If you have a recent llama.cpp build that includes `llama-fit-params`:

```bash
vram-llm serve --model /path/to/model.gguf --use-fit-params
```

This delegates the layer/tensor split calculation to llama.cpp's own parameter fitting logic.

---

## Troubleshooting

### Out of Memory (OOM)

1. Try `--smart-split` for layer-size-aware allocation
2. Use `--no-mmap` to reduce system RAM usage
3. Reduce context: `--n-ctx 2048`
4. Manual conservative split: `--tensor-split "0.30,0.40,0.30"`
5. Partial CPU offload: `--n-gpu-layers 60`

### High System RAM Usage

Use `--no-mmap` to disable memory-mapped loading:

```bash
vram-llm serve --model /path/to/model.gguf --no-mmap
```

### GPU Not Detected

Ensure CUDA is working:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

---

## License

MIT
