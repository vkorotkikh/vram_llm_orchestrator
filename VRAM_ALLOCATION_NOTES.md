# VRAM LLM Orchestrator - Configuration Notes

> **Generated from chat session with Claude Opus 4.5**  
> Date: December 25, 2024

---

## Hardware Configuration

| GPU | Model | VRAM | Physical Index |
|-----|-------|------|----------------|
| GPU 0 | NVIDIA GeForce RTX 4080 Laptop GPU | 12 GB | 0 |
| GPU 1 | NVIDIA GeForce RTX 3090 Ti | 24 GB | 1 |
| GPU 2 | NVIDIA GeForce RTX 3090 Ti | 24 GB | 2 |

**Total VRAM:** ~60 GB  
**Usable VRAM (after reserves):** ~55 GB

---

## Working Configuration

### Model: Nemotron 49B Q8_0

**Model Path:**
```
/home/nomad/models/models/nemotron49b-q8/nvidia_Llama-3_3-Nemotron-Super-49B-v1_5-Q8_0/nvidia_Llama-3_3-Nemotron-Super-49B-v1_5-Q8_0-00001-of-00002.gguf
```

**Model Size:** ~51 GB (38 GB + 13 GB split files)

### Successful Command

```bash
cd /home/nomad/Documents/vram_llm_orchestrator_poc && \
conda activate vram_llm && \
python -m vram_llm serve \
  --model /home/nomad/models/models/nemotron49b-q8/nvidia_Llama-3_3-Nemotron-Super-49B-v1_5-Q8_0/nvidia_Llama-3_3-Nemotron-Super-49B-v1_5-Q8_0-00001-of-00002.gguf \
  --no-mmap \
  --n-ctx 2048 \
  --tensor-split "0.35,0.45,0.20"
```

### Resulting VRAM Allocation

| CUDA Device | Physical GPU | Allocation | VRAM Used |
|-------------|--------------|------------|-----------|
| CUDA0 | RTX 3090 Ti (#2) | 35% | 22.7 GB |
| CUDA1 | RTX 3090 Ti (#1) | 45% | 18.6 GB |
| CUDA2 | RTX 4080 Laptop | 20% | 10.9 GB |

### Performance

- **Load time:** 32.34 seconds
- **Inference speed:** 12.63 tokens/second
- **Per-token latency:** 79.18 ms

---

## Key Learnings

### 1. Early Layer Compensation

Transformer models have **larger early layers** (embeddings, first attention blocks). The automatic allocation was adjusted to reduce GPU 0's proportion by 15%:

```
Before: [0.43, 0.40, 0.17]
After:  [0.37, 0.45, 0.18]
```

### 2. Memory-Mapped Loading Issue

Using `--no-mmap` is recommended for large models to:
- Reduce system RAM usage during loading
- Avoid mapping the entire model into RAM before GPU transfer

### 3. Model Analyzer Results

The 49B model has:
- **80 layers** with sizes ranging from **0 - 867 MB**
- **Average layer size:** 449 MB
- **Embedding size:** 1.06 GB
- **Output layer size:** 1.06 GB

---

## Alternative Configurations

### Option 1: Automatic Compensation (Default)

```bash
python -m vram_llm serve \
  --model /path/to/model.gguf \
  --no-mmap \
  --n-ctx 2048
```

### Option 2: Conservative Manual Split

```bash
python -m vram_llm serve \
  --model /path/to/model.gguf \
  --no-mmap \
  --n-ctx 2048 \
  --tensor-split "0.30,0.45,0.25"
```

Layer distribution:
- CUDA0: 30% → ~24 layers
- CUDA1: 45% → ~37 layers
- CUDA2: 25% → ~20 layers

### Option 3: Partial CPU Offload (for OOM issues)

```bash
python -m vram_llm serve \
  --model /path/to/model.gguf \
  --no-mmap \
  --n-ctx 2048 \
  --n-gpu-layers 70 \
  --tensor-split "0.35,0.45,0.20"
```

This offloads 11 layers to CPU RAM.

### Option 4: Smart Split (Analyzes Model First)

```bash
python -m vram_llm serve \
  --model /path/to/model.gguf \
  --smart-split \
  --no-mmap \
  --n-ctx 4096
```

---

## Using the Server

### API Endpoints

- `GET /health` - Health check
- `GET /v1/models` - List loaded models
- `POST /v1/chat/completions` - Chat completion (OpenAI-compatible)
- `POST /v1/completions` - Text completion (OpenAI-compatible)

### Quick Test with curl

```bash
# Health check
curl http://127.0.0.1:8080/health

# Chat completion
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 128,
    "temperature": 0.7
  }'
```

### Python Client (OpenAI SDK)

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

## CLI Commands Reference

### Analyze a Model

```bash
vram-llm analyze --model /path/to/model.gguf
```

### Show GPU Allocation Plan

```bash
vram-llm plan
```

### Serve a Model

```bash
vram-llm serve --model /path/to/model.gguf [OPTIONS]
```

**Key Options:**
| Flag | Description |
|------|-------------|
| `--tensor-split "0.35,0.45,0.20"` | Manual split proportions |
| `--smart-split` | Auto-analyze model for optimal split |
| `--no-mmap` | Disable memory-mapped loading (lower RAM) |
| `--n-ctx 2048` | Context window size |
| `--n-gpu-layers 70` | Limit GPU layers (rest on CPU) |
| `--reserve-mib 1024` | VRAM reserve per GPU |

---

## Troubleshooting

### OOM Errors

1. Reduce context size: `--n-ctx 2048` or `--n-ctx 1024`
2. Use manual tensor split with smaller GPU0 proportion
3. Offload layers to CPU: `--n-gpu-layers 70`
4. Use smaller quantization (Q4_K_M instead of Q8_0)

### High System RAM Usage

- Use `--no-mmap` to disable memory-mapped file loading

### Terminating the Server

```bash
# If in foreground
Ctrl+C

# Find and kill by port
kill $(lsof -t -i:8080)
```

---

## Notes on Multi-GPU Inference

- **Layer split mode** distributes layers across GPUs sequentially
- GPUs are active in sequence, not simultaneously
- All GPUs should show memory usage even if utilization appears low
- Use `nvidia-smi dmon -s u -d 1` for accurate utilization monitoring

