# VRAM LLM Orchestrator (POC)

Backend-only service that loads GGUF models via **llama.cpp** (through **llama-cpp-python**) and **automatically balances model layers across all visible CUDA GPUs** based on *free* VRAM (not total VRAM).

Key ideas:
- Query free VRAM via NVML.
- Optionally reorder `CUDA_VISIBLE_DEVICES` so the biggest free-memory GPU becomes logical GPU 0.
- Compute `tensor_split` weights proportional to per-GPU free VRAM minus a safety reserve.
- Use llama.cpp multi-GPU support (`split_mode=layer` by default) and offload *all* layers when possible.

> This is a proof-of-concept skeleton meant to be extended (auth, multi-model, metrics, structured logging, etc.).

## Requirements
- Python 3.10+
- NVIDIA driver + CUDA runtime (for CUDA inference)
- `llama-cpp-python` built/installed **with CUDA enabled**.

## Install
```bash
python -m venv .venv
source .venv/bin/activate            # (Linux/macOS)
# .venv\Scripts\activate             # (Windows)

pip install -r requirements.txt
```

## Run (API server)
```bash
vram-llm serve --model /path/to/model.gguf --host 0.0.0.0 --port 8080
```

### Useful flags
- `--reserve-mib 1024` : keep this much VRAM free per GPU (default 1024 MiB).
- `--device-order free_desc` : reorder GPUs by free VRAM (default).
- `--split-mode layer` : layer split (default) or `row` / `none`.
- `--no-set-cuda-visible-devices` : do not rewrite CUDA_VISIBLE_DEVICES.
- `--devices 0,2` : use only a subset of physical GPU indices.

## API
- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions` (OpenAI-ish)
- `POST /v1/completions` (OpenAI-ish)

Example:
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role":"user","content":"Write a haiku about PCIe lanes."}],
    "max_tokens": 128,
    "temperature": 0.7
  }'
```

## Notes
- `split_mode=layer` splits layers (and KV cache) across GPUs; this avoids the “small GPU becomes the KV cache sink” problem that can happen in row-split if the main GPU is small.
- If you are mixing a small “display GPU” with larger compute GPUs, the default GPU reordering helps keep the heaviest allocations off that smaller device.

## Optional: llama-fit-params (experimental)
If you have a recent llama.cpp build that includes `llama-fit-params`, you can enable it:

```bash
vram-llm serve --model /path/to/model.gguf --use-fit-params
```

This delegates the layer/tensor split calculation to llama.cpp's own parameter fitting logic.
