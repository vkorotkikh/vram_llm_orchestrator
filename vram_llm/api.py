from __future__ import annotations

from dataclasses import asdict
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .allocator import AllocationPlan
from .gpu import GPUInfo, gpus_markdown_table
from .engines.llama_cpp_engine import LlamaCppEngine


def _error(message: str, code: int = 400, type_: str = "invalid_request_error") -> dict[str, Any]:
    return {"error": {"message": message, "type": type_, "code": code}}


def create_app(
    *,
    engine: LlamaCppEngine,
    plan: AllocationPlan,
    gpus: list[GPUInfo],
) -> FastAPI:
    app = FastAPI(title="VRAM LLM Orchestrator", version="0.1.0")

    app.state.engine = engine
    app.state.plan = plan
    app.state.gpus = gpus

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(_request: Request, exc: Exception):
        # Don't leak internals; return a clean error shape
        return JSONResponse(status_code=500, content=_error(str(exc), code=500, type_="server_error"))

    @app.get("/health")
    async def health():
        loaded = engine.loaded
        return {
            "status": "ok",
            "gpus": [asdict(g) for g in gpus],
            "cuda_visible_devices": plan.cuda_visible_devices,
            "tensor_split": plan.tensor_split,
            "split_mode": plan.split_mode,
            "loaded_model": asdict(loaded) if loaded else None,
        }

    @app.get("/v1/models")
    async def list_models():
        loaded = engine.loaded
        data = []
        if loaded:
            data.append(
                {
                    "id": loaded.model_path,
                    "object": "model",
                    "created": loaded.created_at,
                    "owned_by": "local",
                }
            )
        return {"object": "list", "data": data}

    @app.post("/v1/models/load")
    async def load_model(body: dict[str, Any]):
        model_path = body.get("model_path") or body.get("model") or body.get("path")
        if not model_path:
            raise HTTPException(status_code=400, detail=_error("model_path is required")["error"])

        # Optional overrides
        n_ctx = int(body.get("n_ctx", body.get("ctx", 4096)))
        n_batch = int(body.get("n_batch", body.get("batch", 512)))
        n_threads = int(body.get("n_threads", body.get("threads", 0)))
        n_gpu_layers = int(body.get("n_gpu_layers", body.get("gpu_layers", -1)))
        chat_format = body.get("chat_format")

        engine.unload()
        try:
            loaded = engine.load(
                model_path=model_path,
                plan=plan,
                n_ctx=n_ctx,
                n_batch=n_batch,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
                chat_format=chat_format,
                verbose=True,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=_error(f"Failed to load model: {e}", code=500)["error"])
        return {"ok": True, "loaded_model": asdict(loaded)}

    @app.post("/v1/chat/completions")
    async def chat_completions(payload: dict[str, Any]):
        # If the caller doesn't pass a model id, OpenAI SDKs will still work with many servers.
        payload = dict(payload)
        payload.pop("model", None)

        try:
            return await engine.chat_completions(payload)
        except Exception as e:
            raise HTTPException(status_code=500, detail=_error(str(e), code=500)["error"])

    @app.post("/v1/completions")
    async def completions(payload: dict[str, Any]):
        payload = dict(payload)
        payload.pop("model", None)

        try:
            return await engine.completions(payload)
        except Exception as e:
            raise HTTPException(status_code=500, detail=_error(str(e), code=500)["error"])

    @app.get("/")
    async def root():
        # No frontend UI — but having a human-readable root is useful.
        # We keep it texty on purpose.
        return {
            "service": "vram-llm-orchestrator",
            "endpoints": ["/health", "/v1/models", "/v1/models/load", "/v1/chat/completions", "/v1/completions"],
            "gpus_markdown": gpus_markdown_table(gpus),
        }

    return app
