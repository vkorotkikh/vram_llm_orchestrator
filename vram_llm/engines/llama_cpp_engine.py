from __future__ import annotations

import asyncio
import gc
import time
from dataclasses import dataclass
from typing import Any, Optional

from ..allocator import AllocationPlan


@dataclass
class LoadedModel:
    model_path: str
    created_at: int
    n_ctx: int
    n_batch: int
    n_threads: int
    split_mode: str
    tensor_split: list[float]
    n_gpu_layers: int


class LlamaCppEngine:
    """In-process llama-cpp-python engine with multi-GPU VRAM-aware loading."""

    def __init__(self) -> None:
        self._llama = None
        self._loaded: Optional[LoadedModel] = None
        self._lock = asyncio.Lock()

    @property
    def loaded(self) -> Optional[LoadedModel]:
        return self._loaded

    def unload(self) -> None:
        """Unload current model and free memory."""
        self._llama = None
        self._loaded = None
        gc.collect()

    def _split_mode_const(self, llama_cpp: Any, split_mode: str) -> int:
        sm = split_mode.lower().strip()
        if sm == "none":
            return int(getattr(llama_cpp, "LLAMA_SPLIT_MODE_NONE"))
        if sm == "layer":
            return int(getattr(llama_cpp, "LLAMA_SPLIT_MODE_LAYER"))
        if sm == "row":
            return int(getattr(llama_cpp, "LLAMA_SPLIT_MODE_ROW"))
        raise ValueError(f"Unknown split_mode: {split_mode!r}")

    def load(
        self,
        *,
        model_path: str,
        plan: AllocationPlan,
        n_ctx: int = 4096,
        n_batch: int = 512,
        n_threads: int = 0,
        n_gpu_layers: int = -1,
        use_mmap: bool = True,
        use_mlock: bool = False,
        seed: int = 0,
        verbose: bool = True,
        chat_format: Optional[str] = None,
    ) -> LoadedModel:
        """Load a GGUF model using llama-cpp-python.

        Args:
            plan: AllocationPlan computed from free VRAM.
            n_gpu_layers: -1 means "try to offload all layers". If this OOMs, reduce.
        """
        # Lazy import so CUDA_VISIBLE_DEVICES can be set in the CLI before we import llama_cpp
        from llama_cpp import Llama  # type: ignore
        import llama_cpp  # type: ignore

        split_mode_const = self._split_mode_const(llama_cpp, plan.split_mode)

        # If all weights are 0, don't pass tensor_split (llama.cpp interprets None as "no split")
        tensor_split = plan.tensor_split
        if sum(tensor_split) <= 0:
            tensor_split = []
        else:
            # Normalize tensor_split to proportions (sum to 1.0)
            total = sum(tensor_split)
            tensor_split = [w / total for w in tensor_split]
            
            # IMPORTANT: Apply correction factors for VRAM overhead
            # - KV cache takes ~500MB-2GB per GPU depending on context
            # - Compute buffers take ~400MB per GPU
            # - Early layers (on GPU 0) are typically 20-30% larger than average
            # 
            # We reduce GPU 0's proportion to account for:
            # 1. Larger early layers (embeddings, first attention blocks)
            # 2. Additional overhead that accumulates on main_gpu
            if len(tensor_split) > 1:
                # Reserve extra headroom for GPU 0 (early layers are bigger)
                # Reduce GPU 0 by 15%, redistribute to other GPUs
                early_layer_penalty = 0.15
                reduction = tensor_split[0] * early_layer_penalty
                tensor_split[0] -= reduction
                
                # Distribute the reduction to other GPUs proportionally
                remaining_total = sum(tensor_split[1:])
                if remaining_total > 0:
                    for i in range(1, len(tensor_split)):
                        tensor_split[i] += reduction * (tensor_split[i] / remaining_total)
                
                # Re-normalize to ensure sum is exactly 1.0
                total = sum(tensor_split)
                tensor_split = [w / total for w in tensor_split]
            
            if verbose:
                print(f"[vram-llm] adjusted tensor_split (with early-layer compensation): {tensor_split}")

        # In row-split mode, llama.cpp may keep some allocations on main_gpu.
        # For heterogeneous GPUs, it's generally safer to keep the biggest-free GPU first (CUDA device 0),
        # which our CLI does by default via CUDA_VISIBLE_DEVICES ordering.
        main_gpu = 0

        # llama-cpp-python expects n_threads=0 to mean "auto" in some builds; if not, set explicitly.
        if n_threads <= 0:
            # conservative default: leave 2 cores for OS; if unknown, fall back to 8
            try:
                import os
                n_threads = max(1, (os.cpu_count() or 8) - 2)
            except Exception:
                n_threads = 8

        if verbose:
            print(f"[vram-llm] Loading model with:")
            print(f"  model_path: {model_path}")
            print(f"  n_gpu_layers: {n_gpu_layers}")
            print(f"  split_mode: {plan.split_mode} (const={split_mode_const})")
            print(f"  main_gpu: {main_gpu}")
            print(f"  tensor_split: {tensor_split if tensor_split else None}")
            print(f"  n_ctx: {n_ctx}, n_batch: {n_batch}")

        t0 = time.time()
        llama = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            split_mode=split_mode_const,
            main_gpu=main_gpu,
            tensor_split=tensor_split if tensor_split else None,
            use_mmap=use_mmap,
            use_mlock=use_mlock,
            seed=seed,
            verbose=verbose,
            chat_format=chat_format,
        )
        dt = time.time() - t0

        self._llama = llama
        self._loaded = LoadedModel(
            model_path=model_path,
            created_at=int(time.time()),
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_threads=n_threads,
            split_mode=plan.split_mode,
            tensor_split=plan.tensor_split,
            n_gpu_layers=n_gpu_layers,
        )
        if verbose:
            print(f"[vram-llm] model loaded in {dt:.2f}s: {model_path}")
        return self._loaded

    async def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        """OpenAI-ish /v1/chat/completions passthrough."""
        if self._llama is None:
            raise RuntimeError("No model loaded")

        async with self._lock:
            # llama-cpp-python already supports the OpenAI-like schema.
            return self._llama.create_chat_completion(**payload)

    async def completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        """OpenAI-ish /v1/completions passthrough."""
        if self._llama is None:
            raise RuntimeError("No model loaded")

        async with self._lock:
            return self._llama.create_completion(**payload)
