from __future__ import annotations

import argparse
import os
from dataclasses import asdict, replace
from typing import Optional

import uvicorn

from .allocator import apply_cuda_visible_devices, make_allocation_plan
from .api import create_app
from .engines.llama_cpp_engine import LlamaCppEngine
from .gpu import gpus_markdown_table, list_gpus


def _parse_int_list(csv: Optional[str]) -> Optional[list[int]]:
    if not csv:
        return None
    out: list[int] = []
    for p in csv.split(","):
        p = p.strip()
        if not p:
            continue
        if not p.isdigit():
            raise argparse.ArgumentTypeError(f"Expected comma-separated ints, got: {csv!r}")
        out.append(int(p))
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vram-llm", description="VRAM-aware multi-GPU llama.cpp backend (POC).")
    sub = p.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="Start the HTTP API server")
    serve.add_argument("--model", "--model-path", dest="model_path", required=True, help="Path to a GGUF model file")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)

    serve.add_argument("--n-ctx", type=int, default=4096)
    serve.add_argument("--n-batch", type=int, default=512)
    serve.add_argument("--n-ubatch", type=int, default=None, help="Optional micro-batch size (passed to fit tool only)")
    serve.add_argument("--n-threads", type=int, default=0)
    serve.add_argument("--n-gpu-layers", type=int, default=-1, help="-1 = try offload all layers")

    serve.add_argument("--split-mode", choices=["layer", "row", "none"], default="layer")

    serve.add_argument("--reserve-mib", type=int, default=1024, help="VRAM reserve per GPU (MiB)")
    serve.add_argument("--min-budget-mib", type=int, default=1024, help="Minimum usable VRAM budget per GPU (MiB)")
    serve.add_argument("--device-order", choices=["free_desc", "total_desc", "index"], default="free_desc")
    serve.add_argument(
        "--devices",
        type=str,
        default=None,
        help="Comma-separated physical GPU indices to use (e.g. '0,2'). If omitted, use all GPUs.",
    )

    serve.add_argument(
        "--no-set-cuda-visible-devices",
        action="store_true",
        help="Do not modify CUDA_VISIBLE_DEVICES (respect current environment).",
    )
    serve.add_argument(
        "--override-cuda-visible-devices",
        action="store_true",
        help="Override CUDA_VISIBLE_DEVICES even if already set.",
    )

    # Optional: use llama-fit-params to compute n_gpu_layers / tensor_split using llama.cpp's own memory model.
    serve.add_argument(
        "--use-fit-params",
        action="store_true",
        help="If llama-fit-params is available, use it to compute n_gpu_layers/tensor_split. Experimental.",
    )
    serve.add_argument(
        "--llama-fit-params-bin",
        type=str,
        default=None,
        help="Path to llama-fit-params binary (if not in PATH).",
    )
    serve.add_argument(
        "--fit-target-mib",
        type=int,
        default=None,
        help="Target free VRAM per GPU for fit tool (MiB). Optional and may be ignored by some versions.",
    )

    serve.add_argument("--dry-run", action="store_true", help="Print allocation plan and exit without serving.")
    serve.add_argument("--chat-format", default=None, help="Optional llama.cpp chat format override.")

    plan = sub.add_parser("plan", help="Print computed GPU allocation plan and exit")
    plan.add_argument("--split-mode", choices=["layer", "row", "none"], default="layer")
    plan.add_argument("--reserve-mib", type=int, default=1024)
    plan.add_argument("--min-budget-mib", type=int, default=1024)
    plan.add_argument("--device-order", choices=["free_desc", "total_desc", "index"], default="free_desc")
    plan.add_argument("--devices", type=str, default=None)

    return p


def cmd_plan(args: argparse.Namespace) -> int:
    gpus = list_gpus()
    print("[vram-llm] detected GPUs:")
    print(gpus_markdown_table(gpus))
    devices = _parse_int_list(args.devices)

    plan = make_allocation_plan(
        gpus,
        devices=devices,
        device_order=args.device_order,
        reserve_mib=args.reserve_mib,
        min_budget_mib=args.min_budget_mib,
        split_mode=args.split_mode,
    )
    print("\n[vram-llm] allocation plan:")
    print(asdict(plan))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    gpus = list_gpus()
    if not gpus:
        raise SystemExit("No CUDA GPUs detected. This POC targets multi-GPU CUDA systems.")

    print("[vram-llm] detected GPUs:")
    print(gpus_markdown_table(gpus))

    devices = _parse_int_list(args.devices)

    plan = make_allocation_plan(
        gpus,
        devices=devices,
        device_order=args.device_order,
        reserve_mib=args.reserve_mib,
        min_budget_mib=args.min_budget_mib,
        split_mode=args.split_mode,
    )

    print("\n[vram-llm] computed allocation plan (NVML-based heuristic):")
    print(asdict(plan))

    if not args.no_set_cuda_visible_devices:
        apply_cuda_visible_devices(plan, override=args.override_cuda_visible_devices)
        print(f"\n[vram-llm] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    plan_for_model = plan
    n_gpu_layers = args.n_gpu_layers

    if args.use_fit_params:
        from .llama_fit_params import run_llama_fit_params

        try:
            fit = run_llama_fit_params(
                model_path=args.model_path,
                n_ctx=args.n_ctx,
                n_batch=args.n_batch,
                n_ubatch=args.n_ubatch,
                split_mode=args.split_mode,
                fit_target_mib=args.fit_target_mib,
                binary=args.llama_fit_params_bin,
            )
            print("\n[vram-llm] llama-fit-params suggestion:")
            print(fit.raw_cli)
            if fit.tensor_split:
                plan_for_model = replace(plan_for_model, tensor_split=fit.tensor_split)
            if fit.n_gpu_layers is not None:
                n_gpu_layers = int(fit.n_gpu_layers)
        except Exception as e:
            print(f"\n[vram-llm] WARNING: fit-params requested but failed: {e}")
            print("[vram-llm] Falling back to NVML heuristic split.")

    if args.dry_run:
        print("\n[vram-llm] dry-run: exiting.")
        return 0

    engine = LlamaCppEngine()
    engine.load(
        model_path=args.model_path,
        plan=plan_for_model,
        n_ctx=args.n_ctx,
        n_batch=args.n_batch,
        n_threads=args.n_threads,
        n_gpu_layers=n_gpu_layers,
        chat_format=args.chat_format,
        verbose=True,
    )

    app = create_app(engine=engine, plan=plan_for_model, gpus=gpus)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "plan":
        raise SystemExit(cmd_plan(args))
    if args.cmd == "serve":
        raise SystemExit(cmd_serve(args))

    raise SystemExit(f"Unknown command: {args.cmd}")
