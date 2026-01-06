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
    
    # Memory management options
    serve.add_argument(
        "--no-mmap",
        action="store_true",
        help="Disable memory-mapped file loading. Reduces system RAM usage but may be slower.",
    )
    serve.add_argument(
        "--mlock",
        action="store_true",
        help="Lock model in RAM (prevent swapping). Requires sufficient RAM.",
    )
    serve.add_argument(
        "--tensor-split",
        type=str,
        default=None,
        help="Manual tensor split proportions (comma-separated floats, e.g. '0.35,0.40,0.25'). Overrides auto-calculation.",
    )
    serve.add_argument(
        "--smart-split",
        action="store_true",
        help="Use GGUF model analysis to compute optimal layer distribution (slower startup, better allocation).",
    )

    plan = sub.add_parser("plan", help="Print computed GPU allocation plan and exit")
    plan.add_argument("--split-mode", choices=["layer", "row", "none"], default="layer")
    plan.add_argument("--reserve-mib", type=int, default=1024)
    plan.add_argument("--min-budget-mib", type=int, default=1024)
    plan.add_argument("--device-order", choices=["free_desc", "total_desc", "index"], default="free_desc")
    plan.add_argument("--devices", type=str, default=None)

    # Analyze command - analyze a model and show recommended tensor_split
    analyze = sub.add_parser("analyze", help="Analyze a GGUF model and show recommended GPU allocation")
    analyze.add_argument("--model", "--model-path", dest="model_path", required=True, help="Path to a GGUF model file")
    analyze.add_argument("--reserve-mib", type=int, default=1024, help="VRAM reserve per GPU (MiB)")

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


def cmd_analyze(args: argparse.Namespace) -> int:
    """Analyze a GGUF model and show recommended GPU allocation."""
    from .model_analyzer import (
        analyze_gguf_model,
        print_model_analysis,
        get_gpus_sorted_by_vram,
        allocate_layers_sequential,
        print_allocation_plan,
        TorchGPUStats,
    )
    
    # Get GPUs via NVML (always available)
    nvml_gpus = list_gpus()
    print("[vram-llm] detected GPUs (NVML):")
    print(gpus_markdown_table(nvml_gpus))
    
    print(f"\n[vram-llm] Analyzing model: {args.model_path}")
    
    try:
        analysis = analyze_gguf_model(args.model_path)
        print_model_analysis(analysis)
        
        # Try PyTorch GPU detection first, fall back to NVML
        sorted_gpus = get_gpus_sorted_by_vram(by_free=True)
        
        if not sorted_gpus and nvml_gpus:
            # Fallback: Convert NVML GPUs to TorchGPUStats format and sort by free VRAM
            print("\n[vram-llm] PyTorch not available, using NVML GPU info...")
            nvml_sorted = sorted(nvml_gpus, key=lambda g: g.free_mib, reverse=True)
            sorted_gpus = [
                TorchGPUStats(
                    index=g.index,
                    name=g.name,
                    total_mib=g.total_mib,
                    free_mib=g.free_mib,
                    used_mib=g.used_mib,
                    reordered_index=i,
                )
                for i, g in enumerate(nvml_sorted)
            ]
        
        if sorted_gpus:
            print(f"\n[vram-llm] GPUs sorted by free VRAM (GPU0 = highest):")
            for g in sorted_gpus:
                print(f"  GPU {g.reordered_index} (CUDA:{g.index}): {g.name} - "
                      f"{g.free_mib:,} / {g.total_mib:,} MiB")
            
            # Calculate layer allocation using sorted GPU budgets
            gpu_budgets = [g.free_mib for g in sorted_gpus]
            plan = allocate_layers_sequential(
                analysis,
                gpu_budgets,
                reserve_mib=args.reserve_mib,
                gpus=sorted_gpus,
            )
            
            # Print detailed allocation plan
            print_allocation_plan(plan, analysis)
            
            # Check if model fits
            total_model_gb = analysis.total_size_bytes / (1024**3)
            total_vram_gb = sum(gpu_budgets) / 1024
            usable_vram_gb = (sum(gpu_budgets) - args.reserve_mib * len(sorted_gpus)) / 1024
            
            print(f"  Model size: {total_model_gb:.1f} GB")
            print(f"  Total VRAM: {total_vram_gb:.1f} GB")
            print(f"  Usable VRAM (after reserve): {usable_vram_gb:.1f} GB")
            
            if total_model_gb > usable_vram_gb * 0.95:
                print(f"\n  ⚠️  WARNING: Model is close to or exceeds usable VRAM!")
                print(f"     Consider: smaller quantization, fewer layers (--n-gpu-layers), or smaller context (--n-ctx)")
        else:
            print("\n[vram-llm] No GPUs detected. Cannot compute allocation.")
        
    except Exception as e:
        import traceback
        print(f"[vram-llm] ERROR: Failed to analyze model: {e}")
        traceback.print_exc()
        return 1
    
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

    # Allow manual tensor_split override
    if args.tensor_split:
        try:
            manual_split = [float(x.strip()) for x in args.tensor_split.split(",") if x.strip()]
            if len(manual_split) != len(plan.cuda_visible_devices):
                print(f"[vram-llm] WARNING: --tensor-split has {len(manual_split)} values but {len(plan.cuda_visible_devices)} GPUs detected. Using auto-calculation.")
            else:
                plan_for_model = replace(plan_for_model, tensor_split=manual_split)
                print(f"[vram-llm] Using manual tensor_split: {manual_split}")
        except ValueError as e:
            print(f"[vram-llm] WARNING: Failed to parse --tensor-split: {e}. Using auto-calculation.")
    
    # Use smart allocation based on actual model layer sizes
    elif args.smart_split:
        try:
            from .model_analyzer import (
                analyze_gguf_model,
                print_model_analysis,
                get_gpus_sorted_by_vram,
                allocate_layers_sequential,
                print_allocation_plan,
                TorchGPUStats,
            )
            
            print(f"\n[vram-llm] Analyzing model for smart layer-aware allocation...")
            analysis = analyze_gguf_model(args.model_path)
            print_model_analysis(analysis)
            
            # Get GPUs sorted by VRAM (highest first) - try PyTorch first, fallback to NVML
            sorted_gpus = get_gpus_sorted_by_vram(by_free=True)
            
            if not sorted_gpus and gpus:
                # Fallback: Convert NVML GPUs to TorchGPUStats format
                print("[vram-llm] PyTorch not available, using NVML GPU info...")
                nvml_sorted = sorted(gpus, key=lambda g: g.free_mib, reverse=True)
                sorted_gpus = [
                    TorchGPUStats(
                        index=g.index,
                        name=g.name,
                        total_mib=g.total_mib,
                        free_mib=g.free_mib,
                        used_mib=g.used_mib,
                        reordered_index=i,
                    )
                    for i, g in enumerate(nvml_sorted)
                ]
            
            if sorted_gpus:
                # Use sorted GPU budgets for allocation
                gpu_budgets = [g.free_mib for g in sorted_gpus]
                
                allocation_plan = allocate_layers_sequential(
                    analysis,
                    gpu_budgets,
                    reserve_mib=args.reserve_mib,
                    gpus=sorted_gpus,
                )
                
                print_allocation_plan(allocation_plan, analysis)
                
                # Update the plan with computed tensor_split
                plan_for_model = replace(plan_for_model, tensor_split=allocation_plan.tensor_split)
                
                # Also update CUDA_VISIBLE_DEVICES to match the sorted order
                sorted_cuda_devices = [g.index for g in sorted_gpus]
                if sorted_cuda_devices != plan.cuda_visible_devices:
                    print(f"[vram-llm] Reordering GPUs by VRAM: {sorted_cuda_devices}")
                    plan_for_model = replace(plan_for_model, cuda_visible_devices=sorted_cuda_devices)
                    # Re-apply CUDA_VISIBLE_DEVICES with new order
                    if not args.no_set_cuda_visible_devices:
                        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in sorted_cuda_devices)
                        print(f"[vram-llm] Updated CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
            else:
                print("[vram-llm] WARNING: No GPUs detected for smart split.")
                print("[vram-llm] Falling back to VRAM-based heuristic.")
            
        except Exception as e:
            import traceback
            print(f"\n[vram-llm] WARNING: Smart split failed: {e}")
            traceback.print_exc()
            print("[vram-llm] Falling back to VRAM-based heuristic.")

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
    
    use_mmap = not args.no_mmap
    if not use_mmap:
        print("[vram-llm] mmap disabled - loading directly to GPU (lower RAM usage)")
    
    # Skip tensor_split adjustment if using smart-split (already accounts for layer sizes)
    skip_adjustment = args.smart_split or (args.tensor_split is not None)
    
    engine.load(
        model_path=args.model_path,
        plan=plan_for_model,
        n_ctx=args.n_ctx,
        n_batch=args.n_batch,
        n_threads=args.n_threads,
        n_gpu_layers=n_gpu_layers,
        use_mmap=use_mmap,
        use_mlock=args.mlock,
        chat_format=args.chat_format,
        verbose=True,
        skip_tensor_split_adjustment=skip_adjustment,
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
    if args.cmd == "analyze":
        raise SystemExit(cmd_analyze(args))

    raise SystemExit(f"Unknown command: {args.cmd}")
