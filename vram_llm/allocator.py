from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import os

from .gpu import GPUInfo, parse_cuda_visible_devices


@dataclass(frozen=True)
class AllocationPlan:
    # Physical device indices (NVML) in the order we will expose them to CUDA
    cuda_visible_devices: list[int]
    # tensor_split weights in the same order as cuda_visible_devices
    tensor_split: list[float]
    # split_mode string: "layer" | "row" | "none"
    split_mode: str
    # per-GPU effective budget (MiB), after reserve
    budgets_mib: list[int]
    # reserve per GPU (MiB)
    reserve_mib: int


def _normalize_weights(weights: list[float]) -> list[float]:
    s = float(sum(weights))
    if s <= 0:
        return []
    return [w / s for w in weights]


def choose_device_order(
    gpus: list[GPUInfo],
    policy: str = "free_desc",
) -> list[int]:
    """Return physical GPU indices in desired order."""
    if policy == "index":
        return [g.index for g in sorted(gpus, key=lambda x: x.index)]
    if policy == "free_desc":
        return [g.index for g in sorted(gpus, key=lambda x: x.free_mib, reverse=True)]
    if policy == "total_desc":
        return [g.index for g in sorted(gpus, key=lambda x: x.total_mib, reverse=True)]
    raise ValueError(f"Unknown device order policy: {policy!r}")


def compute_tensor_split(
    gpus_in_cuda_order: list[GPUInfo],
    reserve_mib: int = 1024,
    min_budget_mib: int = 1024,
    strategy: str = "free",
    normalize: bool = False,
) -> tuple[list[float], list[int]]:
    """Compute llama.cpp tensor_split weights.

    - reserve_mib: keep this much VRAM free per GPU to avoid OOM / fragmentation.
    - min_budget_mib: GPUs with less than this budget are treated as 0-weight.
    - strategy: currently only 'free' (use free VRAM).
    - normalize: if True, return weights that sum to 1.0. Otherwise return raw budgets.
    """
    if strategy != "free":
        raise ValueError(f"Unknown split strategy: {strategy!r}")

    budgets: list[int] = []
    weights: list[float] = []

    for g in gpus_in_cuda_order:
        budget = int(g.free_mib) - int(reserve_mib)
        if budget < min_budget_mib:
            budget = 0
        budgets.append(budget)
        weights.append(float(max(budget, 0)))

    if normalize:
        weights = _normalize_weights(weights)

    return weights, budgets


def make_allocation_plan(
    gpus: list[GPUInfo],
    *,
    devices: Optional[list[int]] = None,
    device_order: str = "free_desc",
    reserve_mib: int = 1024,
    min_budget_mib: int = 1024,
    split_mode: str = "layer",
    normalize_split: bool = False,
    drop_zero_budget: bool = True,
) -> AllocationPlan:
    """Build an allocation plan from physical GPUs.

    If CUDA_VISIBLE_DEVICES is already set in the environment, we treat it as authoritative
    and won't reorder unless the caller passes an explicit `devices` list.
    """
    env_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env_cvd and devices is None:
        # Respect the user's explicit env var, but only support numeric format.
        env_indices = parse_cuda_visible_devices(env_cvd)
        devices = env_indices
        ordered = devices  # keep that order
    else:
        if devices is None:
            ordered = choose_device_order(gpus, policy=device_order)
        else:
            ordered = list(devices)

    g_map = {g.index: g for g in gpus}
    g_ordered = [g_map[i] for i in ordered if i in g_map]

    weights, budgets = compute_tensor_split(
        g_ordered,
        reserve_mib=reserve_mib,
        min_budget_mib=min_budget_mib,
        normalize=normalize_split,
    )

    if drop_zero_budget:
        keep = [i for i, b in enumerate(budgets) if b > 0]
        g_ordered = [g_ordered[i] for i in keep]
        weights = [weights[i] for i in keep]
        budgets = [budgets[i] for i in keep]

    if not g_ordered:
        raise RuntimeError(
            "No GPUs left after budgeting. "
            "Try lowering --reserve-mib or --min-budget-mib, or free up VRAM."
        )

    return AllocationPlan(
        cuda_visible_devices=[g.index for g in g_ordered],
        tensor_split=weights,
        split_mode=split_mode,
        budgets_mib=budgets,
        reserve_mib=reserve_mib,
    )


def apply_cuda_visible_devices(plan: AllocationPlan, *, override: bool = False) -> None:
    """Set CUDA_VISIBLE_DEVICES to match plan.cuda_visible_devices.

    Must be called **before importing llama_cpp** (or torch) to be fully reliable.
    """
    if (not override) and ("CUDA_VISIBLE_DEVICES" in os.environ):
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in plan.cuda_visible_devices)
