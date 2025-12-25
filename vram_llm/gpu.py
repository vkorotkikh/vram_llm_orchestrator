from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import os


@dataclass(frozen=True)
class GPUInfo:
    """Physical GPU as reported by NVML (preferred) or torch.cuda (fallback)."""
    index: int
    name: str
    total_mib: int
    free_mib: int
    used_mib: int
    uuid: Optional[str] = None


def _mib(bytes_: int) -> int:
    return int(bytes_ // (1024 * 1024))


def list_nvidia_gpus() -> list[GPUInfo]:
    """Return physical NVIDIA GPUs with memory stats.

    Prefers NVML because it reports *global* free/used memory, not just what your
    current process has allocated.
    """
    try:
        from pynvml import (
            nvmlInit,
            nvmlShutdown,
            nvmlDeviceGetCount,
            nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetName,
            nvmlDeviceGetUUID,
            nvmlDeviceGetMemoryInfo,
        )
    except Exception:
        return []

    gpus: list[GPUInfo] = []
    nvmlInit()
    try:
        count = nvmlDeviceGetCount()
        for idx in range(count):
            h = nvmlDeviceGetHandleByIndex(idx)
            name = nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            uuid = nvmlDeviceGetUUID(h)
            if isinstance(uuid, bytes):
                uuid = uuid.decode("utf-8", errors="replace")
            mem = nvmlDeviceGetMemoryInfo(h)
            gpus.append(
                GPUInfo(
                    index=idx,
                    name=str(name),
                    total_mib=_mib(int(mem.total)),
                    free_mib=_mib(int(mem.free)),
                    used_mib=_mib(int(mem.used)),
                    uuid=str(uuid),
                )
            )
    finally:
        try:
            nvmlShutdown()
        except Exception:
            pass
    return gpus


def list_cuda_gpus_fallback_torch() -> list[GPUInfo]:
    """Fallback GPU listing using torch.cuda, if available.

    This is less reliable for 'free' memory than NVML, but better than nothing.
    """
    try:
        import torch
    except Exception:
        return []

    if not torch.cuda.is_available():
        return []

    gpus: list[GPUInfo] = []
    for idx in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(idx)
        total_mib = _mib(int(prop.total_memory))
        # torch.cuda.mem_get_info reports free/total for current device
        try:
            torch.cuda.set_device(idx)
            free_b, _total_b = torch.cuda.mem_get_info()
            free_mib = _mib(int(free_b))
            used_mib = total_mib - free_mib
        except Exception:
            free_mib = 0
            used_mib = 0
        gpus.append(
            GPUInfo(
                index=idx,
                name=str(prop.name),
                total_mib=total_mib,
                free_mib=free_mib,
                used_mib=used_mib,
                uuid=None,
            )
        )
    return gpus


def list_gpus() -> list[GPUInfo]:
    gpus = list_nvidia_gpus()
    if gpus:
        return gpus
    return list_cuda_gpus_fallback_torch()


def parse_cuda_visible_devices(env_value: str) -> list[int]:
    """Parse CUDA_VISIBLE_DEVICES like "2,0,1".

    Notes:
    - This function only supports the common numeric form.
    - CUDA_VISIBLE_DEVICES can also contain UUIDs; we intentionally do not
      support that in this POC.
    """
    parts = [p.strip() for p in env_value.split(",") if p.strip()]
    out: list[int] = []
    for p in parts:
        if not p.isdigit():
            raise ValueError(f"CUDA_VISIBLE_DEVICES contains non-numeric entry: {p!r}")
        out.append(int(p))
    return out


def filter_gpus_by_physical_indices(gpus: Iterable[GPUInfo], indices: Optional[Iterable[int]]) -> list[GPUInfo]:
    if indices is None:
        return list(gpus)
    wanted = set(indices)
    return [g for g in gpus if g.index in wanted]


def gpus_markdown_table(gpus: list[GPUInfo]) -> str:
    if not gpus:
        return "(no CUDA GPUs detected)"

    header = "| idx | name | free MiB | used MiB | total MiB | uuid |\n|---:|---|---:|---:|---:|---|\n"
    rows = []
    for g in gpus:
        rows.append(f"| {g.index} | {g.name} | {g.free_mib} | {g.used_mib} | {g.total_mib} | {g.uuid or ''} |")
    return header + "\n".join(rows)
