"""
GPU Detection and Memory Reporting Module.

This module provides utilities for detecting NVIDIA GPUs and reporting their
memory status (total, free, used VRAM). It serves as the foundational layer
for the VRAM LLM Orchestrator, enabling intelligent GPU memory allocation
decisions when loading and running Large Language Models (LLMs).

Key responsibilities:
    - Enumerate available NVIDIA GPUs using NVML (preferred) or PyTorch (fallback)
    - Report accurate global memory statistics (not just per-process allocations)
    - Parse and handle CUDA_VISIBLE_DEVICES environment variable
    - Filter GPUs by physical indices for multi-GPU setups
    - Generate human-readable GPU status reports

The module prioritizes NVML (NVIDIA Management Library) over PyTorch's CUDA
interface because NVML reports *system-wide* memory usage, which is critical
for accurate VRAM budgeting when other processes may be using GPU memory.

Typical usage:
    >>> from vram_llm.gpu import list_gpus, gpus_markdown_table
    >>> gpus = list_gpus()
    >>> print(gpus_markdown_table(gpus))

Dependencies:
    - pynvml (optional, preferred): For accurate system-wide GPU memory stats
    - torch (optional, fallback): Used when pynvml is unavailable
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, List

import os


@dataclass(frozen=True)
class GPUInfo:
    """Immutable representation of a physical NVIDIA GPU and its memory state.
    
    This dataclass captures a snapshot of GPU information as reported by either
    NVML (preferred) or torch.cuda (fallback). The frozen=True ensures instances
    are hashable and can be used in sets or as dictionary keys.
    
    Attributes:
        index: Physical GPU index as reported by the driver (0-based).
               This corresponds to the device enumeration order in nvidia-smi.
        name: Human-readable GPU model name (e.g., "NVIDIA GeForce RTX 4090").
        total_mib: Total GPU memory in Mebibytes (MiB = 1024² bytes).
        free_mib: Currently available GPU memory in MiB.
                  Note: NVML reports global free memory; torch may only see
                  memory not allocated by the current process.
        used_mib: Currently used GPU memory in MiB (total - free).
        uuid: Optional unique identifier for the GPU (NVML only).
              Format: "GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
              Useful for identifying specific physical devices across reboots.
    
    Example:
        >>> gpu = GPUInfo(
        ...     index=0,
        ...     name="NVIDIA GeForce RTX 4090",
        ...     total_mib=24564,
        ...     free_mib=22000,
        ...     used_mib=2564,
        ...     uuid="GPU-12345678-1234-1234-1234-123456789abc"
        ... )
        >>> gpu.free_mib
        22000
    """
    index: int
    name: str
    total_mib: int
    free_mib: int
    used_mib: int
    uuid: Optional[str] = None


def _mib(bytes_: int) -> int:
    """Convert bytes to Mebibytes (MiB).
    
    Uses integer division for consistent, floor-rounded results.
    1 MiB = 1024 * 1024 = 1,048,576 bytes.
    
    Args:
        bytes_: Memory size in bytes.
    
    Returns:
        Memory size in MiB (rounded down to nearest integer).
    
    Example:
        >>> _mib(1073741824)  # 1 GiB in bytes
        1024
    """
    return int(bytes_ // (1024 * 1024))


def list_nvidia_gpus() -> List[GPUInfo]:
    """Enumerate NVIDIA GPUs using NVML (NVIDIA Management Library).
    
    This is the preferred method for GPU detection because NVML reports
    *global* system-wide memory usage, not just allocations from the current
    process. This is critical for accurate VRAM budgeting when other
    applications (browsers, other ML workloads, etc.) may be using GPU memory.
    
    The function handles the NVML lifecycle internally:
        1. Initialize NVML
        2. Query all available GPUs
        3. Shutdown NVML (cleanup, even on errors)
    
    Returns:
        List of GPUInfo objects for each detected NVIDIA GPU.
        Returns an empty list if:
            - pynvml is not installed
            - No NVIDIA GPUs are present
            - NVML initialization fails
    
    Note:
        GPU names and UUIDs may be returned as bytes by older pynvml versions;
        this function handles the decoding automatically.
    
    Example:
        >>> gpus = list_nvidia_gpus()
        >>> if gpus:
        ...     print(f"Found {len(gpus)} GPU(s)")
        ...     print(f"GPU 0 has {gpus[0].free_mib} MiB free")
    """
    # Attempt to import pynvml; return empty list if unavailable
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
        # pynvml not installed or import failed
        return []

    gpus: List[GPUInfo] = []
    
    # Initialize NVML library (required before any NVML calls)
    nvmlInit()
    try:
        count = nvmlDeviceGetCount()
        
        for idx in range(count):
            # Get device handle for this GPU index
            h = nvmlDeviceGetHandleByIndex(idx)
            
            # Get GPU name (may be bytes in older pynvml versions)
            name = nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            
            # Get GPU UUID for unique identification
            uuid = nvmlDeviceGetUUID(h)
            if isinstance(uuid, bytes):
                uuid = uuid.decode("utf-8", errors="replace")
            
            # Get memory info (total, free, used in bytes)
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
        # Always attempt to shutdown NVML to release resources
        try:
            nvmlShutdown()
        except Exception:
            pass  # Ignore shutdown errors; we've already collected the data
    
    return gpus


def list_cuda_gpus_fallback_torch() -> List[GPUInfo]:
    """Enumerate CUDA GPUs using PyTorch as a fallback.
    
    This fallback method is used when NVML (pynvml) is unavailable. While
    functional, it has limitations compared to NVML:
    
    Limitations:
        - Memory stats may only reflect the current process's view
        - Free memory may be inaccurate if other CUDA contexts exist
        - No UUID support (returns None)
        - Requires PyTorch with CUDA support
    
    Returns:
        List of GPUInfo objects for each detected CUDA GPU.
        Returns an empty list if:
            - PyTorch is not installed
            - CUDA is not available
            - No CUDA-capable GPUs are detected
    
    Note:
        This function temporarily sets the current CUDA device to query
        memory info for each GPU, which may have minor side effects if
        called from within a CUDA context.
    
    Example:
        >>> gpus = list_cuda_gpus_fallback_torch()
        >>> for gpu in gpus:
        ...     print(f"{gpu.name}: {gpu.free_mib} MiB free")
    """
    # Attempt to import torch; return empty list if unavailable
    try:
        import torch
    except Exception:
        return []

    # Check if CUDA is available on this system
    if not torch.cuda.is_available():
        return []

    gpus: List[GPUInfo] = []
    
    for idx in range(torch.cuda.device_count()):
        # Get device properties (name, total memory, etc.)
        prop = torch.cuda.get_device_properties(idx)
        total_mib = _mib(int(prop.total_memory))
        
        # Query current free/total memory for this device
        # Note: Requires temporarily switching the current device
        try:
            torch.cuda.set_device(idx)
            free_b, _total_b = torch.cuda.mem_get_info()
            free_mib = _mib(int(free_b))
            used_mib = total_mib - free_mib
        except Exception:
            # If memory query fails, report zeros rather than crashing
            free_mib = 0
            used_mib = 0
        
        gpus.append(
            GPUInfo(
                index=idx,
                name=str(prop.name),
                total_mib=total_mib,
                free_mib=free_mib,
                used_mib=used_mib,
                uuid=None,  # PyTorch doesn't expose GPU UUID
            )
        )
    
    return gpus


def list_gpus() -> List[GPUInfo]:
    """Detect available NVIDIA/CUDA GPUs with memory statistics.
    
    This is the main entry point for GPU detection. It tries NVML first
    (via list_nvidia_gpus) for accurate system-wide memory stats, then
    falls back to PyTorch's CUDA interface if NVML is unavailable.
    
    Detection order:
        1. NVML (pynvml) - Preferred for accurate global memory stats
        2. PyTorch CUDA - Fallback when pynvml is not installed
    
    Returns:
        List of GPUInfo objects representing all detected GPUs.
        Returns an empty list if no GPUs are detected by either method.
    
    Example:
        >>> gpus = list_gpus()
        >>> if not gpus:
        ...     print("No GPUs detected!")
        ... else:
        ...     total_free = sum(g.free_mib for g in gpus)
        ...     print(f"Total free VRAM: {total_free} MiB")
    """
    # Try NVML first (preferred for accurate memory reporting)
    gpus = list_nvidia_gpus()
    if gpus:
        return gpus
    
    # Fall back to PyTorch CUDA if NVML unavailable
    return list_cuda_gpus_fallback_torch()


def parse_cuda_visible_devices(env_value: str) -> List[int]:
    """Parse CUDA_VISIBLE_DEVICES environment variable value.
    
    Converts a comma-separated string of GPU indices into a list of integers.
    This is used to respect user-specified GPU restrictions when planning
    memory allocation.
    
    Args:
        env_value: The value of CUDA_VISIBLE_DEVICES (e.g., "0,2,1" or "0").
    
    Returns:
        List of physical GPU indices in the order specified.
    
    Raises:
        ValueError: If the string contains non-numeric entries.
    
    Limitations:
        - Only supports numeric GPU indices (the common case)
        - Does NOT support UUID-based device specification
          (e.g., "GPU-12345678-1234-1234-1234-123456789abc")
        - This is intentional for this POC to keep parsing simple
    
    Examples:
        >>> parse_cuda_visible_devices("0,1,2")
        [0, 1, 2]
        >>> parse_cuda_visible_devices("2,0,1")
        [2, 0, 1]
        >>> parse_cuda_visible_devices("0")
        [0]
        >>> parse_cuda_visible_devices("")
        []
    """
    # Split by comma and strip whitespace from each part
    parts = [p.strip() for p in env_value.split(",") if p.strip()]
    
    out: List[int] = []
    for p in parts:
        # Validate that each part is a non-negative integer
        if not p.isdigit():
            raise ValueError(f"CUDA_VISIBLE_DEVICES contains non-numeric entry: {p!r}")
        out.append(int(p))
    
    return out


def filter_gpus_by_physical_indices(
    gpus: Iterable[GPUInfo],
    indices: Optional[Iterable[int]]
) -> List[GPUInfo]:
    """Filter a collection of GPUs to only those with specified physical indices.
    
    This is useful when you want to restrict operations to a subset of
    available GPUs, such as when respecting CUDA_VISIBLE_DEVICES or when
    the user explicitly specifies which GPUs to use.
    
    Args:
        gpus: Iterable of GPUInfo objects to filter.
        indices: Physical GPU indices to keep. If None, all GPUs are returned
                 (no filtering applied).
    
    Returns:
        List of GPUInfo objects whose index is in the specified set.
        Order is preserved from the input iterable.
    
    Examples:
        >>> all_gpus = [GPUInfo(index=0, ...), GPUInfo(index=1, ...), GPUInfo(index=2, ...)]
        >>> filter_gpus_by_physical_indices(all_gpus, [0, 2])
        [GPUInfo(index=0, ...), GPUInfo(index=2, ...)]
        >>> filter_gpus_by_physical_indices(all_gpus, None)  # No filtering
        [GPUInfo(index=0, ...), GPUInfo(index=1, ...), GPUInfo(index=2, ...)]
    """
    # If no indices specified, return all GPUs (no-op filter)
    if indices is None:
        return list(gpus)
    
    # Convert to set for O(1) membership testing
    wanted = set(indices)
    return [g for g in gpus if g.index in wanted]


def gpus_markdown_table(gpus: List[GPUInfo]) -> str:
    """Generate a Markdown-formatted table of GPU information.
    
    Creates a human-readable table suitable for display in terminals,
    documentation, or Markdown-compatible interfaces. Useful for debugging
    and status reporting.
    
    Args:
        gpus: List of GPUInfo objects to display.
    
    Returns:
        A string containing a Markdown table with columns:
            - idx: Physical GPU index
            - name: GPU model name
            - free MiB: Available VRAM
            - used MiB: Used VRAM
            - total MiB: Total VRAM
            - uuid: GPU UUID (if available)
        
        Returns "(no CUDA GPUs detected)" if the list is empty.
    
    Example output:
        | idx | name | free MiB | used MiB | total MiB | uuid |
        |---:|---|---:|---:|---:|---|
        | 0 | NVIDIA GeForce RTX 4090 | 22000 | 2564 | 24564 | GPU-1234... |
        | 1 | NVIDIA GeForce RTX 3080 | 9000 | 1000 | 10000 | GPU-5678... |
    """
    # Handle empty GPU list
    if not gpus:
        return "(no CUDA GPUs detected)"

    # Build table header with column alignment hints
    # (right-align numeric columns, left-align text columns)
    header = "| idx | name | free MiB | used MiB | total MiB | uuid |\n|---:|---|---:|---:|---:|---|\n"
    
    # Build table rows
    rows = []
    for g in gpus:
        # Use empty string for None UUID to keep table clean
        uuid_str = g.uuid or ''
        rows.append(f"| {g.index} | {g.name} | {g.free_mib} | {g.used_mib} | {g.total_mib} | {uuid_str} |")
    
    return header + "\n".join(rows)
