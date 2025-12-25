from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class FitParams:
    n_ctx: Optional[int]
    n_gpu_layers: Optional[int]
    tensor_split: Optional[list[float]]
    raw_cli: str
    stdout: str
    stderr: str


_N_CTX_RE = re.compile(r"(?:^|\s)(?:-c|--ctx-size)\s+(\d+)(?:\s|$)")
_NGL_RE = re.compile(r"(?:^|\s)(?:-ngl|--n-gpu-layers)\s+(\d+)(?:\s|$)")
_TS_RE = re.compile(r"(?:^|\s)(?:-ts|--tensor-split)\s+([0-9eE+.,-]+)(?:\s|$)")


def _parse_tensor_split(ts: str) -> list[float]:
    parts = [p.strip() for p in ts.split(",") if p.strip()]
    out: list[float] = []
    for p in parts:
        out.append(float(p))
    return out


def parse_fit_params_output(text: str) -> Optional[FitParams]:
    # Heuristic: find the last line that looks like a CLI arg string and contains -ngl or -ts.
    candidate_lines = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("-")]
    if not candidate_lines:
        return None

    # Prefer a line containing -ngl
    cli = None
    for ln in reversed(candidate_lines):
        if "-ngl" in ln or "--n-gpu-layers" in ln:
            cli = ln
            break
    if cli is None:
        cli = candidate_lines[-1]

    m_ctx = _N_CTX_RE.search(cli)
    m_ngl = _NGL_RE.search(cli)
    m_ts = _TS_RE.search(cli)

    n_ctx = int(m_ctx.group(1)) if m_ctx else None
    n_gpu_layers = int(m_ngl.group(1)) if m_ngl else None
    tensor_split = _parse_tensor_split(m_ts.group(1)) if m_ts else None

    return FitParams(
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        tensor_split=tensor_split,
        raw_cli=cli,
        stdout=text,
        stderr="",
    )


def run_llama_fit_params(
    *,
    model_path: str,
    n_ctx: Optional[int] = None,
    n_batch: Optional[int] = None,
    n_ubatch: Optional[int] = None,
    split_mode: Optional[str] = None,
    fit_target_mib: Optional[int] = None,
    binary: Optional[str] = None,
    timeout_s: int = 120,
) -> FitParams:
    """Run llama-fit-params if available and parse suggested args.

    This tool is very new upstream and its CLI has been in flux. We try a few
    common invocation patterns and accept the first one that succeeds.
    """
    bin_path = binary or shutil.which("llama-fit-params")
    if not bin_path:
        raise FileNotFoundError("llama-fit-params not found in PATH (set --llama-fit-params-bin)")

    # Try a small set of CLI patterns.
    base_args_sets = [
        ["--model", model_path],
        ["-m", model_path],
    ]

    extra: list[str] = []
    if n_ctx is not None:
        extra += ["-c", str(int(n_ctx))]
    if n_batch is not None:
        extra += ["-b", str(int(n_batch))]
    if n_ubatch is not None:
        extra += ["-ub", str(int(n_ubatch))]
    if split_mode is not None:
        extra += ["-sm", str(split_mode)]
    if fit_target_mib is not None:
        extra += ["--fit-target", str(int(fit_target_mib))]

    last_err = None
    for base_args in base_args_sets:
        cmd = [bin_path] + base_args + extra
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        except Exception as e:
            last_err = e
            continue

        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if proc.returncode != 0:
            last_err = RuntimeError(f"llama-fit-params failed: rc={proc.returncode}, cmd={cmd}, out=\n{out}")
            continue

        parsed = parse_fit_params_output(out)
        if parsed is None:
            last_err = RuntimeError(f"Could not parse llama-fit-params output. cmd={cmd}, out=\n{out}")
            continue

        return FitParams(
            n_ctx=parsed.n_ctx,
            n_gpu_layers=parsed.n_gpu_layers,
            tensor_split=parsed.tensor_split,
            raw_cli=parsed.raw_cli,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
        )

    raise RuntimeError(f"Unable to run llama-fit-params successfully. Last error: {last_err}")
