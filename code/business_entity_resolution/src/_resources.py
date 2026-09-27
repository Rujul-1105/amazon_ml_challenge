"""RAM + core auto-detection. Call BEFORE any large allocation.

Every blocking / training script in this project imports this module first.
The detect_resources() helper reads /proc/meminfo and chooses a safe worker
pool size + polars thread count based on currently-available RAM.

Usage at the top of any script (BEFORE importing polars):

    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    from _resources import detect_resources, configure_polars_threads

    _RES = detect_resources()
    configure_polars_threads(_RES["polars_threads"])
    import polars as pl   # polars now respects the thread cap

Then later, for any worker pool:

    import multiprocessing as mp
    with mp.Pool(_RES["n_workers"]) as pool:
        ...

Worker cap policy:
    - hard_cap_workers default = 8 (allows parallel MinHash build).
    - On 8 GiB boxes, n_workers auto-reduces to 1-2 (1.5 GB/worker × 8
      = 12 GB headroom needed; not safe on small boxes).
    - On 32+ GiB boxes, n_workers = 8 by default.
    - On 128+ GiB boxes (like your VM), n_workers stays at 8 (polars is
      already multi-threaded internally; >8 workers rarely helps for the
      per-row MinHash loop due to GIL).
    - Override via env var MAX_WORKERS or pass to detect_resources().

Why we cap polars threads at n_workers * 2:
    - Polars is NUMA-unaware; doubling threads beyond physical cores gives
      no speedup but inflates peak memory due to per-thread arena allocators.
"""
from __future__ import annotations
import os
from typing import Optional


def _read_meminfo() -> dict:
    """Parse /proc/meminfo into a {label: int_kB} dict."""
    out: dict[str, int] = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, _, v = line.partition(":")
            if not v:
                continue
            parts = v.strip().split()
            try:
                out[k.strip()] = int(parts[0])  # kB
            except (ValueError, IndexError):
                pass
    return out


def detect_resources(
    min_free_gb: float = 6.0,
    hard_cap_workers: int = 8,
    per_worker_gb: float = 1.5,
    headroom_gb: float = 6.0,
) -> dict:
    """Detect available RAM + cores and choose safe worker counts.

    Parameters
    ----------
    min_free_gb : float
        Refuse to launch (raise RuntimeError) if available RAM is below this.
        Default 6.0 GiB is the minimum safe budget for any blocking run.
    hard_cap_workers : int
        Maximum number of worker processes. Default 8 — covers MinHash
        parallel build + rapidfuzz + polars arena.
    per_worker_gb : float
        RAM budget per worker. Default 1.5 GiB.
    headroom_gb : float
        RAM reserved for OS + polars internal arenas + Python overhead.
        Default 6.0 GiB.

    Returns
    -------
    dict with keys:
        free_gb        (float) — MemAvailable in GiB
        total_gb       (float) — MemTotal in GiB
        cores          (int)   — os.cpu_count()
        n_workers      (int)   — safe worker pool size, 1..hard_cap_workers
        polars_threads (int)   — safe thread count for polars
    """
    mem = _read_meminfo()
    avail_kb = mem.get("MemAvailable", mem.get("MemFree", 0))
    total_kb = mem.get("MemTotal", 0)
    free_gb = avail_kb / 1024 / 1024
    total_gb = total_kb / 1024 / 1024
    cores = os.cpu_count() or 4

    budget = max(0.0, free_gb - headroom_gb)
    n_workers = int(budget / per_worker_gb)
    n_workers = max(1, min(n_workers, hard_cap_workers, cores))

    # Polars threads: cap at cores but at least n_workers*2 so polars can
    # actually use the workers we provisioned.
    polars_threads = max(1, min(cores, max(n_workers * 2, 8)))

    if free_gb < min_free_gb:
        raise RuntimeError(
            f"Insufficient RAM: {free_gb:.1f} GiB available, "
            f"need >= {min_free_gb:.1f} GiB. Free memory and retry."
        )

    return {
        "free_gb": free_gb,
        "total_gb": total_gb,
        "cores": cores,
        "n_workers": n_workers,
        "polars_threads": polars_threads,
    }


def configure_polars_threads(threads: int) -> None:
    """Set POLARS_MAX_THREADS BEFORE polars is imported in the calling process.

    Polars reads this env var only at import time. If polars is already
    imported, this is a no-op (caller should re-import or restart).
    """
    os.environ["POLARS_MAX_THREADS"] = str(int(threads))


def log_resources(stage: str, res: Optional[dict] = None) -> dict:
    """Print current resources; return detect_resources() dict."""
    if res is None:
        res = detect_resources()
    print(
        f"[{stage}] RAM {res['free_gb']:.1f}/{res['total_gb']:.1f} GiB free, "
        f"{res['cores']} cores, n_workers={res['n_workers']}, "
        f"polars_threads={res['polars_threads']}"
    )
    return res


__all__ = ["detect_resources", "configure_polars_threads", "log_resources"]
