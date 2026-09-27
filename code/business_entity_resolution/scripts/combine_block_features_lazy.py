"""RAM-safe combine of N block_*.parquet files.

Uses polars lazy evaluation + sink_parquet (streaming write) so peak RAM stays
under ~500 MB regardless of input size. Original `combine_block_features.py`
loads both into eager DataFrames (~3x disk size each) then concats — ~15-18 GB
peak for the current train data, which OOMs on an 8-16 GB box.

Output writes to `artifacts/block_features.parquet` (same path the original
script uses), so downstream callers don't need any change.

Usage:
    # Default (S2 + S3):
    python scripts/combine_block_features_lazy.py

    # Custom inputs:
    python scripts/combine_block_features_lazy.py \
        --inputs artifacts/block_S2_features.parquet \
                 artifacts/block_S3_features.parquet \
        --output artifacts/block_features.parquet
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
from pathlib import Path

if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "artifacts"

DEFAULT_INPUTS = [
    ARTIFACTS / "block_S2_features.parquet",
    ARTIFACTS / "block_S3_features.parquet",
]
DEFAULT_OUTPUT = ARTIFACTS / "block_features.parquet"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--inputs", nargs="+", default=None,
                   help=f"Input parquet paths (default: {len(DEFAULT_INPUTS)} standard paths)")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                   help="Output combined parquet path")
    args = p.parse_args()

    inputs = [Path(p) for p in args.inputs] if args.inputs else DEFAULT_INPUTS

    for p in inputs:
        if not p.exists():
            print(f"ERROR: {p} not found. Run block_features.py for the missing source first.", flush=True)
            return 1

    lazy_frames = []
    for p in inputs:
        print(f"[scan] {p.name} (lazy)", flush=True)
        lazy_frames.append(pl.scan_parquet(p))

    print(f"[concat] vertical_relaxed across {len(lazy_frames)} inputs (lazy)", flush=True)
    t0 = time.time()
    combined = pl.concat(lazy_frames, how="vertical_relaxed")

    print(f"[sink]  writing to {args.output.name} (streaming, zstd)", flush=True)
    t1 = time.time()

    combined.sink_parquet(
        args.output,
        compression="zstd",
        compression_level=3,
        row_group_size=262144,
    )

    out_size_gb = args.output.stat().st_size / (1024 ** 3)
    n_rows = pl.scan_parquet(args.output).select(pl.len()).collect(engine="streaming").item()

    print(f"        sink: {time.time() - t1:.1f}s", flush=True)
    print(f"        total: {time.time() - t0:.1f}s", flush=True)
    print(f"        wrote {n_rows:,} pairs, {out_size_gb:.2f} GB on disk", flush=True)
    print(f"        path: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
