"""Combine block_S2_features.parquet + block_S3_features.parquet into block_features.parquet.

Usage:
    python scripts/combine_block_features.py
"""
from __future__ import annotations

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

S2_PATH = ARTIFACTS / "block_S2_features.parquet"
S3_PATH = ARTIFACTS / "block_S3_features.parquet"
OUT_PATH = ARTIFACTS / "block_features.parquet"


def main() -> int:
    for p in [S2_PATH, S3_PATH]:
        if not p.exists():
            print(f"ERROR: {p} not found. Run block_features.py for both S2 and S3 first.", flush=True)
            return 1

    print(f"[load] {S2_PATH.name}", flush=True)
    t0 = time.time()
    s2 = pl.read_parquet(S2_PATH)
    print(f"        {s2.height:,} rows, {s2.width} cols, {time.time() - t0:.1f}s", flush=True)

    print(f"[load] {S3_PATH.name}", flush=True)
    t0 = time.time()
    s3 = pl.read_parquet(S3_PATH)
    print(f"        {s3.height:,} rows, {s3.width} cols, {time.time() - t0:.1f}s", flush=True)

    print("[concat] vertical_relaxed ...", flush=True)
    t0 = time.time()
    final = pl.concat([s2, s3], how="vertical_relaxed")
    print(f"        {final.height:,} rows, {final.width} cols, {time.time() - t0:.1f}s", flush=True)

    print(f"[write] {OUT_PATH}", flush=True)
    final.write_parquet(OUT_PATH, compression="zstd", compression_level=3)
    print(f"[done] {OUT_PATH} ({final.height:,} pairs)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())