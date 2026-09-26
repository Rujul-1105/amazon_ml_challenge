"""RAM-safe combine of block_S2_features.parquet + block_S3_features.parquet.

Uses polars lazy evaluation + sink_parquet (streaming write) so peak RAM stays
under ~500 MB regardless of input size. Original `combine_block_features.py`
loads both into eager DataFrames (~3x disk size each) then concats — ~15-18 GB
peak for the current train data, which OOMs on an 8-16 GB box.

Output schema matches the v3 design (40 cols). Writes to
`artifacts/block_features.parquet` (same path the original script uses), so
downstream callers don't need any change.

Usage:
    python scripts/combine_block_features_lazy.py
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
    for p in (S2_PATH, S3_PATH):
        if not p.exists():
            print(f"ERROR: {p} not found. Run block_features.py for both S2 and S3 first.", flush=True)
            return 1

    print(f"[scan] {S2_PATH.name} (lazy)", flush=True)
    s2 = pl.scan_parquet(S2_PATH)
    print(f"[scan] {S3_PATH.name} (lazy)", flush=True)
    s3 = pl.scan_parquet(S3_PATH)

    print("[concat] vertical_relaxed (lazy)", flush=True)
    t0 = time.time()
    combined = pl.concat([s2, s3], how="vertical_relaxed")

    # Add a streaming-row-count so we can show progress; not free, but ~50ms.
    print(f"[sink]  writing to {OUT_PATH.name} (streaming, zstd)", flush=True)
    t1 = time.time()

    # sink_parquet writes in batches without materializing the full DataFrame.
    # Polars docs: "The advantage of sink_parquet over write_parquet is that
    # less RAM is required, as the data is processed in batches."
    combined.sink_parquet(
        OUT_PATH,
        compression="zstd",
        compression_level=3,
        # ~256k rows per batch keeps RAM low; polars default is ~512k.
        row_group_size=262144,
    )

    # Cheap post-hoc summary: file size + parquet metadata row count (no
    # full scan, polars reads the footer only).
    out_size_gb = OUT_PATH.stat().st_size / (1024 ** 3)
    n_rows = pl.scan_parquet(OUT_PATH).select(pl.len()).collect(engine="streaming").item()

    print(f"        sink: {time.time() - t1:.1f}s", flush=True)
    print(f"        total: {time.time() - t0:.1f}s", flush=True)
    print(f"        wrote {n_rows:,} pairs, {out_size_gb:.2f} GB on disk", flush=True)
    print(f"        path: {OUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
