"""Validate blocking recall vs train_ground_truth.tsv on a 10% cluster-aware holdout.

Usage:
    python scripts/validate_block_recall.py
    python scripts/validate_block_recall.py --blocks artifacts/block_S2_features.parquet artifacts/block_S3_features.parquet
"""
from __future__ import annotations

import argparse
import io
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "artifacts"
DATASET_TRAIN = PROJECT_ROOT / "dataset" / "train"

SEED = 42
HOLDOUT_FRAC = 0.10


def load_truth() -> dict[str, set[str]]:
    """dict[s1_id -> set[matched_id]]"""
    print(f"[load] ground truth: {DATASET_TRAIN / 'train_ground_truth.tsv'}", flush=True)
    t0 = time.time()
    gt = pl.read_csv(
        DATASET_TRAIN / "train_ground_truth.tsv",
        separator="\t", encoding="utf8-lossy",
        schema_overrides={"source1_entity_id": pl.Utf8, "matched_entity_ids": pl.Utf8},
    )
    print(f"        {gt.height:,} rows in {time.time() - t0:.1f}s", flush=True)

    print("[explode] ground truth pairs ...", flush=True)
    t0 = time.time()
    pairs = (gt.with_columns(pl.col("matched_entity_ids").str.split(",").alias("matched_list"))
               .explode("matched_list")
               .rename({"matched_list": "matched_id"})
               .with_columns(pl.col("matched_id").str.strip_chars())
               .filter(pl.col("matched_id").is_not_null() & (pl.col("matched_id") != ""))
               .select("source1_entity_id", "matched_id"))
    print(f"        {pairs.height:,} positive pairs in {time.time() - t0:.1f}s", flush=True)

    truth: dict[str, set[str]] = defaultdict(set)
    for r in pairs.iter_rows(named=True):
        truth[r["source1_entity_id"]].add(r["matched_id"])
    return truth


def load_blocks(paths: list[Path]) -> dict[str, set[str]]:
    """dict[s1_id -> set[candidate_id]] from one or more block parquets."""
    pred: dict[str, set[str]] = defaultdict(set)
    for path in paths:
        if not path.exists():
            print(f"  WARN: {path} not found, skipping", flush=True)
            continue
        print(f"[load] blocks: {path.name}", flush=True)
        t0 = time.time()
        df = pl.read_parquet(path, columns=["source1_entity_id", "candidate_entity_id"])
        for r in df.iter_rows(named=True):
            pred[r["source1_entity_id"]].add(r["candidate_entity_id"])
        print(f"        {df.height:,} rows in {time.time() - t0:.1f}s", flush=True)
    return pred


def all_s1_ids() -> list[str]:
    """Enumerate all S1 ids in train (including singletons)."""
    print(f"[load] all S1 ids from s1_norm_train.parquet", flush=True)
    t0 = time.time()
    df = pl.read_parquet(ARTIFACTS / "s1_norm_train.parquet", columns=["entity_id"])
    print(f"        {df.height:,} ids in {time.time() - t0:.1f}s", flush=True)
    return df["entity_id"].to_list()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--blocks", nargs="+",
                   default=[str(ARTIFACTS / "block_S2_features.parquet"),
                            str(ARTIFACTS / "block_S3_features.parquet")])
    p.add_argument("--min-recall", type=float, default=0.85)
    args = p.parse_args()

    truth = load_truth()
    pred = load_blocks([Path(x) for x in args.blocks])
    all_ids = all_s1_ids()

    # Holdout split (deterministic)
    rng = random.Random(SEED)
    rng.shuffle(all_ids)
    holdout_size = int(HOLDOUT_FRAC * len(all_ids))
    holdout = set(all_ids[:holdout_size])
    print(f"[holdout] {holdout_size:,} S1 ids ({HOLDOUT_FRAC * 100:.0f}%)", flush=True)

    # Recall per held-out S1 with non-empty true matches
    per_s1_recall: list[float] = []
    per_country: dict[str, list[float]] = defaultdict(list)
    n_singletons_in_holdout = 0
    n_non_singleton_holdout = 0

    s1_country_lookup: dict[str, str] = {}
    print("[load] s1 country lookup ...", flush=True)
    df_s1 = pl.read_parquet(ARTIFACTS / "s1_norm_train.parquet", columns=["entity_id", "country"])
    for r in df_s1.iter_rows(named=True):
        s1_country_lookup[r["entity_id"]] = r["country"]

    for s1_id in holdout:
        true = truth.get(s1_id, set())
        if not true:
            n_singletons_in_holdout += 1
            continue
        n_non_singleton_holdout += 1
        p_set = pred.get(s1_id, set())
        hit = len(true & p_set)
        rec = hit / len(true)
        per_s1_recall.append(rec)
        c = s1_country_lookup.get(s1_id, "Unknown")
        per_country[c].append(rec)

    if not per_s1_recall:
        print("ERROR: no held-out S1 with non-empty true matches", flush=True)
        return 2

    overall = sum(per_s1_recall) / len(per_s1_recall)
    per_s1_recall.sort()
    median = per_s1_recall[len(per_s1_recall) // 2]
    p25 = per_s1_recall[len(per_s1_recall) // 4]
    p75 = per_s1_recall[3 * len(per_s1_recall) // 4]

    n_cands_total = sum(len(pred.get(s1, set())) for s1 in holdout)
    mean_cands = n_cands_total / max(holdout_size, 1)

    print("", flush=True)
    print("=" * 60, flush=True)
    print("BLOCK RECALL REPORT", flush=True)
    print("=" * 60, flush=True)
    print(f"Holdout S1 ids (10%):              {holdout_size:,}", flush=True)
    print(f"  Singletons:                       {n_singletons_in_holdout:,}", flush=True)
    print(f"  Non-singletons (evaluated):       {n_non_singleton_holdout:,}", flush=True)
    print(f"Mean candidates per holdout S1:     {mean_cands:.1f}", flush=True)
    print(f"Mean recall:                        {overall:.4f}", flush=True)
    print(f"Median recall:                      {median:.4f}", flush=True)
    print(f"P25 recall:                         {p25:.4f}", flush=True)
    print(f"P75 recall:                         {p75:.4f}", flush=True)
    print(f"Min recall:                         {per_s1_recall[0]:.4f}", flush=True)
    print(f"Max recall:                         {per_s1_recall[-1]:.4f}", flush=True)
    print(f"% S1 with recall == 1.0:            "
          f"{sum(1 for r in per_s1_recall if r >= 0.9999) / len(per_s1_recall):.4f}", flush=True)
    print("", flush=True)
    print("Per-country recall:", flush=True)
    for c in sorted(per_country.keys()):
        rs = per_country[c]
        if rs:
            print(f"  {c:>8s}: mean={sum(rs)/len(rs):.4f}  n={len(rs):,}", flush=True)
    print("=" * 60, flush=True)

    if overall >= args.min_recall:
        print(f"PASS (recall {overall:.4f} >= {args.min_recall})", flush=True)
        return 0
    else:
        print(f"FAIL (recall {overall:.4f} < {args.min_recall})", flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())