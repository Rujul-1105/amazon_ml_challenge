"""RAM-safe block recall validator (lazy polars).

Computes per-S1 recall = |predicted ∩ true| / |true| for a holdout split
of S1 ids, then aggregates to mean/median/quartiles + per-country breakdown.

The original `validate_block_recall.py` materializes:
  - one Python `set` per predicted (S1, cand) pair (~100 B/row)  → ~20 GB at 190 M rows
  - one Python `set` per true (S1, cand) pair                     → ~1 GB
  - one Python float per held-out S1                            → ~1 MB

That OOMs on 8-16 GB boxes for the combined S2+S3 train output. This script
keeps everything inside polars' streaming engine:
  - ground truth: small (~7.6 M pairs, fully materializable)
  - block parquets: scanned lazily, only `(s1_id, cand_id)` columns selected
  - recall math: single lazy query that joins, groups, and aggregates
  - peak RAM: well under 2 GB

Usage:
    python scripts/validate_block_recall_lazy.py
    python scripts/validate_block_recall_lazy.py \
        --blocks artifacts/block_S2_features.parquet artifacts/block_S3_features.parquet
"""
from __future__ import annotations

import argparse
import io
import os
import random
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
DATASET_TRAIN = PROJECT_ROOT.parent.parent / "dataset" / "student_resource" / "dataset" / "train"

SEED = 42
HOLDOUT_FRAC = 0.10


def load_truth_pairs() -> pl.DataFrame:
    """Return a small frame of (source1_entity_id, matched_id) positive pairs."""
    print(f"[load] ground truth: {DATASET_TRAIN / 'train_ground_truth.tsv'}", flush=True)
    t0 = time.time()
    gt = pl.read_csv(
        DATASET_TRAIN / "train_ground_truth.tsv",
        separator="\t", encoding="utf8-lossy",
    ).rename({c: c.strip() for c in pl.read_csv(
        DATASET_TRAIN / "train_ground_truth.tsv",
        separator="\t", encoding="utf8-lossy", n_rows=0,
    ).columns})
    pairs = (
        gt
        .with_columns(pl.col("matched_entity_ids").str.split(",").alias("matched_list"))
        .explode("matched_list")
        .rename({"matched_list": "matched_id"})
        .with_columns(pl.col("matched_id").str.strip_chars())
        .filter(pl.col("matched_id").is_not_null() & (pl.col("matched_id") != ""))
        .select("source1_entity_id", "matched_id")
    )
    print(f"        {pairs.height:,} positive pairs in {time.time() - t0:.1f}s", flush=True)
    return pairs


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--blocks", nargs="+",
                   default=[str(ARTIFACTS / "block_S2_features.parquet"),
                            str(ARTIFACTS / "block_S3_features.parquet")])
    p.add_argument("--min-recall", type=float, default=0.85)
    args = p.parse_args()

    truth_pairs = load_truth_pairs()

    # ---- 1. Holdout split (deterministic) ----
    print("[holdout] sampling 10% S1 ids deterministically ...", flush=True)
    s1_ids = (truth_pairs.select("source1_entity_id").unique()["source1_entity_id"].to_list())
    rng = random.Random(SEED)
    rng.shuffle(s1_ids)
    holdout_size = int(HOLDOUT_FRAC * len(s1_ids))
    holdout = s1_ids[:holdout_size]
    holdout_set = set(holdout)
    print(f"        {holdout_size:,} held out of {len(s1_ids):,} S1 with >=1 true match", flush=True)

    # Also pull ALL S1 ids (incl. singletons) for the candidate-per-S1 denominator.
    print("[load] all S1 ids from s1_norm_train.parquet (incl. singletons) ...", flush=True)
    all_s1 = pl.read_parquet(ARTIFACTS / "s1_norm_train.parquet",
                             columns=["entity_id", "country"])
    print(f"        {all_s1.height:,} S1 ids", flush=True)

    # ---- 2. Lazy union of block predictions, dedup to (s1_id, cand_id) ----
    print(f"[scan] {len(args.blocks)} block parquet(s) ...", flush=True)
    block_lfs = [
        pl.scan_parquet(path)
          .select("source1_entity_id", "candidate_entity_id")
        for path in args.blocks
    ]
    pred_lf = pl.concat(block_lfs, how="vertical_relaxed").unique(
        subset=["source1_entity_id", "candidate_entity_id"]
    )

    # ---- 3. Lazy query: per-S1 candidate count + hit count ----
    holdout_frame = pl.DataFrame({"source1_entity_id": list(holdout_set)})

    print("[query] computing per-S1 candidate counts, hit counts, true counts (streaming) ...", flush=True)
    t0 = time.time()

    # All S1 ids in the holdout (incl. singletons) → join to pred to get n_cands per S1
    holdout_with_country = holdout_frame.join(
        all_s1.rename({"entity_id": "source1_entity_id"}),
        on="source1_entity_id", how="left",
    ).lazy()

    cand_per_s1 = (
        holdout_with_country
        .join(pred_lf, on="source1_entity_id", how="left")
        .group_by("source1_entity_id")
        .agg(pl.col("candidate_entity_id").n_unique().alias("n_cands"))
        .with_columns(pl.col("n_cands").fill_null(0))
    )

    # True matches per S1 (from ground truth)
    true_per_s1 = (
        truth_pairs.lazy()
        .filter(pl.col("source1_entity_id").is_in(holdout_set))
        .group_by("source1_entity_id")
        .agg(pl.col("matched_id").n_unique().alias("n_true"))
    )

    # Hits per S1: join truth on (s1, matched == candidate) inside the holdout
    truth_in_holdout = truth_pairs.lazy().filter(pl.col("source1_entity_id").is_in(holdout_set))
    hits_per_s1 = (
        truth_in_holdout
        .rename({"matched_id": "candidate_entity_id"})
        .join(pred_lf, on=["source1_entity_id", "candidate_entity_id"], how="inner")
        .group_by("source1_entity_id")
        .agg(pl.len().alias("n_hits"))
    )

    # ---- 4. Stitch together, compute recall ----
    recall_lf = (
        holdout_with_country.select("source1_entity_id", "country")
        .join(cand_per_s1, on="source1_entity_id", how="left")
        .join(true_per_s1, on="source1_entity_id", how="left")
        .join(hits_per_s1, on="source1_entity_id", how="left")
        .with_columns([
            pl.col("n_cands").fill_null(0),
            pl.col("n_true").fill_null(0),
            pl.col("n_hits").fill_null(0),
            pl.col("country").fill_null("Unknown"),
        ])
        .with_columns(
            pl.when(pl.col("n_true") > 0)
              .then(pl.col("n_hits") / pl.col("n_true"))
              .otherwise(None)  # singleton → recall undefined; skip in aggregation
              .alias("recall")
        )
    )

    # Force materialization in streaming mode.
    df = recall_lf.collect(engine="streaming")
    print(f"        query took {time.time() - t0:.1f}s, frame has {df.height:,} rows", flush=True)

    # ---- 5. Aggregate ----
    n_singletons = df.filter(pl.col("n_true") == 0).height
    n_eval = df.filter(pl.col("n_true") > 0).height

    if n_eval == 0:
        print("ERROR: no held-out S1 with non-empty true matches", flush=True)
        return 2

    # Polars quantile for percentiles (uses T-Digest; small memory).
    recalls = df.filter(pl.col("recall").is_not_null())["recall"]
    overall = float(recalls.mean())
    quantiles = recalls.quantile(quantile=[0.25, 0.50, 0.75])
    q25 = float(quantiles[0])
    median = float(quantiles[1])
    q75 = float(quantiles[2])
    p100 = float(recalls.max())
    p0 = float(recalls.min())
    perfect_frac = float((recalls >= 0.9999).mean())

    mean_cands = float(df["n_cands"].mean())

    print("", flush=True)
    print("=" * 60, flush=True)
    print("BLOCK RECALL REPORT (lazy / RAM-safe)", flush=True)
    print("=" * 60, flush=True)
    print(f"Holdout S1 ids (10%):              {holdout_size:,}", flush=True)
    print(f"  Singletons (skipped):             {n_singletons:,}", flush=True)
    print(f"  Non-singletons (evaluated):       {n_eval:,}", flush=True)
    print(f"Mean candidates per holdout S1:     {mean_cands:.1f}", flush=True)
    print(f"Mean recall:                        {overall:.4f}", flush=True)
    print(f"Median recall:                      {median:.4f}", flush=True)
    print(f"P25 recall:                         {q25:.4f}", flush=True)
    print(f"P75 recall:                         {q75:.4f}", flush=True)
    print(f"Min recall:                         {p0:.4f}", flush=True)
    print(f"Max recall:                         {p100:.4f}", flush=True)
    print(f"% S1 with recall == 1.0:            {perfect_frac:.4f}", flush=True)
    print("", flush=True)
    print("Per-country recall:", flush=True)
    per_country = (
        df
        .filter(pl.col("recall").is_not_null())
        .group_by("country")
        .agg([
            pl.col("recall").mean().alias("mean_recall"),
            pl.len().alias("n"),
        ])
        .sort("country")
    )
    for row in per_country.iter_rows(named=True):
        print(f"  {row['country']:>8s}: mean={row['mean_recall']:.4f}  n={row['n']:,}",
              flush=True)
    print("=" * 60, flush=True)

    if overall >= args.min_recall:
        print(f"PASS (recall {overall:.4f} >= {args.min_recall})", flush=True)
        return 0
    else:
        print(f"FAIL (recall {overall:.4f} < {args.min_recall})", flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
