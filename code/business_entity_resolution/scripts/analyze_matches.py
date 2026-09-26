"""Side-by-side analysis of ground-truth matches.

For every (source1_entity_id, matched_entity_id) pair in `train_ground_truth.tsv`,
fetches the `business_name` and `business_address` from the corresponding
source file and writes a flat TSV that can be eyeballed or fed into any further
similarity analysis.

Also emits a small `match_insights.json` summarising:
  * match counts per country / per source
  * name- and address-similarity distributions (rapidfuzz) on a 5k-pair sample
  * same-country rate
  * exact-name / exact-address rates

Run from anywhere:

    python scripts/analyze_matches.py

Outputs land in ``amazon_ml_challenge/artifacts/``:
  * matched_pairs.tsv        (one row per (S1, matched) pair, ~rows × 8 cols)
  * match_insights.json      (summary numbers + similarity histograms)
  * matched_pairs_sample.tsv (5,000-row sample for quick inspection)
"""
from __future__ import annotations

import io
import json
import os
import random
import sys
import time
from pathlib import Path

# Force UTF-8 stdout/stderr on Windows (cp1252 default breaks on unicode arrows)
if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import polars as pl
import polars.selectors as cs
from rapidfuzz import fuzz

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = PROJECT_ROOT / "dataset" / "train"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

SOURCE1_TSV = TRAIN_DIR / "train_source1.tsv"
SOURCE2_TSV = TRAIN_DIR / "train_source2.tsv"
SOURCE3_TSV = TRAIN_DIR / "train_source3.tsv"
GT_TSV = TRAIN_DIR / "train_ground_truth.tsv"

OUT_PAIRS = ARTIFACTS_DIR / "matched_pairs.tsv"
OUT_SAMPLE = ARTIFACTS_DIR / "matched_pairs_sample.tsv"
OUT_INSIGHTS = ARTIFACTS_DIR / "match_insights.json"

# ---------------------------------------------------------------------------
# Loaders (project standard: utf8-lossy, tab-separated, schema-enforced)
# ---------------------------------------------------------------------------

SOURCE_SCHEMA = {
    "entity_id": pl.Utf8,
    "business_name": pl.Utf8,
    "business_address": pl.Utf8,
    "country": pl.Utf8,
}
GT_SCHEMA = {
    "source1_entity_id": pl.Utf8,
    "matched_entity_ids": pl.Utf8,
}


def _read_source(path: Path, label: str) -> pl.DataFrame:
    print(f"[load] {label}: {path.name} ({path.stat().st_size / 1e9:.2f} GB)")
    t0 = time.time()
    df = pl.read_csv(
        path,
        separator="\t",
        encoding="utf8-lossy",
        schema_overrides=SOURCE_SCHEMA,
        null_values=["", "NA", "null", "NaN"],
        ignore_errors=False,
        low_memory=False,
    )
    print(f"        → {len(df):,} rows in {time.time() - t0:.1f}s")
    return df


def _read_ground_truth(path: Path) -> pl.DataFrame:
    print(f"[load] ground truth: {path.name}")
    t0 = time.time()
    df = pl.read_csv(
        path,
        separator="\t",
        encoding="utf8-lossy",
        schema_overrides=GT_SCHEMA,
        null_values=["", "NA", "null"],
        ignore_errors=False,
        low_memory=False,
    )
    print(f"        → {len(df):,} rows in {time.time() - t0:.1f}s")
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_matched_pairs() -> pl.DataFrame:
    """Return one row per (S1 entity, matched entity) pair with both sides' names/addresses."""
    s1 = _read_source(SOURCE1_TSV, "source1")
    s2 = _read_source(SOURCE2_TSV, "source2").with_columns(pl.lit("S2").alias("matched_source"))
    s3 = _read_source(SOURCE3_TSV, "source3").with_columns(pl.lit("S3").alias("matched_source"))
    gt = _read_ground_truth(GT_TSV)

    # S2 + S3 joined into one lookup frame; entity_ids are globally unique across S2/S3
    lookup = pl.concat([s2, s3], how="vertical_relaxed").rename({
        "entity_id": "matched_entity_id",
        "business_name": "matched_business_name",
        "business_address": "matched_business_address",
        "country": "matched_country",
    })

    # Explode the comma-separated matched list → one row per pair
    print("[join] exploding ground truth ...")
    pairs = (
        gt
        .with_columns(pl.col("matched_entity_ids").str.split(",").alias("matched_list"))
        .explode("matched_list")
        .rename({"matched_list": "matched_entity_id"})
        .with_columns(pl.col("matched_entity_id").str.strip_chars())
        .filter(pl.col("matched_entity_id").is_not_null() & (pl.col("matched_entity_id") != ""))
        .select("source1_entity_id", "matched_entity_id")
    )
    print(f"        → {len(pairs):,} positive (S1, matched) pairs")

    # Join with S1 to attach name/address/country of the reference entity
    print("[join] attaching S1 metadata ...")
    pairs = pairs.join(
        s1.select(
            "entity_id",
            pl.col("business_name").alias("s1_business_name"),
            pl.col("business_address").alias("s1_business_address"),
            pl.col("country").alias("s1_country"),
        ),
        left_on="source1_entity_id",
        right_on="entity_id",
        how="left",
    )
    print(f"        → {len(pairs):,} rows after S1 join")

    # Join with S2+S3 lookup to attach matched-side name/address/country
    print("[join] attaching matched-side metadata (S2 + S3) ...")
    pairs = pairs.join(lookup, on="matched_entity_id", how="left")
    print(f"        → {len(pairs):,} rows after matched join")

    # Sanity check: how many joined successfully?
    n_total = len(pairs)
    n_missing_matched = pairs.filter(pl.col("matched_business_name").is_null()).height
    n_missing_s1 = pairs.filter(pl.col("s1_business_name").is_null()).height
    print(f"[stats] rows: {n_total:,} | missing matched meta: {n_missing_matched:,} | missing S1 meta: {n_missing_s1:,}")

    return pairs.select([
        "source1_entity_id",
        "matched_entity_id",
        "matched_source",
        "s1_country",
        "matched_country",
        "s1_business_name",
        "matched_business_name",
        "s1_business_address",
        "matched_business_address",
    ])


# ---------------------------------------------------------------------------
# Insights
# ---------------------------------------------------------------------------

def compute_insights(pairs: pl.DataFrame) -> dict:
    out: dict = {"summary": {}, "similarity": {}}

    # 1) Summary counts
    s = {}
    s["total_pairs"] = pairs.height
    s["unique_s1"] = pairs["source1_entity_id"].n_unique()
    s["unique_matched"] = pairs["matched_entity_id"].n_unique()
    s["matched_source_counts"] = (
        pairs.group_by("matched_source").len().sort("matched_source").to_dicts()
    )
    s["s1_country_counts"] = (
        pairs.group_by("s1_country").len().sort("s1_country").to_dicts()
    )

    # Same country?
    same_country = (pairs["s1_country"] == pairs["matched_country"]).sum()
    cross_country = (pairs["s1_country"] != pairs["matched_country"]).sum()
    s["same_country_pairs"] = same_country
    s["cross_country_pairs"] = cross_country
    s["same_country_rate"] = round(same_country / max(s["total_pairs"], 1), 4)

    out["summary"] = s

    # 2) Exact-match rates (cheap, no fuzzy)
    name_eq = pairs.filter(pl.col("s1_business_name") == pl.col("matched_business_name")).height
    addr_eq = pairs.filter(pl.col("s1_business_address") == pl.col("matched_business_address")).height
    out["summary"]["name_exact_match_rate"] = round(name_eq / max(s["total_pairs"], 1), 4)
    out["summary"]["address_exact_match_rate"] = round(addr_eq / max(s["total_pairs"], 1), 4)

    # 3) Similarity distribution on a 5,000-pair sample
    print("[sims] sampling 5,000 pairs for fuzzy similarity ...")
    pdf = pairs.sample(n=min(5000, pairs.height), seed=42, with_replacement=False).to_pandas()

    name_ratios = []
    addr_ratios = []
    name_token_set = []
    for _, row in pdf.iterrows():
        s1n = "" if row["s1_business_name"] is None or (isinstance(row["s1_business_name"], float) and row["s1_business_name"] != row["s1_business_name"]) else str(row["s1_business_name"])
        smn = "" if row["matched_business_name"] is None or (isinstance(row["matched_business_name"], float) and row["matched_business_name"] != row["matched_business_name"]) else str(row["matched_business_name"])
        s1a = "" if row["s1_business_address"] is None or (isinstance(row["s1_business_address"], float) and row["s1_business_address"] != row["s1_business_address"]) else str(row["s1_business_address"])
        sma = "" if row["matched_business_address"] is None or (isinstance(row["matched_business_address"], float) and row["matched_business_address"] != row["matched_business_address"]) else str(row["matched_business_address"])
        if s1n and smn:
            name_ratios.append(fuzz.ratio(s1n, smn))
            name_token_set.append(fuzz.token_set_ratio(s1n, smn))
        if s1a and sma:
            addr_ratios.append(fuzz.ratio(s1a, sma))

    def _hist(values, bins=(0, 25, 50, 70, 85, 95, 101)):
        if not values:
            return {}
        out_hist = {}
        for lo, hi in zip(bins[:-1], bins[1:]):
            key = f"{lo}-{hi - 1}"
            out_hist[key] = sum(1 for v in values if lo <= v < hi)
        return out_hist

    out["similarity"]["name_ratio"] = {
        "n": len(name_ratios),
        "mean": round(sum(name_ratios) / max(len(name_ratios), 1), 2),
        "median": sorted(name_ratios)[len(name_ratios) // 2] if name_ratios else None,
        "hist": _hist(name_ratios),
    }
    out["similarity"]["name_token_set_ratio"] = {
        "n": len(name_token_set),
        "mean": round(sum(name_token_set) / max(len(name_token_set), 1), 2),
        "median": sorted(name_token_set)[len(name_token_set) // 2] if name_token_set else None,
        "hist": _hist(name_token_set),
    }
    out["similarity"]["address_ratio"] = {
        "n": len(addr_ratios),
        "mean": round(sum(addr_ratios) / max(len(addr_ratios), 1), 2),
        "median": sorted(addr_ratios)[len(addr_ratios) // 2] if addr_ratios else None,
        "hist": _hist(addr_ratios),
    }

    # 4) Token-level first-word / city-prefix matches (strong structural signal)
    def first_token(s):
        if s is None or (isinstance(s, float) and s != s):  # NaN check
            return ""
        s = (str(s) or "").strip().lower()
        if not s:
            return ""
        # take the first run of letters/digits
        i = 0
        while i < len(s) and s[i].isalnum():
            i += 1
        return s[:i]

    pdf["s1_first_tok"] = pdf["s1_business_name"].map(first_token)
    pdf["m_first_tok"] = pdf["matched_business_name"].map(first_token)
    pdf["s1_addr_first_tok"] = pdf["s1_business_address"].map(first_token)
    pdf["m_addr_first_tok"] = pdf["matched_business_address"].map(first_token)
    name_first_eq = (pdf["s1_first_tok"] == pdf["m_first_tok"]).sum()
    addr_first_eq = (pdf["s1_addr_first_tok"] == pdf["m_addr_first_tok"]).sum()
    out["summary"]["name_first_token_eq_rate"] = round(float(name_first_eq) / max(len(pdf), 1), 4)
    out["summary"]["address_first_token_eq_rate"] = round(float(addr_first_eq) / max(len(pdf), 1), 4)

    # 5) How many pairs share at least one name token (cheap proxy for token overlap)
    def toks(s):
        if s is None or (isinstance(s, float) and s != s):
            return set()
        s = str(s).lower()
        return set(t for t in s.split() if len(t) >= 3)

    pdf["s1_name_toks"] = pdf["s1_business_name"].map(toks)
    pdf["m_name_toks"] = pdf["matched_business_name"].map(toks)
    pdf["name_tok_overlap"] = pdf.apply(lambda r: len(r["s1_name_toks"] & r["m_name_toks"]), axis=1)
    out["summary"]["name_token_overlap_distribution"] = {
        "0": int((pdf["name_tok_overlap"] == 0).sum()),
        "1": int((pdf["name_tok_overlap"] == 1).sum()),
        "2": int((pdf["name_tok_overlap"] == 2).sum()),
        "3+": int((pdf["name_tok_overlap"] >= 3).sum()),
    }

    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    pairs = build_matched_pairs()

    # Write full pairs (UTF-8, tab-separated, no quoting to match project convention)
    print(f"[write] {OUT_PAIRS} ({len(pairs):,} rows)")
    pairs.write_csv(OUT_PAIRS, separator="\t", quote_style="never")

    # 5,000-row sample for quick inspection
    sample = pairs.sample(n=min(5000, len(pairs)), seed=42, with_replacement=False)
    print(f"[write] {OUT_SAMPLE} ({len(sample):,} rows)")
    sample.write_csv(OUT_SAMPLE, separator="\t", quote_style="never")

    # Insights
    print("[insights] computing ...")
    insights = compute_insights(pairs)
    with OUT_INSIGHTS.open("w", encoding="utf-8") as f:
        json.dump(insights, f, indent=2, ensure_ascii=False)
    print(f"[write] {OUT_INSIGHTS}")

    print(f"[done] total wall-clock: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()