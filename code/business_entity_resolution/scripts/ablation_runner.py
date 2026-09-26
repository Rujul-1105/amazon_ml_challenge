"""Ablation harness: test 16 candidate-generation methods on a 10K S1 sample.

Apples-to-apples comparison: every method scores on the same S1 sample
(stratified by country × match-count). For each method we measure:
  - mean / median / P25 / P75 / max recall
  - % S1 with recall = 1.0
  - mean candidates per S1
  - per-country mean recall (US / India)
  - per-failure-category miss rate (where we can detect the category)

Methods tested (16, organized in 8 groups):
  Group A (single index baselines): M01-M04
  Group B (v3 structural combos):   M05-M07
  Group C (ngram keys added):       M08
  Group D (Latin tokens):           M09-M10
  Group E (liberal floors):         M11-M12
  Group F (cap bumps):              M13-M14
  Group G (Floor B relaxed):        M15
  Group H (proposed v4 setting):    M16

Output: artifacts/ablation_full.csv (one row per method)

The 10K sample is stratified by (country, match-count bucket) to mirror the
full-data distribution. Seed=42 for reproducibility.

CLI:
  python scripts/ablation_runner.py --sample-size 10000 --output artifacts/ablation_full.csv
  python scripts/ablation_runner.py --sample-size 1000 --output artifacts/ablation_smoke.csv
"""
from __future__ import annotations
import argparse
import gc
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from _resources import detect_resources, configure_polars_threads, log_resources

_RES = detect_resources()
configure_polars_threads(_RES["polars_threads"])

import numpy as np
import polars as pl
from rapidfuzz import fuzz

ARTIFACTS = PROJECT_ROOT / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

S1_NORM = ARTIFACTS / "s1_norm_train.parquet"
S2_NORM = ARTIFACTS / "s2_norm_train.parquet"
S3_NORM = ARTIFACTS / "s3_norm_train.parquet"
TRAIN_GT = PROJECT_ROOT.parent.parent / "dataset" / "student_resource" / "dataset" / "train" / "train_ground_truth.tsv"

# Stop tokens (literal mirror of block_features.py)
STOP_TOKENS = {
    "the", "a", "an", "and", "or", "of", "in", "at", "on", "to", "for", "with", "by", "from",
    "as", "is", "are", "was", "were", "be", "been", "it", "its", "this", "that", "these", "those",
    "inc", "incorporated", "ltd", "limited", "llc", "llp", "corp", "corporation",
    "company", "co", "companies", "pvt", "private", "plc", "gmbh", "sa", "srl",
    "group", "holdings", "partners", "associates", "enterprises",
    "international", "global", "world",
    "no", "number", "de", "la", "el", "los", "las", "san", "santa",
    "new", "old", "north", "south", "east", "west", "central",
    "city", "state", "india", "usa", "us", "uk",
}


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in toks if len(t) >= 3 and t not in STOP_TOKENS and not t.isdigit()]


def _s(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and x != x:
        return ""
    return str(x)


# ---------------------------------------------------------------------------
# Data loading (cached once)
# ---------------------------------------------------------------------------
def load_sources() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    cols = [
        "entity_id", "country",
        "name_clean", "name_latin", "name_tokens", "name_dev_ratio", "name_missing",
        "addr_clean", "addr_latin",
        "addr_first_word", "addr_last_word",
        "addr_city", "addr_state",
        "addr_house_number", "addr_road", "addr_unit", "addr_suburb",
        "name_ngram_key", "addr_ngram_key",
        "addr_missing",
    ]
    log_resources("load")
    print(f"[load] s1 ...", flush=True)
    t0 = time.time()
    s1 = pl.read_parquet(S1_NORM, columns=cols)
    print(f"        {s1.height:,} rows, {time.time()-t0:.1f}s", flush=True)
    print(f"[load] s2 ...", flush=True)
    t0 = time.time()
    s2 = pl.read_parquet(S2_NORM, columns=cols)
    print(f"        {s2.height:,} rows, {time.time()-t0:.1f}s", flush=True)
    print(f"[load] s3 ...", flush=True)
    t0 = time.time()
    s3 = pl.read_parquet(S3_NORM, columns=cols)
    print(f"        {s3.height:,} rows, {time.time()-t0:.1f}s", flush=True)
    return s1, s2, s3


def load_ground_truth() -> dict[str, set[str]]:
    """Return {s1_id: {m_id, m_id, ...}} mapping from train ground truth."""
    print(f"[load] ground truth ...", flush=True)
    t0 = time.time()
    gt = (
        pl.read_csv(TRAIN_GT, separator="\t", encoding="utf8-lossy")
          .rename({c: c.strip() for c in
                   pl.read_csv(TRAIN_GT, separator="\t", encoding="utf8-lossy").columns})
          .with_columns(pl.col("matched_entity_ids").str.split(",").alias("matched_list"))
          .explode("matched_list")
          .with_columns(pl.col("matched_list").str.strip_chars().alias("m"))
          .filter(pl.col("m") != "")
          .select("source1_entity_id", "m")
    )
    gt_dict: dict[str, set[str]] = {}
    for row in gt.iter_rows(named=True):
        gt_dict.setdefault(row["source1_entity_id"], set()).add(row["m"])
    print(f"        {len(gt_dict):,} S1 entities, {gt.height:,} pairs in {time.time()-t0:.1f}s", flush=True)
    return gt_dict


def sample_s1_ids(s1: pl.DataFrame, gt: dict[str, set[str]], n: int, seed: int) -> pl.DataFrame:
    """Sample n S1 IDs stratified by (country, match-count bucket).

    Buckets:
      - singleton (n_matches == 0)
      - 1-2 matches
      - 3-5 matches
      - 6+ matches
    """
    print(f"[sample] drawing {n:,} S1 ids (seed={seed}) ...", flush=True)
    rng = np.random.default_rng(seed)
    s1_meta = s1.select(["entity_id", "country"]).with_columns(
        pl.col("entity_id").alias("source1_entity_id"),
    ).with_columns(
        pl.col("source1_entity_id").map_elements(
            lambda sid: len(gt.get(sid, set())),
            return_dtype=pl.Int32,
        ).alias("n_true"),
    ).with_columns(
        pl.when(pl.col("n_true") == 0).then(pl.lit("singleton"))
         .when(pl.col("n_true") <= 2).then(pl.lit("1-2"))
         .when(pl.col("n_true") <= 5).then(pl.lit("3-5"))
         .otherwise(pl.lit("6+"))
         .alias("bucket"),
    )
    s1_meta = s1_meta.select(["source1_entity_id", "country", "bucket"])

    # Stratified sampling: proportional allocation by (country, bucket) frequency.
    counts = s1_meta.group_by(["country", "bucket"]).agg(pl.len().alias("pop")).with_columns(
        (pl.col("pop") * n / s1_meta.height).cast(pl.Int32).alias("take")
    )
    # Build the sampled set
    sampled_ids: list[str] = []
    s1_by_cb = {row["country"]+"_"+row["bucket"]: row for row in counts.iter_rows(named=True)}
    for row in counts.iter_rows(named=True):
        cb = row["country"] + "_" + row["bucket"]
        take = row["take"]
        pool = s1_meta.filter((pl.col("country") == row["country"]) & (pl.col("bucket") == row["bucket"]))
        if take >= pool.height:
            sampled_ids.extend(pool["source1_entity_id"].to_list())
        else:
            chosen = rng.choice(pool["source1_entity_id"].to_list(), size=take, replace=False)
            sampled_ids.extend(chosen.tolist())
    # Add s1_country for each
    s1_country_map = dict(zip(s1["entity_id"].to_list(), s1["country"].to_list()))
    out = pl.DataFrame({
        "source1_entity_id": sampled_ids,
        "s1_country": [s1_country_map.get(s, "US") for s in sampled_ids],
    })
    print(f"        {out.height:,} S1 sampled; country/bucket distribution:", flush=True)
    print(s1_meta.filter(pl.col("source1_entity_id").is_in(sampled_ids))
                   .group_by(["country", "bucket"]).agg(pl.len().alias("n")).sort(["country", "bucket"]))
    return out


# ---------------------------------------------------------------------------
# Index builders (per-method config)
# ---------------------------------------------------------------------------
def make_key_extractors(struct_keys: list[str]) -> dict[str, pl.Expr]:
    """Build a KEY_EXTRACTORS subset for the active struct keys.

    Mirrors block_features.py KEY_EXTRACTORS (block_features.py:129-149) with
    the optional name_ngram_key / addr_ngram_key additions.
    """
    all_keys = {
        "city":       pl.col("addr_city").fill_null("").str.strip_chars().str.to_lowercase(),
        "state":      pl.col("addr_state").fill_null("").str.strip_chars().str.to_lowercase(),
        "road":       pl.col("addr_road").fill_null("").str.strip_chars().str.to_lowercase(),
        "house":      pl.col("addr_house_number").fill_null("").str.strip_chars().str.to_lowercase(),
        "name_fw":    pl.col("name_clean").fill_null("").str.split(" ").list.first()
                         .fill_null("").str.strip_chars().str.to_lowercase(),
        "addr_fw":    pl.col("addr_first_word").fill_null("").str.strip_chars().str.to_lowercase(),
        "city_state": pl.concat_str(
                           [pl.col("addr_city").fill_null(""), pl.lit("|"),
                            pl.col("addr_state").fill_null("")],
                           separator="", ignore_nulls=True)
                       .str.strip_chars().str.to_lowercase(),
        "house_road": pl.concat_str(
                           [pl.col("addr_house_number").fill_null(""), pl.lit("|"),
                            pl.col("addr_road").fill_null("")],
                           separator="", ignore_nulls=True)
                       .str.strip_chars().str.to_lowercase(),
        "name_ngram_key": pl.col("name_ngram_key").fill_null("").str.strip_chars().str.to_lowercase(),
        "addr_ngram_key": pl.col("addr_ngram_key").fill_null("").str.strip_chars().str.to_lowercase(),
    }
    return {k: all_keys[k] for k in struct_keys}


def build_structural_indexes_subset(cand: pl.DataFrame,
                                    key_extractors: dict[str, pl.Expr],
                                    cap: int) -> dict[str, pl.DataFrame]:
    inv: dict[str, pl.DataFrame] = {}
    for name, expr in key_extractors.items():
        df = (cand.with_columns(expr.alias("_k"))
                  .filter(pl.col("_k").is_not_null() & (pl.col("_k") != ""))
                  .select(["_k", pl.col("entity_id").alias("_id")])
                  .unique(subset=["_k", "_id"])
                  .group_by("_k")
                  .agg(pl.col("_id"))
                  .with_columns(pl.col("_id").list.slice(0, cap))
                  .explode("_id"))
        inv[name] = df
    return inv


def build_token_index_subset(cand: pl.DataFrame, cap: int,
                             include_latin: bool) -> pl.DataFrame:
    if include_latin:
        text_expr = (pl.col("name_clean").fill_null("") + pl.lit(" ") +
                     pl.col("addr_clean").fill_null("") + pl.lit(" ") +
                     pl.col("name_latin").fill_null("") + pl.lit(" ") +
                     pl.col("addr_latin").fill_null(""))
    else:
        text_expr = (pl.col("name_clean").fill_null("") + pl.lit(" ") +
                     pl.col("addr_clean").fill_null(""))

    cand_tok = (cand.with_columns(text_expr.alias("_text"))
                   .with_columns(
                       pl.col("_text").map_batches(
                           lambda s: pl.Series([list(set(_tokenize(t) if t else [])) for t in s]),
                           return_dtype=pl.List(pl.Utf8),
                       ).alias("_toks"))
                   .select(["entity_id", "_toks"]).explode("_toks").rename({"_toks": "_tok"})
                   .filter(pl.col("_tok").is_not_null()))
    return (cand_tok
            .unique(subset=["_tok", "entity_id"])
            .group_by("_tok").agg(pl.col("entity_id"))
            .with_columns(pl.col("entity_id").list.slice(0, cap))
            .explode("entity_id")
            .rename({"entity_id": "_id"}))


def build_sorted_neighborhood_subset(cand: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    canon_to_id: dict[str, str] = {}
    for r in cand.iter_rows(named=True):
        text = (r.get("name_clean") or "") + " " + (r.get("addr_clean") or "")
        canon = _tokenize(text)
        canon = " ".join(sorted(set(canon)))
        canon_to_id.setdefault(canon or " ", r["entity_id"])
    sorted_canon = np.array(sorted(canon_to_id.keys()), dtype=object)
    n = len(sorted_canon)
    ids_aligned = np.empty(n, dtype=object)
    for i, c in enumerate(sorted_canon):
        ids_aligned[i] = canon_to_id[c]
    return sorted_canon, ids_aligned


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------
def probe_structural_subset(s1_chunk: pl.DataFrame,
                            inv: dict[str, pl.DataFrame],
                            top_k: int) -> pl.DataFrame:
    parts: list[pl.DataFrame] = []
    for name, idx_df in inv.items():
        s1k = s1_chunk.with_columns(
            # Use the corresponding key extractor logic; cheap path: re-derive
            # the same key on the S1 chunk. For ablation, we just call
            # make_key_extractors(name) again.
            make_key_extractors([name])[name].alias("_k")
        ).filter(pl.col("_k").is_not_null() & (pl.col("_k") != "")
        ).select(["source1_entity_id", "_k"])
        if s1k.is_empty():
            continue
        joined = s1k.join(idx_df, on="_k", how="inner")
        parts.append(joined.with_columns(pl.lit(name).alias("_kt")))
    if not parts:
        return pl.DataFrame(schema={
            "source1_entity_id": pl.Utf8, "candidate_entity_id": pl.Utf8,
            "n_struct_keys": pl.Int8, "_key_types": pl.List(pl.Utf8),
        })
    all_pairs = pl.concat(parts).select(["source1_entity_id", "_id", "_kt"])
    n_struct = (all_pairs.group_by(["source1_entity_id", "_id"])
                .agg([
                    pl.col("_kt").n_unique().cast(pl.Int8).alias("n_struct_keys"),
                    pl.col("_kt").alias("_key_types"),
                ]))
    return n_struct.rename({"_id": "candidate_entity_id"})


def probe_tokens_subset(s1_chunk: pl.DataFrame, token_idx: pl.DataFrame,
                        top_k: int, include_latin: bool) -> pl.DataFrame:
    if include_latin:
        text_expr = (pl.col("name_clean").fill_null("") + pl.lit(" ") +
                     pl.col("addr_clean").fill_null("") + pl.lit(" ") +
                     pl.col("name_latin").fill_null("") + pl.lit(" ") +
                     pl.col("addr_latin").fill_null(""))
    else:
        text_expr = (pl.col("name_clean").fill_null("") + pl.lit(" ") +
                     pl.col("addr_clean").fill_null(""))
    s1_tok = (s1_chunk.with_columns(text_expr.alias("_text"))
                       .select(["source1_entity_id", "_text"])
                       .with_columns(
                           pl.col("_text").map_batches(
                               lambda s: pl.Series([list(set(_tokenize(t) if t else [])) for t in s]),
                               return_dtype=pl.List(pl.Utf8),
                           ).alias("_toks"))
                       .select(["source1_entity_id", "_toks"]).explode("_toks")
                       .filter(pl.col("_toks").is_not_null())
                       .rename({"_toks": "_tok"}))
    if s1_tok.is_empty():
        return pl.DataFrame(schema={
            "source1_entity_id": pl.Utf8, "candidate_entity_id": pl.Utf8,
            "n_tokens_shared": pl.Int8,
        })
    joined = s1_tok.join(token_idx, on="_tok", how="inner")
    if joined.is_empty():
        return pl.DataFrame(schema={
            "source1_entity_id": pl.Utf8, "candidate_entity_id": pl.Utf8,
            "n_tokens_shared": pl.Int8,
        })
    return (joined.group_by(["source1_entity_id", "_id"])
                 .agg(pl.col("_tok").n_unique().cast(pl.Int8).alias("n_tokens_shared"))
                 .rename({"_id": "candidate_entity_id"}))


def probe_sorted_neighborhood_subset(s1_chunk: pl.DataFrame,
                                     sorted_canon: np.ndarray,
                                     ids_aligned: np.ndarray,
                                     window: int) -> pl.DataFrame:
    s1_rows = s1_chunk.select(["source1_entity_id", "name_clean", "addr_clean"]).to_dicts()
    pairs: list[tuple[str, str, int]] = []
    for row in s1_rows:
        s1_id = row["source1_entity_id"]
        text = (row.get("name_clean") or "") + " " + (row.get("addr_clean") or "")
        canon = _tokenize(text)
        canon = " ".join(sorted(set(canon)))
        if not canon or canon == " ":
            continue
        idx = int(np.searchsorted(sorted_canon, canon))
        lo = max(0, idx - window)
        hi = min(len(sorted_canon), idx + window + 1)
        for k in range(lo, hi):
            if k == idx:
                continue
            pairs.append((s1_id, ids_aligned[k], abs(k - idx)))
    if not pairs:
        return pl.DataFrame(schema={
            "source1_entity_id": pl.Utf8, "candidate_entity_id": pl.Utf8,
            "sortedn_rank": pl.Int16,
        })
    return pl.DataFrame(pairs, schema=["source1_entity_id", "candidate_entity_id", "sortedn_rank"], orient="row")


# ---------------------------------------------------------------------------
# Field attachment (vectorized polars joins; subset of block_features.attach_fields)
# ---------------------------------------------------------------------------
def attach_fields(pairs: pl.DataFrame, s1_meta: pl.DataFrame,
                  cand_df: pl.DataFrame) -> pl.DataFrame:
    s1_cols = ["source1_entity_id", "country", "name_clean", "name_latin",
               "name_dev_ratio", "name_missing",
               "addr_clean", "addr_latin", "addr_missing",
               "addr_city", "addr_state",
               "addr_first_word", "addr_last_word",
               "addr_house_number", "addr_road", "addr_unit", "addr_suburb"]
    s1_view = s1_meta.select([c for c in s1_cols if c in s1_meta.columns]).rename({
        "country": "s1_country", "name_clean": "s1_name_clean",
        "name_latin": "s1_name_latin", "name_dev_ratio": "s1_name_dev_ratio",
        "name_missing": "s1_name_missing", "addr_clean": "s1_addr_clean",
        "addr_latin": "s1_addr_latin", "addr_missing": "s1_addr_missing",
        "addr_city": "s1_addr_city", "addr_state": "s1_addr_state",
        "addr_first_word": "s1_addr_first_word", "addr_last_word": "s1_addr_last_word",
        "addr_house_number": "s1_addr_house_number", "addr_road": "s1_addr_road",
        "addr_unit": "s1_addr_unit", "addr_suburb": "s1_addr_suburb",
    })
    pairs = pairs.join(s1_view, on="source1_entity_id", how="left")

    cand_cols = [c for c in s1_cols if c != "source1_entity_id"]
    cand_view = cand_df.select(["entity_id"] + cand_cols).rename({
        "entity_id": "_m_join_id", "country": "m__country",
        "name_clean": "m__name_clean", "name_latin": "m__name_latin",
        "name_dev_ratio": "m__name_dev_ratio", "name_missing": "m_name_missing",
        "addr_clean": "m__addr_clean", "addr_latin": "m__addr_latin",
        "addr_missing": "m_addr_missing", "addr_city": "m__addr_city",
        "addr_state": "m__addr_state", "addr_first_word": "m__addr_first_word",
        "addr_last_word": "m__addr_last_word",
        "addr_house_number": "m__addr_house_number", "addr_road": "m__addr_road",
        "addr_unit": "m__addr_unit", "addr_suburb": "m__addr_suburb",
    })
    pairs = pairs.join(cand_view, left_on="candidate_entity_id",
                       right_on="_m_join_id", how="left")
    return pairs


# ---------------------------------------------------------------------------
# Features for liberal floors E/F/G/H
# ---------------------------------------------------------------------------
def compute_basic_features(df: pl.DataFrame) -> pl.DataFrame:
    """Cheap polars-side features for floors E/F/G/H."""
    return df.with_columns([
        # name_token_jaccard
        pl.when(
            (pl.col("s1_name_clean").is_not_null()) & (pl.col("m__name_clean").is_not_null())
        ).then(
            # |A ∩ B| / |A ∪ B|  via set op on token lists
            (pl.col("s1_name_clean").fill_null("").str.split(" ").list.set_difference([""])
             .list.intersection(pl.col("m__name_clean").fill_null("").str.split(" ").list.set_difference([""]))
             .list.len()
             /
             pl.col("s1_name_clean").fill_null("").str.split(" ").list.set_difference([""])
             .list.union(pl.col("m__name_clean").fill_null("").str.split(" ").list.set_difference([""]))
             .list.len())
        ).otherwise(0.0).cast(pl.Float32).alias("name_token_jaccard"),
        # addr_token_jaccard (analog)
        pl.when(
            (pl.col("s1_addr_clean").is_not_null()) & (pl.col("m__addr_clean").is_not_null())
        ).then(
            (pl.col("s1_addr_clean").fill_null("").str.split(" ").list.set_difference([""])
             .list.intersection(pl.col("m__addr_clean").fill_null("").str.split(" ").list.set_difference([""]))
             .list.len()
             /
             pl.col("s1_addr_clean").fill_null("").str.split(" ").list.set_difference([""])
             .list.union(pl.col("m__addr_clean").fill_null("").str.split(" ").list.set_difference([""]))
             .list.len())
        ).otherwise(0.0).cast(pl.Float32).alias("addr_token_jaccard"),
        # cross_script_pair
        pl.when(
            ((pl.col("s1_name_dev_ratio").fill_null(0.0) > 0.2) & (pl.col("m__name_dev_ratio").fill_null(0.0) <= 0.2))
            | ((pl.col("s1_name_dev_ratio").fill_null(0.0) <= 0.2) & (pl.col("m__name_dev_ratio").fill_null(0.0) > 0.2))
        ).then(1).otherwise(0).cast(pl.Int8).alias("cross_script_pair"),
        # addr_city_eq
        pl.when(
            (pl.col("s1_addr_city").is_not_null()) & (pl.col("m__addr_city").is_not_null())
            & (pl.col("s1_addr_city") == pl.col("m__addr_city"))
        ).then(1).otherwise(0).cast(pl.Int8).alias("addr_city_eq"),
    ])


def compute_name_fuzzy_features(df: pl.DataFrame) -> pl.DataFrame:
    """name_partial_ratio via rapidfuzz (Python loop; bounded by candidate count)."""
    if df.is_empty():
        return df.with_columns(pl.lit(0.0).cast(pl.Float32).alias("name_partial_ratio"))
    rows = df.select(["s1_name_clean", "m__name_clean"]).to_dicts()
    partial = np.zeros(len(rows), dtype=np.float32)
    for i, r in enumerate(rows):
        a = _s(r.get("s1_name_clean")); b = _s(r.get("m__name_clean"))
        if a and b:
            partial[i] = float(fuzz.partial_ratio(a, b))
    return df.with_columns(pl.Series("name_partial_ratio", partial, dtype=pl.Float32))


# ---------------------------------------------------------------------------
# Quality floors
# ---------------------------------------------------------------------------
def apply_floor(df: pl.DataFrame, floor_name: str) -> pl.DataFrame:
    """Apply one of the named quality floors; return filtered df."""
    if floor_name == "v3":
        # Floors A/B/C/D
        return df.filter(
            (pl.col("n_struct_keys").fill_null(0) >= 2)
            | (pl.col("n_tokens_shared").fill_null(0) >= 2)
            | (
                pl.col("s1_addr_road").is_not_null() & pl.col("m__addr_road").is_not_null()
                & (pl.col("s1_addr_road") != "") & (pl.col("m__addr_road") != "")
                & (pl.col("s1_addr_road") == pl.col("m__addr_road"))
                & pl.col("s1_addr_city").is_not_null() & pl.col("m__addr_city").is_not_null()
                & (pl.col("s1_addr_city") != "") & (pl.col("m__addr_city") != "")
                & (pl.col("s1_addr_city") == pl.col("m__addr_city"))
            )
            | (
                (pl.col("sortedn_rank").fill_null(0) > 0) & (pl.col("sortedn_rank") <= 10)
                & (pl.col("n_tokens_shared").fill_null(0) >= 1)
            )
        )
    elif floor_name == "liberal_v1":
        # v3 floors + E/F/G/H
        return df.filter(
            (pl.col("n_struct_keys").fill_null(0) >= 2)
            | (pl.col("n_tokens_shared").fill_null(0) >= 2)
            | (
                pl.col("s1_addr_road").is_not_null() & pl.col("m__addr_road").is_not_null()
                & (pl.col("s1_addr_road") != "") & (pl.col("m__addr_road") != "")
                & (pl.col("s1_addr_road") == pl.col("m__addr_road"))
                & pl.col("s1_addr_city").is_not_null() & pl.col("m__addr_city").is_not_null()
                & (pl.col("s1_addr_city") != "") & (pl.col("m__addr_city") != "")
                & (pl.col("s1_addr_city") == pl.col("m__addr_city"))
            )
            | (
                (pl.col("sortedn_rank").fill_null(0) > 0) & (pl.col("sortedn_rank") <= 10)
                & (pl.col("n_tokens_shared").fill_null(0) >= 1)
            )
            | (pl.col("name_partial_ratio").fill_null(0.0) >= 80)
            | (pl.col("name_token_jaccard").fill_null(0.0) >= 0.5)
            | (pl.col("addr_token_jaccard").fill_null(0.0) >= 0.4)
            | (
                (pl.col("cross_script_pair").fill_null(0) == 1)
                & (pl.col("n_tokens_shared").fill_null(0) >= 1)
                & (pl.col("addr_city_eq").fill_null(0) == 1)
            )
        )
    elif floor_name == "liberal_v2":
        # liberal_v1 with Floor B relaxed (>= 1 instead of >= 2)
        return df.filter(
            (pl.col("n_struct_keys").fill_null(0) >= 2)
            | (pl.col("n_tokens_shared").fill_null(0) >= 1)
            | (
                pl.col("s1_addr_road").is_not_null() & pl.col("m__addr_road").is_not_null()
                & (pl.col("s1_addr_road") != "") & (pl.col("m__addr_road") != "")
                & (pl.col("s1_addr_road") == pl.col("m__addr_road"))
                & pl.col("s1_addr_city").is_not_null() & pl.col("m__addr_city").is_not_null()
                & (pl.col("s1_addr_city") != "") & (pl.col("m__addr_city") != "")
                & (pl.col("s1_addr_city") == pl.col("m__addr_city"))
            )
            | (
                (pl.col("sortedn_rank").fill_null(0) > 0) & (pl.col("sortedn_rank") <= 10)
                & (pl.col("n_tokens_shared").fill_null(0) >= 1)
            )
            | (pl.col("name_partial_ratio").fill_null(0.0) >= 80)
            | (pl.col("name_token_jaccard").fill_null(0.0) >= 0.5)
            | (pl.col("addr_token_jaccard").fill_null(0.0) >= 0.4)
            | (
                (pl.col("cross_script_pair").fill_null(0) == 1)
                & (pl.col("n_tokens_shared").fill_null(0) >= 1)
                & (pl.col("addr_city_eq").fill_null(0) == 1)
            )
        )
    elif floor_name == "A_only":
        return df.filter(pl.col("n_struct_keys").fill_null(0) >= 1)
    else:
        raise ValueError(f"Unknown floor: {floor_name}")


# ---------------------------------------------------------------------------
# Recall computation
# ---------------------------------------------------------------------------
def compute_recall(cand_pairs: pl.DataFrame, gt: dict[str, set[str]],
                   s1_meta: pl.DataFrame) -> dict:
    """Compute per-method recall metrics vs ground truth."""
    # s1_meta has source1_entity_id + s1_country
    # cand_pairs has source1_entity_id + candidate_entity_id
    s1_ids = s1_meta["source1_entity_id"].to_list()
    s1_country_map = dict(zip(s1_meta["source1_entity_id"].to_list(),
                               s1_meta["s1_country"].to_list()))

    # Build predicted set per S1
    pred: dict[str, set[str]] = {sid: set() for sid in s1_ids}
    if not cand_pairs.is_empty():
        for row in cand_pairs.select(["source1_entity_id", "candidate_entity_id"]).iter_rows(named=True):
            pred[row["source1_entity_id"]].add(row["candidate_entity_id"])

    per_s1: list[float] = []
    per_country: dict[str, list[float]] = {"US": [], "IN": []}
    n_singletons = 0
    for sid in s1_ids:
        true_set = gt.get(sid, set())
        if not true_set:
            n_singletons += 1
            continue
        pred_set = pred.get(sid, set())
        hit = len(true_set & pred_set)
        rec = hit / len(true_set)
        per_s1.append(rec)
        c = s1_country_map.get(sid, "US")
        per_country[c].append(rec)

    if not per_s1:
        return {"mean": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0,
                "max": 0.0, "pct_perfect": 0.0, "n_nonsingleton": 0,
                "us_mean": 0.0, "in_mean": 0.0, "mean_cands": 0.0}

    arr = np.array(per_s1)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(arr.max()),
        "pct_perfect": float((arr >= 1.0).mean()),
        "n_nonsingleton": len(per_s1),
        "us_mean": float(np.mean(per_country["US"])) if per_country["US"] else 0.0,
        "in_mean": float(np.mean(per_country["IN"])) if per_country["IN"] else 0.0,
        "mean_cands": float(np.mean([len(p) for p in pred.values()])),
    }


# ---------------------------------------------------------------------------
# Per-method runner
# ---------------------------------------------------------------------------
# 8 default struct keys (the v3 set)
DEFAULT_STRUCT_KEYS = ["city", "state", "road", "house", "name_fw", "addr_fw",
                       "city_state", "house_road"]
NGRAM_KEYS = ["name_ngram_key", "addr_ngram_key"]


METHODS = [
    # Group A: single-key baselines
    {"name": "M01_name_fw_only",         "struct_keys": ["name_fw"], "token": False, "latin": False, "sortedn": False, "ngram": False, "floor": "A_only",    "cap": 100},
    {"name": "M02_addr_fw_only",         "struct_keys": ["addr_fw"], "token": False, "latin": False, "sortedn": False, "ngram": False, "floor": "A_only",    "cap": 100},
    {"name": "M03_city_only",            "struct_keys": ["city"],    "token": False, "latin": False, "sortedn": False, "ngram": False, "floor": "A_only",    "cap": 100},
    {"name": "M04_name_fw+addr_fw",      "struct_keys": ["name_fw", "addr_fw"], "token": False, "latin": False, "sortedn": False, "ngram": False, "floor": "A_only", "cap": 100},

    # Group B: v3 structural combos
    {"name": "M05_8struct_only",         "struct_keys": DEFAULT_STRUCT_KEYS, "token": False, "latin": False, "sortedn": False, "ngram": False, "floor": "v3", "cap": 50},
    {"name": "M06_8struct+token",        "struct_keys": DEFAULT_STRUCT_KEYS, "token": True,  "latin": False, "sortedn": False, "ngram": False, "floor": "v3", "cap": 50},
    {"name": "M07_8struct+token+sortedn_v3", "struct_keys": DEFAULT_STRUCT_KEYS, "token": True, "latin": False, "sortedn": True, "ngram": False, "floor": "v3", "cap": 50},

    # Group C: ngram keys added on top of v3
    {"name": "M08_M07+ngram_keys",       "struct_keys": DEFAULT_STRUCT_KEYS + NGRAM_KEYS, "token": True, "latin": False, "sortedn": True, "ngram": True, "floor": "v3", "cap": 50},

    # Group D: Latin tokens added
    {"name": "M09_M07+latin_tokens",     "struct_keys": DEFAULT_STRUCT_KEYS, "token": True, "latin": True, "sortedn": True, "ngram": False, "floor": "v3", "cap": 50},
    {"name": "M10_M08+latin_tokens",     "struct_keys": DEFAULT_STRUCT_KEYS + NGRAM_KEYS, "token": True, "latin": True, "sortedn": True, "ngram": True, "floor": "v3", "cap": 50},

    # Group E: liberal floors on best Group D methods
    {"name": "M11_M09+liberal_v1",       "struct_keys": DEFAULT_STRUCT_KEYS, "token": True, "latin": True, "sortedn": True, "ngram": False, "floor": "liberal_v1", "cap": 50},
    {"name": "M12_M10+liberal_v1",       "struct_keys": DEFAULT_STRUCT_KEYS + NGRAM_KEYS, "token": True, "latin": True, "sortedn": True, "ngram": True, "floor": "liberal_v1", "cap": 50},

    # Group F: cap bumps (50 → 100)
    {"name": "M13_M11+cap100",           "struct_keys": DEFAULT_STRUCT_KEYS, "token": True, "latin": True, "sortedn": True, "ngram": False, "floor": "liberal_v1", "cap": 100},
    {"name": "M14_M12+cap100",           "struct_keys": DEFAULT_STRUCT_KEYS + NGRAM_KEYS, "token": True, "latin": True, "sortedn": True, "ngram": True, "floor": "liberal_v1", "cap": 100},

    # Group G: Floor B relaxed (>=1 token)
    {"name": "M15_M14+floor_B_relaxed",  "struct_keys": DEFAULT_STRUCT_KEYS + NGRAM_KEYS, "token": True, "latin": True, "sortedn": True, "ngram": True, "floor": "liberal_v2", "cap": 100},

    # Group H: proposed v4 production setting (best of everything so far)
    {"name": "M16_v4_production",        "struct_keys": DEFAULT_STRUCT_KEYS + NGRAM_KEYS, "token": True, "latin": True, "sortedn": True, "ngram": True, "floor": "liberal_v2", "cap": 100},
]


def run_method(method: dict, s1_meta: pl.DataFrame, cand_df: pl.DataFrame,
               gt: dict[str, set[str]]) -> dict:
    """Run one method end-to-end. Return {metrics, n_cands}."""
    name = method["name"]
    print(f"\n[{name}] struct_keys={len(method['struct_keys'])} token={method['token']} "
          f"latin={method['latin']} sortedn={method['sortedn']} ngram={method['ngram']} "
          f"floor={method['floor']} cap={method['cap']}", flush=True)
    t0 = time.time()

    # 1. Build indices (subset)
    inv = build_structural_indexes_subset(
        cand_df,
        make_key_extractors(method["struct_keys"]),
        cap=500,
    )
    token_idx = (build_token_index_subset(cand_df, cap=500, include_latin=method["latin"])
                 if method["token"] else None)
    sn = (build_sorted_neighborhood_subset(cand_df)
          if method["sortedn"] else (np.array([], dtype=object), np.array([], dtype=object)))

    # 2. Probe 10K S1
    s1_chunk = s1_meta  # for ablation, no chunking needed
    struct_df = probe_structural_subset(s1_chunk, inv, top_k=200) if inv else None
    token_df = probe_tokens_subset(s1_chunk, token_idx, top_k=200,
                                   include_latin=method["latin"]) if token_idx is not None else None
    sn_df = probe_sorted_neighborhood_subset(s1_chunk, sn[0], sn[1],
                                             window=50) if method["sortedn"] else None

    # 3. Union (full outer join on pair key)
    pairs = struct_df if struct_df is not None else pl.DataFrame(
        schema={"source1_entity_id": pl.Utf8, "candidate_entity_id": pl.Utf8}
    )
    if token_df is not None:
        pairs = pairs.join(token_df, on=["source1_entity_id", "candidate_entity_id"],
                            how="full", coalesce=True)
    if sn_df is not None:
        pairs = pairs.join(sn_df, on=["source1_entity_id", "candidate_entity_id"],
                            how="full", coalesce=True)
    pairs = pairs.with_columns([
        pl.col("n_struct_keys").fill_null(0).cast(pl.Int8),
        pl.col("n_tokens_shared").fill_null(0).cast(pl.Int8),
        pl.col("sortedn_rank").fill_null(0).cast(pl.Int16),
    ])
    print(f"   probe union: {pairs.height:,} pairs", flush=True)

    # 4. Attach fields
    pairs = attach_fields(pairs, s1_meta, cand_df)

    # 5. Country filter
    pairs = pairs.filter(
        pl.col("s1_country").is_not_null() & pl.col("m__country").is_not_null()
        & (pl.col("s1_country") == pl.col("m__country"))
    )
    print(f"   after country: {pairs.height:,} pairs", flush=True)

    # 6. Compute features for liberal floors
    if method["floor"] in ("liberal_v1", "liberal_v2"):
        pairs = compute_basic_features(pairs)
        pairs = compute_name_fuzzy_features(pairs)
    else:
        # For v3 / A_only floors, no extra features needed
        if "name_token_jaccard" not in pairs.columns:
            pairs = pairs.with_columns(pl.lit(0.0).cast(pl.Float32).alias("name_token_jaccard"))

    # 7. Apply quality floor
    pairs = apply_floor(pairs, method["floor"])
    print(f"   after floor: {pairs.height:,} pairs", flush=True)

    # 8. Top-K cap per S1 (no scoring for ablation — just keep first K per S1)
    pairs = (pairs
        .with_columns(pl.lit(1).alias("_dummy_score"))
        .with_columns(pl.col("_dummy_score").rank(method="ordinal", descending=True)
                          .over("source1_entity_id").alias("_r"))
        .filter(pl.col("_r") <= method["cap"])
        .drop("_r", "_dummy_score"))

    n_cands = pairs.height

    # 9. Recall
    metrics = compute_recall(pairs, gt, s1_meta)
    elapsed = time.time() - t0
    print(f"   mean_recall={metrics['mean']:.4f}  median={metrics['median']:.4f}  "
          f"pct_perfect={metrics['pct_perfect']*100:.1f}%  "
          f"n_cands={n_cands:,}  mean_cands/S1={metrics['mean_cands']:.1f}  "
          f"({elapsed:.1f}s)", flush=True)

    return {
        "method": name,
        "struct_keys": str(method["struct_keys"]),
        "token": method["token"], "latin": method["latin"], "sortedn": method["sortedn"],
        "ngram": method["ngram"], "floor": method["floor"], "cap": method["cap"],
        **metrics,
        "n_cands": n_cands,
        "elapsed_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sample-size", type=int, default=10000)
    p.add_argument("--output", type=Path, default=ARTIFACTS / "ablation_full.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-s1-meta-cache", type=int, default=100000,
                   help="Cap S1 sample size (for smoke testing)")
    args = p.parse_args()

    print("=" * 70, flush=True)
    print(f"Ablation: {len(METHODS)} methods, sample={args.sample_size:,}, seed={args.seed}",
          flush=True)
    print("=" * 70, flush=True)

    s1, s2, s3 = load_sources()
    gt = load_ground_truth()
    s1_meta = sample_s1_ids(s1, gt, args.sample_size, args.seed)

    # We need a richer s1_meta with all the fields needed for probing.
    # Join the sampled ids back with the full s1 fields.
    s1_full = s1.select([
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("country").alias("s1_country"),
        "name_clean", "name_latin", "name_dev_ratio", "name_missing",
        "addr_clean", "addr_latin", "addr_missing",
        "addr_first_word", "addr_last_word",
        "addr_city", "addr_state",
        "addr_house_number", "addr_road", "addr_unit", "addr_suburb",
        "name_ngram_key", "addr_ngram_key",
    ])
    s1_meta = s1_meta.select(["source1_entity_id"]).join(s1_full, on="source1_entity_id", how="left")

    # Candidate pool = union of s2 + s3
    cand_df = pl.concat([
        s2.select([c for c in s2.columns if c != "name_tokens"]),
        s3.select([c for c in s3.columns if c != "name_tokens"]),
    ], how="vertical")

    results: list[dict] = []
    for method in METHODS:
        try:
            row = run_method(method, s1_meta, cand_df, gt)
            results.append(row)
        except Exception as e:
            print(f"   ERROR: {type(e).__name__}: {e}", flush=True)
            results.append({
                "method": method["name"], "error": str(e),
                "mean": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0,
                "max": 0.0, "pct_perfect": 0.0, "n_nonsingleton": 0,
                "us_mean": 0.0, "in_mean": 0.0, "mean_cands": 0.0,
                "n_cands": 0, "elapsed_s": 0.0,
            })
        gc.collect()

    # Write CSV
    results_df = pl.DataFrame(results)
    results_df.write_csv(args.output)
    print("\n" + "=" * 70, flush=True)
    print(f"Wrote {args.output} ({len(results)} rows)", flush=True)
    print("=" * 70, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
