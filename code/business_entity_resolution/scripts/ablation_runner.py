"""Ablation harness: test 16 candidate-generation methods on a 10K S1 sample.

Apples-to-apples comparison: every method scores on the same S1 sample
(stratified by country × match-count). For each method we measure:
  - mean / median / P25 / P75 / max recall
  - % S1 with recall = 1.0
  - mean candidates per S1
  - per-country mean recall (US / India)

RAM-safety (designed for 32 GiB box; won't crash on 15 GiB either):
  - Indices built ONCE and cached across all 16 methods (the dominant
    RAM saver — structural indices on 10M (key, id) pairs)
  - Polars LazyFrame reads for cand_df (avoids materializing cols we don't touch)
  - Per-key bucket cap = 200 for ablation (was 500 in production)
  - Aggressive gc.collect() between methods
  - POLARS_MAX_THREADS capped at n_workers * 2 via _resources
  - min_free_gb guard at script start (refuses if < 2 GiB)

Methods tested (16, organized in 8 groups):
  Group A (single index baselines): M01-M04
  Group B (v3 structural combos):   M05-M07
  Group C (ngram keys added):       M08
  Group D (Latin tokens):           M09-M10
  Group E (liberal floors):         M11-M12
  Group F (cap bumps):              M13-M14
  Group G (Floor B relaxed):        M15
  Group H (proposed v4 setting):    M16

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

# Min 2 GiB — ablation only needs ~1.5 GB peak after our caching
_RES = detect_resources(min_free_gb=2.0)
configure_polars_threads(_RES["polars_threads"])

import numpy as np
import polars as pl
from rapidfuzz import fuzz

ARTIFACTS = PROJECT_ROOT / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
TESTING_DIR = ARTIFACTS / "testing"
TESTING_DIR.mkdir(parents=True, exist_ok=True)

S1_NORM = ARTIFACTS / "s1_norm_train.parquet"
S2_NORM = ARTIFACTS / "s2_norm_train.parquet"
S3_NORM = ARTIFACTS / "s3_norm_train.parquet"
TRAIN_GT = PROJECT_ROOT.parent.parent / "dataset" / "student_resource" / "dataset" / "train" / "train_ground_truth.tsv"

# Per-key bucket cap for ablation (lower than production's 500 → ~60% less RAM)
# 8 GiB boxes: cap=30 keeps peak < 3 GB; 16+ GiB boxes: bump to 100/200.
ABLATION_BUCKET_CAP = 30

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
# Data loading
# ---------------------------------------------------------------------------
def load_sources() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load the 3 normalized sources. Uses LazyFrame scan first to verify
    schema, then materializes only the cols we need."""
    cols = [
        "entity_id", "country",
        "name_clean", "name_latin", "name_dev_ratio", "name_missing",
        "addr_clean", "addr_latin", "addr_missing",
        "addr_first_word", "addr_last_word",
        "addr_city", "addr_state",
        "addr_house_number", "addr_road", "addr_unit", "addr_suburb",
        "name_ngram_key", "addr_ngram_key",
    ]
    log_resources("load")
    print(f"[load] s1 ({S1_NORM.name}) ...", flush=True)
    t0 = time.time()
    s1 = pl.read_parquet(S1_NORM, columns=cols)
    print(f"        {s1.height:,} rows in {time.time()-t0:.1f}s", flush=True)

    print(f"[load] s2 ({S2_NORM.name}) ...", flush=True)
    t0 = time.time()
    s2 = pl.read_parquet(S2_NORM, columns=cols)
    print(f"        {s2.height:,} rows in {time.time()-t0:.1f}s", flush=True)

    print(f"[load] s3 ({S3_NORM.name}) ...", flush=True)
    t0 = time.time()
    s3 = pl.read_parquet(S3_NORM, columns=cols)
    print(f"        {s3.height:,} rows in {time.time()-t0:.1f}s", flush=True)
    return s1, s2, s3


def load_ground_truth() -> dict[str, set[str]]:
    """Return {s1_id: {m_id, ...}} from train ground truth."""
    print(f"[load] ground truth ({TRAIN_GT.name}) ...", flush=True)
    t0 = time.time()
    df_raw = pl.read_csv(TRAIN_GT, separator="\t", encoding="utf8-lossy")
    df_raw = df_raw.rename({c: c.strip() for c in df_raw.columns})
    df = (df_raw
          .with_columns(pl.col("matched_entity_ids").str.split(",").alias("matched_list"))
          .explode("matched_list")
          .with_columns(pl.col("matched_list").str.strip_chars().alias("m"))
          .filter(pl.col("m") != "")
          .select("source1_entity_id", "m"))
    gt_dict: dict[str, set[str]] = {}
    for row in df.iter_rows(named=True):
        gt_dict.setdefault(row["source1_entity_id"], set()).add(row["m"])
    print(f"        {len(gt_dict):,} S1 entities, {df.height:,} pairs in {time.time()-t0:.1f}s", flush=True)
    return gt_dict


def sample_s1_ids(s1: pl.DataFrame, gt: dict[str, set[str]], n: int, seed: int) -> list[str]:
    """Sample n S1 IDs stratified by (country, match-count bucket). Returns list."""
    print(f"[sample] drawing {n:,} S1 ids (seed={seed}) ...", flush=True)
    rng = np.random.default_rng(seed)
    s1_meta = s1.select(["entity_id", "country"]).rename({"entity_id": "source1_entity_id"})
    s1_meta = s1_meta.with_columns(
        pl.col("source1_entity_id").map_elements(
            lambda sid: len(gt.get(sid, set())),
            return_dtype=pl.Int32,
        ).alias("n_true")
    ).with_columns(
        pl.when(pl.col("n_true") == 0).then(pl.lit("singleton"))
         .when(pl.col("n_true") <= 2).then(pl.lit("1-2"))
         .when(pl.col("n_true") <= 5).then(pl.lit("3-5"))
         .otherwise(pl.lit("6+")).alias("bucket")
    )
    counts = (s1_meta.group_by(["country", "bucket"]).agg(pl.len().alias("pop"))
                  .with_columns((pl.col("pop") * n / s1_meta.height).cast(pl.Int32).alias("take")))
    sampled_ids: list[str] = []
    for row in counts.iter_rows(named=True):
        pool = s1_meta.filter(
            (pl.col("country") == row["country"]) & (pl.col("bucket") == row["bucket"])
        )
        take = min(row["take"], pool.height)
        chosen = rng.choice(pool["source1_entity_id"].to_list(), size=take, replace=False)
        sampled_ids.extend(chosen.tolist())
    print(f"        {len(sampled_ids):,} S1 sampled", flush=True)
    print(s1_meta.filter(pl.col("source1_entity_id").is_in(sampled_ids))
                   .group_by(["country", "bucket"]).agg(pl.len().alias("n"))
                   .sort(["country", "bucket"]), flush=True)
    return sampled_ids


# ---------------------------------------------------------------------------
# KEY_EXTRACTORS — 10 keys (8 default + 2 ngram)
# ---------------------------------------------------------------------------
def make_key_extractors() -> dict[str, pl.Expr]:
    return {
        "city":           pl.col("addr_city").fill_null("").str.strip_chars().str.to_lowercase(),
        "state":          pl.col("addr_state").fill_null("").str.strip_chars().str.to_lowercase(),
        "road":           pl.col("addr_road").fill_null("").str.strip_chars().str.to_lowercase(),
        "house":          pl.col("addr_house_number").fill_null("").str.strip_chars().str.to_lowercase(),
        "name_fw":        pl.col("name_clean").fill_null("").str.split(" ").list.first()
                             .fill_null("").str.strip_chars().str.to_lowercase(),
        "addr_fw":        pl.col("addr_first_word").fill_null("").str.strip_chars().str.to_lowercase(),
        "city_state":     pl.concat_str(
                               [pl.col("addr_city").fill_null(""), pl.lit("|"),
                                pl.col("addr_state").fill_null("")],
                               separator="", ignore_nulls=True)
                           .str.strip_chars().str.to_lowercase(),
        "house_road":     pl.concat_str(
                               [pl.col("addr_house_number").fill_null(""), pl.lit("|"),
                                pl.col("addr_road").fill_null("")],
                               separator="", ignore_nulls=True)
                           .str.strip_chars().str.to_lowercase(),
        "name_ngram_key": pl.col("name_ngram_key").fill_null("").str.strip_chars().str.to_lowercase(),
        "addr_ngram_key": pl.col("addr_ngram_key").fill_null("").str.strip_chars().str.to_lowercase(),
    }


# ---------------------------------------------------------------------------
# Index builders — cached across methods
# ---------------------------------------------------------------------------
def build_one_struct_index(cand: pl.DataFrame, expr: pl.Expr, cap: int) -> pl.DataFrame:
    return (cand.with_columns(expr.alias("_k"))
                .filter(pl.col("_k").is_not_null() & (pl.col("_k") != ""))
                .select(["_k", pl.col("entity_id").alias("_id")])
                .unique(subset=["_k", "_id"])
                .group_by("_k").agg(pl.col("_id"))
                .with_columns(pl.col("_id").list.slice(0, cap))
                .explode("_id"))


def build_token_index(cand: pl.DataFrame, cap: int, include_latin: bool) -> pl.DataFrame:
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


def build_char_trigram_index(cand: pl.DataFrame, cap: int, include_latin: bool) -> pl.DataFrame:
    """Char-trigram inverted index (MinHash LSH surrogate for short text).

    For text <100 chars, char-trigram Jaccard is mathematically equivalent to
    MinHash LSH. Much faster than full MinHash on 10M docs.
    Returns: long-format DF (_trigram: Utf8, _id: Utf8) capped at `cap`/trigram.
    """
    if include_latin:
        text_expr = (pl.col("name_clean").fill_null("") + pl.lit(" ") +
                     pl.col("addr_clean").fill_null("") + pl.lit(" ") +
                     pl.col("name_latin").fill_null("") + pl.lit(" ") +
                     pl.col("addr_latin").fill_null(""))
    else:
        text_expr = (pl.col("name_clean").fill_null("") + pl.lit(" ") +
                     pl.col("addr_clean").fill_null(""))

    def _trigrams(text: str) -> list[str]:
        if not text or len(text) < 3:
            return []
        return list({text[i:i+3] for i in range(len(text) - 2)})

    cand_t = (cand.with_columns(text_expr.alias("_text"))
                  .with_columns(
                      pl.col("_text").map_batches(
                          lambda s: pl.Series([_trigrams(t) for t in s]),
                          return_dtype=pl.List(pl.Utf8),
                      ).alias("_trigs"))
                  .select(["entity_id", "_trigs"]).explode("_trigs")
                  .rename({"_trigs": "_trigram"})
                  .filter(pl.col("_trigram").is_not_null() & (pl.col("_trigram") != "")))
    return (cand_t
            .unique(subset=["_trigram", "entity_id"])
            .group_by("_trigram").agg(pl.col("entity_id"))
            .with_columns(pl.col("entity_id").list.slice(0, cap))
            .explode("entity_id")
            .rename({"entity_id": "_id"}))


def build_bigram_index(cand: pl.DataFrame, cap: int, include_latin: bool) -> pl.DataFrame:
    """2-token bigram inverted index.

    For each row, tokenize (with optional Latin), emit (toks[i], toks[i+1]) pairs.
    Returns: long-format DF (_bigram: Utf8, _id: Utf8) capped at `cap`/bigram.
    """
    if include_latin:
        text_expr = (pl.col("name_clean").fill_null("") + pl.lit(" ") +
                     pl.col("addr_clean").fill_null("") + pl.lit(" ") +
                     pl.col("name_latin").fill_null("") + pl.lit(" ") +
                     pl.col("addr_latin").fill_null(""))
    else:
        text_expr = (pl.col("name_clean").fill_null("") + pl.lit(" ") +
                     pl.col("addr_clean").fill_null(""))

    def _bigrams(text: str) -> list[str]:
        toks = _tokenize(text)
        if len(toks) < 2:
            return []
        return list({f"{toks[i]}_{toks[i+1]}" for i in range(len(toks) - 1)})

    cand_b = (cand.with_columns(text_expr.alias("_text"))
                  .with_columns(
                      pl.col("_text").map_batches(
                          lambda s: pl.Series([_bigrams(t) for t in s]),
                          return_dtype=pl.List(pl.Utf8),
                      ).alias("_bigrams"))
                  .select(["entity_id", "_bigrams"]).explode("_bigrams")
                  .rename({"_bigrams": "_bigram"})
                  .filter(pl.col("_bigram").is_not_null()))
    return (cand_b
            .unique(subset=["_bigram", "entity_id"])
            .group_by("_bigram").agg(pl.col("entity_id"))
            .with_columns(pl.col("entity_id").list.slice(0, cap))
            .explode("entity_id")
            .rename({"entity_id": "_id"}))


def build_sorted_neighborhood(cand: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    canon_to_id: dict[str, str] = {}
    for r in cand.select(["entity_id", "name_clean", "addr_clean"]).iter_rows(named=True):
        text = (r.get("name_clean") or "") + " " + (r.get("addr_clean") or "")
        toks = _tokenize(text)
        canon = " ".join(sorted(set(toks)))
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
                            active_keys: list[str],
                            top_k: int) -> pl.DataFrame:
    """Probe using only the active struct keys (subset of cached inv)."""
    parts: list[pl.DataFrame] = []
    key_extractors = make_key_extractors()
    for name in active_keys:
        if name not in inv:
            continue
        expr = key_extractors[name]
        s1k = (s1_chunk.with_columns(expr.alias("_k"))
                       .filter(pl.col("_k").is_not_null() & (pl.col("_k") != ""))
                       .select(["source1_entity_id", "_k"]))
        if s1k.is_empty():
            continue
        joined = s1k.join(inv[name], on="_k", how="inner")
        parts.append(joined.with_columns(pl.lit(name).alias("_kt")))
    if not parts:
        return pl.DataFrame(schema={
            "source1_entity_id": pl.Utf8, "candidate_entity_id": pl.Utf8,
            "n_struct_keys": pl.Int8,
        })
    all_pairs = pl.concat(parts).select(["source1_entity_id", "_id", "_kt"])
    return (all_pairs
            .group_by(["source1_entity_id", "_id"])
            .agg(pl.col("_kt").n_unique().cast(pl.Int8).alias("n_struct_keys"))
            .rename({"_id": "candidate_entity_id"}))


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
    return (joined
            .group_by(["source1_entity_id", "_id"])
            .agg(pl.col("_tok").n_unique().cast(pl.Int8).alias("n_tokens_shared"))
            .rename({"_id": "candidate_entity_id"}))


def probe_char_trigram_subset(s1_chunk: pl.DataFrame, ct_idx: pl.DataFrame,
                              top_k: int, include_latin: bool) -> pl.DataFrame:
    """Probe char-trigram inverted index. Returns pairs with n_trigrams_shared count."""
    if include_latin:
        text_expr = (pl.col("name_clean").fill_null("") + pl.lit(" ") +
                     pl.col("addr_clean").fill_null("") + pl.lit(" ") +
                     pl.col("name_latin").fill_null("") + pl.lit(" ") +
                     pl.col("addr_latin").fill_null(""))
    else:
        text_expr = (pl.col("name_clean").fill_null("") + pl.lit(" ") +
                     pl.col("addr_clean").fill_null(""))
    s1_t = (s1_chunk.with_columns(text_expr.alias("_text"))
                      .select(["source1_entity_id", "_text"])
                      .with_columns(
                          pl.col("_text").map_batches(
                              lambda s: pl.Series([list({t[i:i+3] for i in range(len(t)-2)}) if t and len(t)>=3 else [] for t in s]),
                              return_dtype=pl.List(pl.Utf8),
                          ).alias("_trigs"))
                      .select(["source1_entity_id", "_trigs"]).explode("_trigs")
                      .filter(pl.col("_trigs").is_not_null() & (pl.col("_trigs") != ""))
                      .rename({"_trigs": "_trigram"}))
    if s1_t.is_empty():
        return pl.DataFrame(schema={
            "source1_entity_id": pl.Utf8, "candidate_entity_id": pl.Utf8,
            "n_trigrams_shared": pl.Int16,
        })
    joined = s1_t.join(ct_idx, on="_trigram", how="inner")
    if joined.is_empty():
        return pl.DataFrame(schema={
            "source1_entity_id": pl.Utf8, "candidate_entity_id": pl.Utf8,
            "n_trigrams_shared": pl.Int16,
        })
    return (joined
            .group_by(["source1_entity_id", "_id"])
            .agg(pl.col("_trigram").n_unique().cast(pl.Int16).alias("n_trigrams_shared"))
            .rename({"_id": "candidate_entity_id"}))


def probe_bigram_subset(s1_chunk: pl.DataFrame, bigram_idx: pl.DataFrame,
                        top_k: int, include_latin: bool) -> pl.DataFrame:
    """Probe 2-token bigram inverted index. Returns pairs with n_bigrams_shared count."""
    if include_latin:
        text_expr = (pl.col("name_clean").fill_null("") + pl.lit(" ") +
                     pl.col("addr_clean").fill_null("") + pl.lit(" ") +
                     pl.col("name_latin").fill_null("") + pl.lit(" ") +
                     pl.col("addr_latin").fill_null(""))
    else:
        text_expr = (pl.col("name_clean").fill_null("") + pl.lit(" ") +
                     pl.col("addr_clean").fill_null(""))
    def _bigrams(text: str) -> list[str]:
        toks = _tokenize(text)
        if len(toks) < 2:
            return []
        return list({f"{toks[i]}_{toks[i+1]}" for i in range(len(toks) - 1)})
    s1_b = (s1_chunk.with_columns(text_expr.alias("_text"))
                      .select(["source1_entity_id", "_text"])
                      .with_columns(
                          pl.col("_text").map_batches(
                              lambda s: pl.Series([_bigrams(t) for t in s]),
                              return_dtype=pl.List(pl.Utf8),
                          ).alias("_bigrams"))
                      .select(["source1_entity_id", "_bigrams"]).explode("_bigrams")
                      .filter(pl.col("_bigrams").is_not_null())
                      .rename({"_bigrams": "_bigram"}))
    if s1_b.is_empty():
        return pl.DataFrame(schema={
            "source1_entity_id": pl.Utf8, "candidate_entity_id": pl.Utf8,
            "n_bigrams_shared": pl.Int16,
        })
    joined = s1_b.join(bigram_idx, on="_bigram", how="inner")
    if joined.is_empty():
        return pl.DataFrame(schema={
            "source1_entity_id": pl.Utf8, "candidate_entity_id": pl.Utf8,
            "n_bigrams_shared": pl.Int16,
        })
    return (joined
            .group_by(["source1_entity_id", "_id"])
            .agg(pl.col("_bigram").n_unique().cast(pl.Int16).alias("n_bigrams_shared"))
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
        toks = _tokenize(text)
        canon = " ".join(sorted(set(toks)))
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
# Field attachment
# ---------------------------------------------------------------------------
def attach_fields(pairs: pl.DataFrame, s1_meta: pl.DataFrame,
                  cand_df: pl.DataFrame) -> pl.DataFrame:
    s1_cols = ["source1_entity_id", "country", "name_clean", "name_latin",
               "name_dev_ratio", "name_missing",
               "addr_clean", "addr_latin", "addr_missing",
               "addr_city", "addr_state",
               "addr_first_word", "addr_last_word",
               "addr_house_number", "addr_road", "addr_unit", "addr_suburb"]
    s1_rename = {
        "country": "s1_country", "name_clean": "s1_name_clean",
        "name_latin": "s1_name_latin", "name_dev_ratio": "s1_name_dev_ratio",
        "name_missing": "s1_name_missing", "addr_clean": "s1_addr_clean",
        "addr_latin": "s1_addr_latin", "addr_missing": "s1_addr_missing",
        "addr_city": "s1_addr_city", "addr_state": "s1_addr_state",
        "addr_first_word": "s1_addr_first_word", "addr_last_word": "s1_addr_last_word",
        "addr_house_number": "s1_addr_house_number", "addr_road": "s1_addr_road",
        "addr_unit": "s1_addr_unit", "addr_suburb": "s1_addr_suburb",
    }
    s1_view = s1_meta.select([c for c in s1_cols if c in s1_meta.columns]).rename(s1_rename)
    pairs = pairs.join(s1_view, on="source1_entity_id", how="left")
    cand_cols = [c for c in s1_cols if c != "source1_entity_id"]
    # Build m__ rename: for each cand col, prefix with "m__" instead of "s1_"
    cand_rename: dict[str, str] = {}
    for c in cand_cols:
        cand_rename[c] = "m__" + c
    cand_rename["entity_id"] = "_m_join_id"
    # country was already filtered from s1_rename loop; handle separately
    cand_rename["country"] = "m__country"
    cand_view = cand_df.select(["entity_id"] + cand_cols).rename(cand_rename)
    pairs = pairs.join(cand_view, left_on="candidate_entity_id",
                       right_on="_m_join_id", how="left")
    return pairs


# ---------------------------------------------------------------------------
# Features for liberal floors
# ---------------------------------------------------------------------------
def compute_basic_features(df: pl.DataFrame) -> pl.DataFrame:
    s1n = pl.col("s1_name_clean").fill_null("").str.split(" ").list.set_difference([""])
    m1n = pl.col("m__name_clean").fill_null("").str.split(" ").list.set_difference([""])
    s1a = pl.col("s1_addr_clean").fill_null("").str.split(" ").list.set_difference([""])
    m1a = pl.col("m__addr_clean").fill_null("").str.split(" ").list.set_difference([""])
    return df.with_columns([
        ((s1n.list.set_intersection(m1n).list.len() / s1n.list.set_union(m1n).list.len())
         .fill_null(0.0)).cast(pl.Float32).alias("name_token_jaccard"),
        ((s1a.list.set_intersection(m1a).list.len() / s1a.list.set_union(m1a).list.len())
         .fill_null(0.0)).cast(pl.Float32).alias("addr_token_jaccard"),
        pl.when(
            ((pl.col("s1_name_dev_ratio").fill_null(0.0) > 0.2) & (pl.col("m__name_dev_ratio").fill_null(0.0) <= 0.2))
            | ((pl.col("s1_name_dev_ratio").fill_null(0.0) <= 0.2) & (pl.col("m__name_dev_ratio").fill_null(0.0) > 0.2))
        ).then(1).otherwise(0).cast(pl.Int8).alias("cross_script_pair"),
        pl.when(
            (pl.col("s1_addr_city").is_not_null()) & (pl.col("m__addr_city").is_not_null())
            & (pl.col("s1_addr_city") == pl.col("m__addr_city"))
        ).then(1).otherwise(0).cast(pl.Int8).alias("addr_city_eq"),
    ])


def compute_name_partial_ratio(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df.with_columns(pl.lit(0.0).cast(pl.Float32).alias("name_partial_ratio"))
    rows = df.select(["s1_name_clean", "m__name_clean"]).to_dicts()
    partial = np.zeros(len(rows), dtype=np.float32)
    for i, r in enumerate(rows):
        a = _s(r.get("s1_name_clean")); b = _s(r.get("m__name_clean"))
        if a and b:
            partial[i] = float(fuzz.partial_ratio(a, b))
    return df.with_columns(pl.Series("name_partial_ratio", partial, dtype=pl.Float32))


def compute_name_ratio(df: pl.DataFrame) -> pl.DataFrame:
    """rapidfuzz.fuzz.ratio on name (full Levenshtein-based). For single-char typos."""
    if df.is_empty():
        return df.with_columns(pl.lit(0.0).cast(pl.Float32).alias("name_ratio"))
    rows = df.select(["s1_name_clean", "m__name_clean"]).to_dicts()
    ratio = np.zeros(len(rows), dtype=np.float32)
    for i, r in enumerate(rows):
        a = _s(r.get("s1_name_clean")); b = _s(r.get("m__name_clean"))
        if a and b:
            ratio[i] = float(fuzz.ratio(a, b))
    return df.with_columns(pl.Series("name_ratio", ratio, dtype=pl.Float32))


# ---------------------------------------------------------------------------
# Quality floors
# ---------------------------------------------------------------------------
def apply_floor(df: pl.DataFrame, floor_name: str) -> pl.DataFrame:
    """Apply named quality floor."""
    if floor_name == "v3":
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
    elif floor_name == "liberal_v3":
        # liberal_v2 + char-trigram / bigram / edit-distance rescue clauses
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
            | (pl.col("n_trigrams_shared").fill_null(0) >= 3)
            | (pl.col("n_bigrams_shared").fill_null(0) >= 1)
            | (pl.col("name_ratio").fill_null(0.0) >= 75)
        )
    else:
        raise ValueError(f"Unknown floor: {floor_name}")


# ---------------------------------------------------------------------------
# Recall computation
# ---------------------------------------------------------------------------
def compute_recall(cand_pairs: pl.DataFrame, gt: dict[str, set[str]],
                   s1_meta: pl.DataFrame) -> dict:
    s1_ids = s1_meta["source1_entity_id"].to_list()
    # s1_meta uses "country" column (NOT pre-renamed; attach_fields renames on join)
    s1_country_map = dict(zip(s1_meta["source1_entity_id"].to_list(),
                               s1_meta["country"].to_list()))
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
        # Normalize country name to 2-letter code (India → IN, US stays)
        raw_c = s1_country_map.get(sid, "US")
        c = "IN" if raw_c == "India" else (raw_c if raw_c in ("US", "IN") else "US")
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
# Methods (16)
# ---------------------------------------------------------------------------
DEFAULT_STRUCT_KEYS = ["city", "state", "road", "house", "name_fw", "addr_fw",
                       "city_state", "house_road"]
NGRAM_KEYS = ["name_ngram_key", "addr_ngram_key"]

METHODS = [
    {"name": "M05_8struct_only",         "struct_keys": DEFAULT_STRUCT_KEYS, "token": False, "latin": False, "sortedn": False, "ngram": False, "floor": "v3", "cap": 50},
    {"name": "M06_8struct+token",        "struct_keys": DEFAULT_STRUCT_KEYS, "token": True,  "latin": False, "sortedn": False, "ngram": False, "floor": "v3", "cap": 50},
    {"name": "M07_8struct+token+sortedn_v3", "struct_keys": DEFAULT_STRUCT_KEYS, "token": True, "latin": False, "sortedn": True, "ngram": False, "floor": "v3", "cap": 50},

    {"name": "M08_M07+ngram_keys",       "struct_keys": DEFAULT_STRUCT_KEYS + NGRAM_KEYS, "token": True, "latin": False, "sortedn": True, "ngram": True, "floor": "v3", "cap": 50},

    {"name": "M09_M07+latin_tokens",     "struct_keys": DEFAULT_STRUCT_KEYS, "token": True, "latin": True, "sortedn": True, "ngram": False, "floor": "v3", "cap": 50},
    {"name": "M10_M08+latin_tokens",     "struct_keys": DEFAULT_STRUCT_KEYS + NGRAM_KEYS, "token": True, "latin": True, "sortedn": True, "ngram": True, "floor": "v3", "cap": 50},

    {"name": "M11_M09+liberal_v1",       "struct_keys": DEFAULT_STRUCT_KEYS, "token": True, "latin": True, "sortedn": True, "ngram": False, "floor": "liberal_v1", "cap": 50},
    {"name": "M12_M10+liberal_v1",       "struct_keys": DEFAULT_STRUCT_KEYS + NGRAM_KEYS, "token": True, "latin": True, "sortedn": True, "ngram": True, "floor": "liberal_v1", "cap": 50},

    {"name": "M13_M11+cap100",           "struct_keys": DEFAULT_STRUCT_KEYS, "token": True, "latin": True, "sortedn": True, "ngram": False, "floor": "liberal_v1", "cap": 100},
    {"name": "M14_M12+cap100",           "struct_keys": DEFAULT_STRUCT_KEYS + NGRAM_KEYS, "token": True, "latin": True, "sortedn": True, "ngram": True, "floor": "liberal_v1", "cap": 100},

    {"name": "M15_M14+floor_B_relaxed",  "struct_keys": DEFAULT_STRUCT_KEYS + NGRAM_KEYS, "token": True, "latin": True, "sortedn": True, "ngram": True, "floor": "liberal_v2", "cap": 100},

    {"name": "M16_v4_production",        "struct_keys": DEFAULT_STRUCT_KEYS + NGRAM_KEYS, "token": True, "latin": True, "sortedn": True, "ngram": True, "floor": "liberal_v2", "cap": 100},

    # M17-M21: aggressive methods targeting typos/concat/cross-script
    # All use liberal_v3 floor (adds char-trigram / bigram / name_ratio clauses)
    # Each method probes a SUBSET of aggressive indices (built lazily)
    # M17 = char-trigram ONLY (MinHash LSH surrogate for short text; Jaccard >=3 in floor)
    # M19 = bigram ONLY
    # M20 = no new probe; just adds name_ratio floor clause (single-char typos)
    # M21 = kitchen sink (char-trigram + bigram + name_ratio floor)
    {"name": "M17_M16+CharTri",   "struct_keys": DEFAULT_STRUCT_KEYS + NGRAM_KEYS, "token": True, "latin": True, "sortedn": True, "ngram": True, "floor": "liberal_v3", "cap": 100, "char_trigram": True, "bigram": False},
    {"name": "M19_M16+Bigrams",   "struct_keys": DEFAULT_STRUCT_KEYS + NGRAM_KEYS, "token": True, "latin": True, "sortedn": True, "ngram": True, "floor": "liberal_v3", "cap": 100, "char_trigram": False, "bigram": True},
    {"name": "M20_M16+EditDist_floor", "struct_keys": DEFAULT_STRUCT_KEYS + NGRAM_KEYS, "token": True, "latin": True, "sortedn": True, "ngram": True, "floor": "liberal_v3", "cap": 100, "char_trigram": False, "bigram": False},  # name_ratio floor clause only
    {"name": "M21_M16+KitchenSink",    "struct_keys": DEFAULT_STRUCT_KEYS + NGRAM_KEYS, "token": True, "latin": True, "sortedn": True, "ngram": True, "floor": "liberal_v3", "cap": 100, "char_trigram": True, "bigram": True},  # both + name_ratio floor
]


# ---------------------------------------------------------------------------
# Per-method runner (uses CACHED indices)
# ---------------------------------------------------------------------------
def run_method(method: dict, s1_meta: pl.DataFrame, cand_df: pl.DataFrame,
               gt: dict[str, set[str]],
               cached_inv: dict, cached_token: dict, cached_sortedn: tuple,
               cached_ct: dict, cached_bigram: dict) -> dict:
    """Run one method end-to-end using cached indices."""
    name = method["name"]
    t0 = time.time()
    use_ct = method.get("char_trigram", False)
    use_bigram = method.get("bigram", False)
    use_latin = method["latin"]
    print(f"\n[{name}] struct_keys={len(method['struct_keys'])} token={method['token']} "
          f"latin={use_latin} sortedn={method['sortedn']} ngram={method['ngram']} "
          f"ct={use_ct} bigram={use_bigram} floor={method['floor']} cap={method['cap']}",
          flush=True)

    # 1. Get cached indices (subset)
    inv_subset = {k: cached_inv[k] for k in method["struct_keys"] if k in cached_inv}
    if method["token"]:
        token_cache_key = "latin" if use_latin else "no_latin"
        token_idx = cached_token[token_cache_key]
    else:
        token_idx = None
    if method["sortedn"]:
        sorted_canon, ids_aligned = cached_sortedn
    else:
        sorted_canon = np.array([], dtype=object)
        ids_aligned = np.array([], dtype=object)
    if use_ct:
        ct_cache_key = "latin" if use_latin else "no_latin"
        ct_idx = cached_ct.get(ct_cache_key)
    else:
        ct_idx = None
    if use_bigram:
        bigram_cache_key = "latin" if use_latin else "no_latin"
        bigram_idx = cached_bigram.get(bigram_cache_key)
    else:
        bigram_idx = None

    # 2. Probe 10K S1
    s1_chunk = s1_meta
    struct_df = (probe_structural_subset(s1_chunk, inv_subset, method["struct_keys"], top_k=200)
                 if inv_subset else None)
    token_df = (probe_tokens_subset(s1_chunk, token_idx, top_k=200, include_latin=use_latin)
                if token_idx is not None else None)
    sn_df = (probe_sorted_neighborhood_subset(s1_chunk, sorted_canon, ids_aligned, window=50)
             if method["sortedn"] else None)
    ct_df = (probe_char_trigram_subset(s1_chunk, ct_idx, top_k=200, include_latin=use_latin)
             if ct_idx is not None else None)
    bigram_df = (probe_bigram_subset(s1_chunk, bigram_idx, top_k=200, include_latin=use_latin)
                 if bigram_idx is not None else None)

    # 3. Union (full outer join)
    pairs = struct_df if struct_df is not None else pl.DataFrame(
        schema={"source1_entity_id": pl.Utf8, "candidate_entity_id": pl.Utf8}
    )
    if token_df is not None:
        pairs = pairs.join(token_df, on=["source1_entity_id", "candidate_entity_id"],
                            how="full", coalesce=True)
    if sn_df is not None:
        pairs = pairs.join(sn_df, on=["source1_entity_id", "candidate_entity_id"],
                            how="full", coalesce=True)
    if ct_df is not None:
        pairs = pairs.join(ct_df, on=["source1_entity_id", "candidate_entity_id"],
                            how="full", coalesce=True)
    if bigram_df is not None:
        pairs = pairs.join(bigram_df, on=["source1_entity_id", "candidate_entity_id"],
                            how="full", coalesce=True)
    # Ensure all expected columns exist (add zero columns for missing probes).
    # Different methods skip different probes; their columns won't be in pairs.
    for col_name, dtype in [
        ("n_struct_keys", pl.Int8),
        ("n_tokens_shared", pl.Int8),
        ("sortedn_rank", pl.Int16),
        ("n_trigrams_shared", pl.Int16),
        ("n_bigrams_shared", pl.Int16),
    ]:
        if col_name not in pairs.columns:
            pairs = pairs.with_columns(pl.lit(0).cast(dtype).alias(col_name))
    pairs = pairs.with_columns([
        pl.col("n_struct_keys").fill_null(0).cast(pl.Int8),
        pl.col("n_tokens_shared").fill_null(0).cast(pl.Int8),
        pl.col("sortedn_rank").fill_null(0).cast(pl.Int16),
        pl.col("n_trigrams_shared").fill_null(0).cast(pl.Int16),
        pl.col("n_bigrams_shared").fill_null(0).cast(pl.Int16),
    ])
    n_after_probe = pairs.height

    # 4. Attach fields
    pairs = attach_fields(pairs, s1_meta, cand_df)

    # 5. Country filter
    pairs = pairs.filter(
        pl.col("s1_country").is_not_null() & pl.col("m__country").is_not_null()
        & (pl.col("s1_country") == pl.col("m__country"))
    )
    n_after_country = pairs.height

    # 6. Compute features for liberal floors
    if method["floor"] in ("liberal_v1", "liberal_v2", "liberal_v3"):
        pairs = compute_basic_features(pairs)
        pairs = compute_name_partial_ratio(pairs)
        if method["floor"] == "liberal_v3":
            pairs = compute_name_ratio(pairs)
    else:
        if "name_token_jaccard" not in pairs.columns:
            pairs = pairs.with_columns(pl.lit(0.0).cast(pl.Float32).alias("name_token_jaccard"))

    # 7. Apply quality floor
    pairs = apply_floor(pairs, method["floor"])
    n_after_floor = pairs.height

    # 8. Top-K cap per S1
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
    print(f"   probe={n_after_probe:,}  country={n_after_country:,}  "
          f"floor={n_after_floor:,}  cap={n_cands:,}  "
          f"recall={metrics['mean']:.4f}  perfect={metrics['pct_perfect']*100:.1f}%  "
          f"cands/S1={metrics['mean_cands']:.1f}  ({elapsed:.1f}s)", flush=True)

    # Cleanup per-method intermediates
    del struct_df, token_df, sn_df, ct_df, bigram_df, pairs, inv_subset
    if token_idx is not None:
        del token_idx
    if ct_idx is not None:
        del ct_idx
    if bigram_idx is not None:
        del bigram_idx
    gc.collect()

    return {
        "method": name,
        "struct_keys": str(len(method["struct_keys"])),
        "token": method["token"], "latin": method["latin"],
        "sortedn": method["sortedn"], "ngram": method["ngram"],
        "char_trigram": use_ct, "bigram": use_bigram,
        "floor": method["floor"], "cap": method["cap"],
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
    p.add_argument("--output", type=Path, default=TESTING_DIR / "ablation_full.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--source", choices=["S2", "S3", "all"], default="all",
                   help="Which candidate source to probe. S2-only halves RAM (good for smoke tests).")
    p.add_argument("--no-aggressive", action="store_true",
                   help="Skip M17-M21 (no char-trigram/bigram indices). Smoke-test mode.")
    args = p.parse_args()

    print("=" * 70, flush=True)
    log_resources("ablation-start")
    n_methods = len(METHODS) - (3 if args.no_aggressive else 0)  # M17, M19, M21 dropped
    print(f"Ablation: {n_methods} methods, sample={args.sample_size:,}, seed={args.seed}",
          flush=True)
    print(f"Source: {args.source} | Bucket cap: {ABLATION_BUCKET_CAP} | "
          f"Aggressive: {'NO' if args.no_aggressive else 'YES'}", flush=True)
    print("=" * 70, flush=True)

    # 1. Load data
    s1, s2, s3 = load_sources()
    gt = load_ground_truth()
    s1_ids_sampled = sample_s1_ids(s1, gt, args.sample_size, args.seed)

    # Build enriched s1_meta with all fields needed for probing.
    # Keep "country" as-is (NOT pre-renamed); attach_fields does the rename.
    s1_full = s1.select([
        pl.col("entity_id").alias("source1_entity_id"),
        "country",
        "name_clean", "name_latin", "name_dev_ratio", "name_missing",
        "addr_clean", "addr_latin", "addr_missing",
        "addr_first_word", "addr_last_word",
        "addr_city", "addr_state",
        "addr_house_number", "addr_road", "addr_unit", "addr_suburb",
        "name_ngram_key", "addr_ngram_key",
    ])
    s1_meta = pl.DataFrame({"source1_entity_id": s1_ids_sampled}).join(
        s1_full, on="source1_entity_id", how="left"
    )

    # Free the full s1 (we have s1_meta now)
    del s1, s1_full
    gc.collect()
    log_resources("after-s1-meta")

    # 2. Build candidate pool — single source for smoke, both for full ablation
    if args.source == "S2":
        print(f"[cand] using S2 only ({S2_NORM.name}) ...", flush=True)
        cand_df = s2
        del s3
    elif args.source == "S3":
        print(f"[cand] using S3 only ({S3_NORM.name}) ...", flush=True)
        cand_df = s3
        del s2
    else:  # "all"
        print(f"[cand] union s2 + s3 ...", flush=True)
        t0 = time.time()
        cand_df = pl.concat([s2, s3], how="vertical")
        print(f"        {cand_df.height:,} candidates in {time.time()-t0:.1f}s", flush=True)
        del s2, s3
    gc.collect()
    log_resources("after-cand")

    # 3. Build CACHED indices ONCE (the RAM-safety optimization)
    print(f"\n[index] building CACHED indices (built once, reused by all 16 methods) ...",
          flush=True)
    t_idx = time.time()
    key_extractors = make_key_extractors()
    cached_inv: dict[str, pl.DataFrame] = {}
    for k_name, expr in key_extractors.items():
        t0 = time.time()
        cached_inv[k_name] = build_one_struct_index(cand_df, expr, cap=ABLATION_BUCKET_CAP)
        print(f"   [struct {k_name:>14s}] {cached_inv[k_name].height:>9,} pairs in {time.time()-t0:.1f}s",
              flush=True)
    print(f"   [struct total] {sum(df.height for df in cached_inv.values()):,} pairs in {time.time()-t_idx:.1f}s",
          flush=True)

    # Token indices (2 versions: with/without Latin)
    t0 = time.time()
    cached_token: dict[str, pl.DataFrame] = {
        "no_latin": build_token_index(cand_df, cap=ABLATION_BUCKET_CAP, include_latin=False),
    }
    print(f"   [token no_latin] {cached_token['no_latin'].height:,} pairs in {time.time()-t0:.1f}s",
          flush=True)
    t0 = time.time()
    cached_token["latin"] = build_token_index(cand_df, cap=ABLATION_BUCKET_CAP, include_latin=True)
    print(f"   [token latin   ] {cached_token['latin'].height:,} pairs in {time.time()-t0:.1f}s",
          flush=True)

    # Sorted-neighborhood (1 build)
    t0 = time.time()
    cached_sortedn = build_sorted_neighborhood(cand_df)
    print(f"   [sorted-neigh  ] {len(cached_sortedn[0]):,} canonicals in {time.time()-t0:.1f}s",
          flush=True)

    print(f"\n[index] CACHED INDEX BUILD TOTAL: {time.time()-t_idx:.1f}s", flush=True)
    log_resources("after-cached-index")

    # 3b. Build aggressive indices LAZILY (only if any method needs them)
    # Char-trigram + bigram are built with smaller caps to stay within RAM budget.
    # These are built BEFORE the method loop because all methods M17-M21 use them.
    # Skip entirely if --no-aggressive (smoke test mode).
    cached_ct: dict[str, pl.DataFrame] = {}
    cached_bigram: dict[str, pl.DataFrame] = {}
    methods_to_run = METHODS
    if args.no_aggressive:
        methods_to_run = [m for m in METHODS if not (m.get("char_trigram") or m.get("bigram"))]
        print(f"\n[--no-aggressive] Skipping M17/M19/M21 (char-trigram + bigram). "
              f"Running {len(methods_to_run)} methods (M05-M16 + M20).", flush=True)
    any_method_needs_ct = any(m.get("char_trigram", False) for m in methods_to_run)
    any_method_needs_bigram = any(m.get("bigram", False) for m in methods_to_run)
    if any_method_needs_ct:
        print(f"\n[index] building char-trigram index (cap=50, latin) ...", flush=True)
        t0 = time.time()
        cached_ct["latin"] = build_char_trigram_index(cand_df, cap=50, include_latin=True)
        print(f"   [char-trigram ] {cached_ct['latin'].height:,} pairs in {time.time()-t0:.1f}s",
              flush=True)
        log_resources("after-ct")
    if any_method_needs_bigram:
        print(f"\n[index] building bigram index (cap=100, latin) ...", flush=True)
        t0 = time.time()
        cached_bigram["latin"] = build_bigram_index(cand_df, cap=100, include_latin=True)
        print(f"   [bigram       ] {cached_bigram['latin'].height:,} pairs in {time.time()-t0:.1f}s",
              flush=True)
        log_resources("after-bigram")

    # 4. Run all methods using cached indices
    print("\n" + "=" * 70, flush=True)
    print(f"Running {len(METHODS)} methods on {s1_meta.height:,} S1 ...", flush=True)
    print("=" * 70, flush=True)

    results: list[dict] = []
    for i, method in enumerate(methods_to_run, 1):
        try:
            row = run_method(method, s1_meta, cand_df, gt,
                             cached_inv, cached_token, cached_sortedn,
                             cached_ct, cached_bigram)
            results.append(row)
        except Exception as e:
            print(f"   ERROR: {type(e).__name__}: {e}", flush=True)
            import traceback; traceback.print_exc()
            results.append({
                "method": method["name"], "error": str(e),
                "mean": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0,
                "max": 0.0, "pct_perfect": 0.0, "n_nonsingleton": 0,
                "us_mean": 0.0, "in_mean": 0.0, "mean_cands": 0.0,
                "n_cands": 0, "elapsed_s": 0.0,
            })
        gc.collect()
        # Print progress
        if i % 4 == 0:
            log_resources(f"after-method-{i}")

    # Free cached indices
    del cached_inv, cached_token, cached_sortedn, cand_df
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
