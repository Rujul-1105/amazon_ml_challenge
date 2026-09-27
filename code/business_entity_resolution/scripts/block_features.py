"""Phase C v3 + D: hybrid blocking (3 indexes, NO trigrams) + 40-feature extraction.

LOCKED-IN v3 design (see docs/STATUS.md and docs/RUNBOOK.md).

Architecture
------------
  Index 1 — STRUCTURAL INVERTED (8 keys): dict[key_value -> list[id]], cap 500/bucket.
      Keys: city, state, road, house, name_fw, addr_fw, city_state, house_road.

  Index 2 — WORD-TOKEN INVERTED (replaces v2 char-trigrams):
      tokens = tokenize(name_clean + " " + addr_clean).
      tokenize(text) -> set of lowercase alphanumerics, drop len<3 / pure-digit / STOP_TOKENS.
      dict[token -> list[(id, country)]], cap 500/bucket.

  Index 3 — SORTED-TOKEN NEIGHBORHOOD (replaces v2 trigrams + rescues typos/reorders):
      canonical = " ".join(sorted(tokenize(text))).
      np.argsort; query via np.searchsorted with window ±50 (101 candidates per S1).

Per-chunk (10K S1):
  1. Probe all three indexes -> union with metadata columns:
        n_struct_keys  (Int8)
        n_tokens_shared (Int8)
        sortedn_rank    (Int16, 0 if not from SN)
        from_struct     (bool)
        from_token      (bool)
        from_sortedn    (bool)
  2. HARD country pre-filter (mandatory): drop cand if cand.country != s1.country.
  3. Quality-tier OR filter (one of 4 floors):
        A: n_struct_keys >= 2
        B: n_tokens_shared >= 2
        C: s1_road == m_road AND s1_city == m_city (both non-empty)
        D: 0 < sortedn_rank <= 10 AND n_tokens_shared >= 1
  4. Composite score (for top-50 tiebreak):
        struct_score = min(n_struct_keys, 3) / 3.0
        sortedn_prox = 1.0 / (1 + sortedn_rank) if rank>0 else 0.0
        composite = 0.20*struct_score + 0.45*min(n_tokens_shared, 5)/5.0
                  + 0.20*(1.0 if sortedn_rank > 0 else 0.0) + 0.15*sortedn_prox
  5. Top-50 cap per S1 by composite desc.
  6. Attach S1 + M fields. Compute 27 features (polars + rapidfuzz).
  7. Write per-chunk parquet; final concat at end.

Output schema (40 cols; see STATUS.md):
  source1_entity_id, candidate_entity_id, candidate_source,
  n_struct_keys, n_tokens_shared, sortedn_rank,
  from_struct, from_token, from_sortedn,
  block_score,
  s1_country, m__country,
  country_eq, name_first_token_eq, name_token_jaccard, name_n_chars_diff, cross_script_pair,
  addr_first_word_eq, addr_last_word_eq, addr_city_eq, addr_house_number_eq,
  addr_state_eq, addr_road_eq, addr_zip_eq, addr_unit_eq, addr_suburb_eq,
  s1_name_missing, m_name_missing, s1_addr_missing, m_addr_missing,
  name_token_set_ratio, name_partial_ratio, name_token_sort_ratio,
  name_ratio, name_latin_token_set_ratio,
  addr_token_set_ratio, addr_partial_ratio, addr_token_sort_ratio,
  addr_ratio, addr_latin_token_set_ratio.

CLI:
  python scripts/block_features.py --candidate-source S2 [--dry-run] [--max-chunks N]
                                   [--top-k 50] [--chunk-size 10000]
                                   [--bucket-cap 500] [--sn-window 50]
                                   [--token-cap 500] [--suffix S2]
"""
from __future__ import annotations

import argparse
import gc
import io
import multiprocessing as mp
import os
import re
import sys
import time
from pathlib import Path

if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from datasketch import MinHash, MinHashLSH

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
CHUNK_DIR = ARTIFACTS / "_chunks"
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

S1_NORM = ARTIFACTS / "s1_norm_train.parquet"
S2_NORM = ARTIFACTS / "s2_norm_train.parquet"
S3_NORM = ARTIFACTS / "s3_norm_train.parquet"

# RAM/worker auto-detection (after PROJECT_ROOT is defined so we can add it to sys.path)
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from _resources import detect_resources, configure_polars_threads  # noqa: E402

_RES = detect_resources(min_free_gb=8.0)
configure_polars_threads(_RES["polars_threads"])

# 18 fields loaded (entity_id + 17 features). name_tokens is unused for blocking
# but cheap to keep alongside for downstream training alignment.
FEATURE_FIELDS = [
    "entity_id", "country",
    "name_clean", "name_latin", "name_tokens",
    "name_dev_ratio", "name_missing",
    "addr_clean", "addr_latin",
    "addr_zip", "addr_state", "addr_city",
    "addr_first_word", "addr_last_word",
    "addr_house_number", "addr_road", "addr_unit", "addr_suburb",
    "addr_missing",
]

# ---------------------------------------------------------------------------
# Stop tokens (curated; LITERAL from STATUS.md)
# ---------------------------------------------------------------------------
STOP_TOKENS = {
    # English stop-words
    "the", "a", "an", "and", "or", "of", "in", "at", "on", "to", "for", "with", "by", "from",
    "as", "is", "are", "was", "were", "be", "been", "it", "its", "this", "that", "these", "those",
    # Business legal forms
    "inc", "incorporated", "ltd", "limited", "llc", "llp", "corp", "corporation",
    "company", "co", "companies", "pvt", "private", "plc", "gmbh", "sa", "srl",
    "group", "holdings", "partners", "associates", "enterprises",
    "international", "global", "world",
    # Common nouns (multi-lingual)
    "no", "number", "de", "la", "el", "los", "las", "san", "santa",
    "new", "old", "north", "south", "east", "west", "central",
    "city", "state", "india", "usa", "us", "uk",
}

# ---------------------------------------------------------------------------
# 8 cheap structural blocking keys (matches project's KEY_EXTRACTORS for v2)
# Always wrap with fill_null("") first to avoid null dtype propagation.
# ---------------------------------------------------------------------------
KEY_EXTRACTORS: dict[str, pl.Expr] = {
    "city":       pl.col("addr_city").fill_null("").str.strip_chars().str.to_lowercase(),
    "state":      pl.col("addr_state").fill_null("").str.strip_chars().str.to_lowercase(),
    "road":       pl.col("addr_road").fill_null("").str.strip_chars().str.to_lowercase(),
    "house":      pl.col("addr_house_number").fill_null("").str.strip_chars().str.to_lowercase(),
    "name_fw":    pl.col("name_clean").fill_null("").str.split(" ").list.first()
                     .fill_null("").str.strip_chars().str.to_lowercase(),
    "addr_fw":    pl.col("addr_first_word").fill_null("").str.strip_chars().str.to_lowercase(),
    "city_state": pl.concat_str(
                       [pl.col("addr_city").fill_null(""),
                        pl.lit("|"),
                        pl.col("addr_state").fill_null("")],
                       separator="", ignore_nulls=True)
                   .str.strip_chars().str.to_lowercase(),
    "house_road": pl.concat_str(
                       [pl.col("addr_house_number").fill_null(""),
                        pl.lit("|"),
                        pl.col("addr_road").fill_null("")],
                       separator="", ignore_nulls=True)
                   .str.strip_chars().str.to_lowercase(),
}

# ---------------------------------------------------------------------------
# Token + canonical helpers
# ---------------------------------------------------------------------------
def _tokenize(text: str) -> list[str]:
    """Alphanumeric tokens, len>=3, not in STOP_TOKENS, not pure-digit."""
    if not text:
        return []
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in toks if len(t) >= 3 and t not in STOP_TOKENS and not t.isdigit()]


def _canonical_tokens(text: str) -> str:
    """Sorted set of filtered tokens joined by space — used by sorted-neighborhood."""
    return " ".join(sorted(set(_tokenize(text))))


def _s(x) -> str:
    """Coerce nullable/NaN string to '' (used in rapidfuzz loop)."""
    if x is None:
        return ""
    if isinstance(x, float) and x != x:
        return ""
    return str(x)


# ---------------------------------------------------------------------------
# Source loading
# ---------------------------------------------------------------------------
def load_sources(candidate_source: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    print(f"[load] s1_norm_train ...", flush=True)
    t0 = time.time()
    s1 = pl.read_parquet(S1_NORM, columns=FEATURE_FIELDS)
    print(f"        {s1.height:,} rows, {time.time() - t0:.1f}s", flush=True)

    cand_path = S2_NORM if candidate_source == "S2" else S3_NORM
    print(f"[load] {cand_path.name} ...", flush=True)
    t0 = time.time()
    cand = pl.read_parquet(cand_path, columns=FEATURE_FIELDS)
    print(f"        {cand.height:,} rows, {time.time() - t0:.1f}s", flush=True)
    return s1, cand


# ---------------------------------------------------------------------------
# Index builders
# ---------------------------------------------------------------------------
def build_structural_indexes(cand: pl.DataFrame, cap: int) -> dict[str, pl.DataFrame]:
    """Index 1: 8 structural inverted indexes as polars DataFrames.

    Each index is a long-format DF with columns (_k: Utf8, _id: Utf8),
    capped at `cap` ids per (key_type, key_value) bucket. Polars-side
    representation enables sub-second per-chunk probing via SIMD join.
    """
    print(f"[index-1] building 8 structural inverted indexes (cap={cap}) ...", flush=True)
    inv: dict[str, pl.DataFrame] = {}
    for name, expr in KEY_EXTRACTORS.items():
        t0 = time.time()
        df = (cand.with_columns(expr.alias("_k"))
                  .filter(pl.col("_k").is_not_null() & (pl.col("_k") != ""))
                  .select(["_k", pl.col("entity_id").alias("_id")])
                  .unique(subset=["_k", "_id"])
                  .group_by("_k")
                  .agg(pl.col("_id"))
                  .with_columns(pl.col("_id").list.slice(0, cap))
                  .explode("_id"))
        inv[name] = df
        n_keys = df["_k"].n_unique()
        n_pairs = df.height
        print(f"   [{name:>11s}] {n_keys:>7,} keys, {n_pairs:>9,} (key,id) in {time.time() - t0:.1f}s",
              flush=True)
    return inv


def build_token_index(cand: pl.DataFrame, cap: int) -> pl.DataFrame:
    """Index 2: word-token inverted index as a polars DataFrame.

    Returns a long-format DF with columns (_tok: Utf8, _id: Utf8), capped at
    `cap` ids per token bucket. Country is no longer stored here — it's looked
    up via `attach_fields` (polars join) which is vectorized.
    """
    print(f"[index-2] building word-token inverted index (cap={cap}) ...", flush=True)
    t0 = time.time()
    # Pass 1: collect records (need to know len(cand) per token to apply cap).
    # Group by candidate to dedupe, then trim to cap.
    cand_tok = cand.with_columns([
        (pl.col("name_clean").fill_null("") + pl.lit(" ") + pl.col("addr_clean").fill_null("")).alias("_text"),
    ]).with_columns(
        pl.col("_text").map_batches(
            lambda s: pl.Series([list(set(_tokenize(t)) if t else []) for t in s]),
            return_dtype=pl.List(pl.Utf8),
        ).alias("_toks"),
    ).select(["entity_id", "_toks"]).explode("_toks").rename({"_toks": "_tok"})
    cand_tok = cand_tok.filter(pl.col("_tok").is_not_null())
    # Apply per-token cap (keep first `cap` ids per token; ids are dedupe-stable)
    df = (cand_tok
        .unique(subset=["_tok", "entity_id"])
        .group_by("_tok")
        .agg(pl.col("entity_id"))
        .with_columns(pl.col("entity_id").list.slice(0, cap))
        .explode("entity_id")
        .rename({"entity_id": "_id"}))
    n_tokens = df["_tok"].n_unique()
    n_total = df.height
    print(f"        {n_tokens:,} unique tokens, {n_total:,} (token,id) pairs in {time.time() - t0:.1f}s",
          flush=True)
    return df


def build_sorted_neighborhood(cand: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Index 3: sorted-token neighborhood.

    Returns (sorted_canon, ids_aligned) — numpy object arrays of equal length N
    (one per unique canonical). Sort order in `sorted_canon` is the lexicographic
    order of canonical_strings; ids_aligned[k] is the cand entity_id that maps
    to sorted_canon[k]. Country is NOT stored here because we look it up via
    in `attach_fields` (avoids 5M wasted numpy entries).
    """
    print(f"[index-3] building sorted-token neighborhood ...", flush=True)
    t0 = time.time()
    canon_to_id: dict[str, str] = {}
    for r in cand.iter_rows(named=True):
        text = (r.get("name_clean") or "") + " " + (r.get("addr_clean") or "")
        canon = _canonical_tokens(text)
        # canon may be empty; use a single space placeholder so it sorts but is
        # effectively unmatchable. setdefault avoids overwriting if multiple
        # candidates share the same canonical.
        canon_to_id.setdefault(canon or " ", r["entity_id"])

    sorted_canon = np.array(sorted(canon_to_id.keys()), dtype=object)
    n = len(sorted_canon)
    ids_aligned = np.empty(n, dtype=object)
    for i, c in enumerate(sorted_canon):
        ids_aligned[i] = canon_to_id[c]
    print(f"        {n:,} unique canonicals in {time.time() - t0:.1f}s", flush=True)
    return sorted_canon, ids_aligned


# ---------------------------------------------------------------------------
# Per-S1 probes
# ---------------------------------------------------------------------------
def probe_structural(chunk: pl.DataFrame, inv: dict, top_k: int) -> pl.DataFrame:
    """Vectorized structural probe via 8 polars inner-joins (sub-second on 10K rows).

    For each (key_type, key_value), joins chunk.rows against the inverted index
    and unions all 8 results. Counts distinct `key_type`s per (s1, cand) pair
    to produce `n_struct_keys`, then caps per-S1 at `top_k`.
    """
    parts: list[pl.DataFrame] = []
    for name, expr in KEY_EXTRACTORS.items():
        s1k = (chunk
            .with_columns(expr.alias("_k"))
            .filter(pl.col("_k").is_not_null() & (pl.col("_k") != ""))
            .select(["source1_entity_id", "_k"]))
        if s1k.is_empty():
            continue
        joined = s1k.join(inv[name], on="_k", how="inner")
        parts.append(joined.with_columns(pl.lit(name).alias("_kt")))

    if not parts:
        return pl.DataFrame(schema={
            "source1_entity_id": pl.Utf8,
            "candidate_entity_id": pl.Utf8,
            "n_struct_keys": pl.Int8,
        })
    all_pairs = pl.concat(parts).select(["source1_entity_id", "_id", "_kt"]).unique()
    if all_pairs.is_empty():
        return pl.DataFrame(schema={
            "source1_entity_id": pl.Utf8,
            "candidate_entity_id": pl.Utf8,
            "n_struct_keys": pl.Int8,
        })
    n_struct = (all_pairs
        .group_by(["source1_entity_id", "_id"])
        .agg(pl.col("_kt").n_unique().cast(pl.Int8).alias("n_struct_keys"))
        .with_columns(pl.col("n_struct_keys").rank(method="ordinal", descending=True)
                          .over("source1_entity_id").alias("_r"))
        .filter(pl.col("_r") <= top_k)
        .drop("_r")
        .rename({"_id": "candidate_entity_id"}))
    return n_struct


def probe_tokens(chunk: pl.DataFrame, token_index_df: pl.DataFrame,
                 top_k: int) -> pl.DataFrame:
    """Vectorized token probe via polars join (sub-second on 10K rows).

    1. Tokenize S1 rows (`map_batches` for one-shot Python pass).
    2. Explode to long format (s1_id, token).
    3. Inner-join against token index DF on `_tok`.
    4. Count distinct tokens per (s1, cand) pair as `n_tokens_shared`.
    5. Cap per-S1 at `top_k`.
    """
    s1_tok = (chunk
        .with_columns(
            (pl.col("name_clean").fill_null("") + pl.lit(" ")
             + pl.col("addr_clean").fill_null("")).alias("_text"))
        .select(["source1_entity_id", "_text"])
        .with_columns(pl.col("_text").map_batches(
                lambda s: pl.Series([list(set(_tokenize(t)) if t else []) for t in s]),
                return_dtype=pl.List(pl.Utf8),
            ).alias("_toks"))
        .select(["source1_entity_id", "_toks"])
        .explode("_toks"))
    s1_tok = s1_tok.filter(pl.col("_toks").is_not_null()).rename({"_toks": "_tok"})
    if s1_tok.is_empty():
        return pl.DataFrame(schema={
            "source1_entity_id": pl.Utf8,
            "candidate_entity_id": pl.Utf8,
            "n_tokens_shared": pl.Int8,
        })
    joined = s1_tok.join(token_index_df, on="_tok", how="inner")
    if joined.is_empty():
        return pl.DataFrame(schema={
            "source1_entity_id": pl.Utf8,
            "candidate_entity_id": pl.Utf8,
            "n_tokens_shared": pl.Int8,
        })
    n_tokens = (joined
        .group_by(["source1_entity_id", "_id"])
        .agg(pl.col("_tok").n_unique().cast(pl.Int8).alias("n_tokens_shared"))
        .with_columns(pl.col("n_tokens_shared").rank(method="ordinal", descending=True)
                          .over("source1_entity_id").alias("_r"))
        .filter(pl.col("_r") <= top_k)
        .drop("_r")
        .rename({"_id": "candidate_entity_id"}))
    return n_tokens


def probe_sorted_neighborhood(chunk: pl.DataFrame, sorted_canon: np.ndarray,
                              ids_aligned: np.ndarray, window: int) -> pl.DataFrame:
    """Probe sorted-token neighborhood per S1; window ±W.

    Returns pair list with `sortedn_rank` = |k - idx| distance (1..window) where
    1 = adjacent in sorted order, higher = lower lexical similarity. The
    exact-match position (k == idx) is skipped so we don't pair an S1 with a
    candidate whose canonical is identical (effectively a self-match guard;
    in practice S1 and S2/S3 are disjoint so the skip is rarely triggered).
    Floor D in `process_chunk` reads `sortedn_rank > 0 AND sortedn_rank <= 10`.
    """
    s1_rows = chunk.select(["source1_entity_id", "name_clean", "addr_clean"]).to_dicts()
    pairs: list[tuple[str, str, int]] = []
    for row in s1_rows:
        s1_id = row["source1_entity_id"]
        text = (row.get("name_clean") or "") + " " + (row.get("addr_clean") or "")
        canon = _canonical_tokens(text)
        if not canon or canon == " ":
            continue
        idx = int(np.searchsorted(sorted_canon, canon))
        lo = max(0, idx - window)
        hi = min(len(sorted_canon), idx + window + 1)
        # Walk the ±window; skip the exact-match position; record |k-idx|
        for k in range(lo, hi):
            if k == idx:
                continue
            pairs.append((s1_id, ids_aligned[k], abs(k - idx)))

    if not pairs:
        return pl.DataFrame(schema={
            "source1_entity_id": pl.Utf8,
            "candidate_entity_id": pl.Utf8,
            "sortedn_rank": pl.Int16,
        })
    return pl.DataFrame(
        pairs,
        schema=["source1_entity_id", "candidate_entity_id", "sortedn_rank"],
        orient="row",
    )


# ---------------------------------------------------------------------------
# Field attachment + feature computation
# ---------------------------------------------------------------------------
# Source-side key lists (for rename → prefix). country handled specially.
S1_RENAME = {
    "country":          "s1_country",
    "name_clean":       "s1_name_clean",
    "name_latin":       "s1_name_latin",
    "name_dev_ratio":   "s1_name_dev_ratio",
    "name_missing":     "s1_name_missing",
    "addr_clean":       "s1_addr_clean",
    "addr_latin":       "s1_addr_latin",
    "addr_missing":     "s1_addr_missing",
    "addr_zip":         "s1_addr_zip",
    "addr_state":       "s1_addr_state",
    "addr_city":        "s1_addr_city",
    "addr_first_word":  "s1_addr_first_word",
    "addr_last_word":   "s1_addr_last_word",
    "addr_house_number":"s1_addr_house_number",
    "addr_road":        "s1_addr_road",
    "addr_unit":        "s1_addr_unit",
    "addr_suburb":      "s1_addr_suburb",
}
M_RENAME = {
    "entity_id":        "_m_join_id",   # placeholder; renamed AFTER join
    "country":          "m__country",
    "name_clean":       "m__name_clean",
    "name_latin":       "m__name_latin",
    "name_dev_ratio":   "m__name_dev_ratio",
    "name_missing":     "m__name_missing",
    "addr_clean":       "m__addr_clean",
    "addr_latin":       "m__addr_latin",
    "addr_missing":     "m__addr_missing",
    "addr_zip":         "m__addr_zip",
    "addr_state":       "m__addr_state",
    "addr_city":        "m__addr_city",
    "addr_first_word":  "m__addr_first_word",
    "addr_last_word":   "m__addr_last_word",
    "addr_house_number":"m__addr_house_number",
    "addr_road":        "m__addr_road",
    "addr_unit":        "m__addr_unit",
    "addr_suburb":      "m__addr_suburb",
}


def attach_fields(pairs: pl.DataFrame, s1_chunk: pl.DataFrame,
                  cand_df: pl.DataFrame) -> pl.DataFrame:
    """Attach S1 + candidate fields via polars joins (vectorized, no Python dict).

    - s1 fields prefixed `s1_`     (e.g., s1_country, s1_name_clean)
    - candidate fields prefixed `m__` (e.g., m__country, m__name_clean)
    - Both come from pre-built polars DataFrames, joined on the pair key.
    - Replaces previous Python-dict loop (3-5 s/chunk → ~0.3 s/chunk).
    """
    # S1 fields: pull from chunk and rename with `s1_` prefix.
    s1_view = s1_chunk.select(["source1_entity_id"] + list(S1_RENAME.keys())).rename(S1_RENAME)
    pairs = pairs.join(s1_view, on="source1_entity_id", how="left")

    # M fields: pull from cand_df and rename with `m__` prefix. entity_id is
    # the join key; we keep it as-is (the rename dict maps it to a placeholder
    # so polars' `on=` works without name collision).
    cand_cols = [k for k in M_RENAME.keys() if k != "entity_id"]
    cand_view = cand_df.select(["entity_id"] + cand_cols).rename(M_RENAME)
    # Note: Polars drops the right key (`_m_join_id`) from the result of an
    # asymmetric left_on/right_on join by default, so no explicit .drop() is
    # needed. The placeholder name only exists to avoid a column-name collision
    # between `pairs.candidate_entity_id` and `cand_view.entity_id` pre-join.
    pairs = pairs.join(cand_view, left_on="candidate_entity_id",
                       right_on="_m_join_id", how="left")
    return pairs


def compute_polars_features(df: pl.DataFrame) -> pl.DataFrame:
    """Cheap boolean / structural features via polars vectorised expressions."""
    return df.with_columns([
        # country_eq (mirror the hard-filter check; kept for downstream)
        (pl.col("s1_country") == pl.col("m__country"))
            .cast(pl.Int8).fill_null(0).alias("country_eq"),

        # name_first_token_eq (fill nulls first; str.split requires String dtype)
        (pl.col("s1_name_clean").fill_null("").str.split(" ").list.first()
         == pl.col("m__name_clean").fill_null("").str.split(" ").list.first())
        .cast(pl.Int8).fill_null(0).alias("name_first_token_eq"),

        # name_token_jaccard: |A ∩ B| / |A ∪ B|
        (
            pl.col("s1_name_clean").fill_null("").str.split(" ")
              .list.set_intersection(pl.col("m__name_clean").fill_null("").str.split(" "))
              .list.len().cast(pl.Float32)
            / pl.col("s1_name_clean").fill_null("").str.split(" ")
              .list.set_union(pl.col("m__name_clean").fill_null("").str.split(" "))
              .list.len().cast(pl.Float32).fill_null(1.0)
        ).alias("name_token_jaccard"),

        # addr_token_jaccard: same formula on address fields (used by liberal_v3 floor)
        (
            pl.col("s1_addr_clean").fill_null("").str.split(" ")
              .list.set_intersection(pl.col("m__addr_clean").fill_null("").str.split(" "))
              .list.len().cast(pl.Float32)
            / pl.col("s1_addr_clean").fill_null("").str.split(" ")
              .list.set_union(pl.col("m__addr_clean").fill_null("").str.split(" "))
              .list.len().cast(pl.Float32).fill_null(1.0)
        ).alias("addr_token_jaccard"),

        # name_n_chars_diff (cast to Int32 BEFORE abs to avoid u32 overflow)
        (pl.col("s1_name_clean").fill_null("").str.len_chars().cast(pl.Int32)
         - pl.col("m__name_clean").fill_null("").str.len_chars().cast(pl.Int32)).abs()
        .alias("name_n_chars_diff"),

        # cross_script_pair: dev_ratio > 0.2 on one side, ≤ 0.2 on the other
        ((pl.col("s1_name_dev_ratio").fill_null(0.0) > 0.2)
         != (pl.col("m__name_dev_ratio").fill_null(0.0) > 0.2))
        .cast(pl.Int8).alias("cross_script_pair"),

        # 8 structured address eq
        (pl.col("s1_addr_first_word").fill_null("__null__")
         == pl.col("m__addr_first_word").fill_null("__null__"))
        .cast(pl.Int8).alias("addr_first_word_eq"),
        (pl.col("s1_addr_last_word").fill_null("__null__")
         == pl.col("m__addr_last_word").fill_null("__null__"))
        .cast(pl.Int8).alias("addr_last_word_eq"),
        (pl.col("s1_addr_city").fill_null("__null__")
         == pl.col("m__addr_city").fill_null("__null__"))
        .cast(pl.Int8).alias("addr_city_eq"),
        (pl.col("s1_addr_house_number").fill_null("__null__")
         == pl.col("m__addr_house_number").fill_null("__null__"))
        .cast(pl.Int8).alias("addr_house_number_eq"),
        (pl.col("s1_addr_state").fill_null("__null__")
         == pl.col("m__addr_state").fill_null("__null__"))
        .cast(pl.Int8).alias("addr_state_eq"),
        (pl.col("s1_addr_road").fill_null("__null__")
         == pl.col("m__addr_road").fill_null("__null__"))
        .cast(pl.Int8).alias("addr_road_eq"),
        (pl.col("s1_addr_zip").fill_null("__null__")
         == pl.col("m__addr_zip").fill_null("__null__"))
        .cast(pl.Int8).alias("addr_zip_eq"),
        (pl.col("s1_addr_unit").fill_null("__null__")
         == pl.col("m__addr_unit").fill_null("__null__"))
        .cast(pl.Int8).alias("addr_unit_eq"),
        (pl.col("s1_addr_suburb").fill_null("__null__")
         == pl.col("m__addr_suburb").fill_null("__null__"))
        .cast(pl.Int8).alias("addr_suburb_eq"),

        # 4 missingness flags (source columns are m__name_missing/m__addr_missing;
        # but output alias is single-underscore m_name_missing/m_addr_missing to match
        # the canonical schema in docs/STATUS.md)
        pl.col("s1_name_missing").cast(pl.Int8).fill_null(0).alias("s1_name_missing"),
        pl.col("m__name_missing").cast(pl.Int8).fill_null(0).alias("m_name_missing"),
        pl.col("s1_addr_missing").cast(pl.Int8).fill_null(0).alias("s1_addr_missing"),
        pl.col("m__addr_missing").cast(pl.Int8).fill_null(0).alias("m_addr_missing"),
    ])


def compute_fuzzy_features(df: pl.DataFrame) -> pl.DataFrame:
    """10 rapidfuzz metrics per pair, computed in a Python loop."""
    rows = df.to_dicts()
    n = len(rows)

    name_token_set       = [-1.0] * n
    name_partial         = [-1.0] * n
    name_token_sort      = [-1.0] * n
    name_ratio           = [-1.0] * n
    name_latin_token_set = [-1.0] * n
    addr_token_set       = [-1.0] * n
    addr_partial         = [-1.0] * n
    addr_token_sort      = [-1.0] * n
    addr_ratio           = [-1.0] * n
    addr_latin_token_set = [-1.0] * n

    for i, r in enumerate(rows):
        a_name = _s(r.get("s1_name_clean"))
        b_name = _s(r.get("m__name_clean"))
        a_nl   = _s(r.get("s1_name_latin"))
        b_nl   = _s(r.get("m__name_latin"))
        a_addr = _s(r.get("s1_addr_clean"))
        b_addr = _s(r.get("m__addr_clean"))
        a_al   = _s(r.get("s1_addr_latin"))
        b_al   = _s(r.get("m__addr_latin"))

        if a_name and b_name:
            name_token_set[i]  = float(fuzz.token_set_ratio(a_name, b_name))
            name_partial[i]    = float(fuzz.partial_ratio(a_name, b_name))
            name_token_sort[i] = float(fuzz.token_sort_ratio(a_name, b_name))
            name_ratio[i]      = float(fuzz.ratio(a_name, b_name))
        if a_nl and b_nl:
            name_latin_token_set[i] = float(fuzz.token_set_ratio(a_nl, b_nl))
        if a_addr and b_addr:
            addr_token_set[i]  = float(fuzz.token_set_ratio(a_addr, b_addr))
            addr_partial[i]    = float(fuzz.partial_ratio(a_addr, b_addr))
            addr_token_sort[i] = float(fuzz.token_sort_ratio(a_addr, b_addr))
            addr_ratio[i]      = float(fuzz.ratio(a_addr, b_addr))
        if a_al and b_al:
            addr_latin_token_set[i] = float(fuzz.token_set_ratio(a_al, b_al))

    return df.with_columns([
        pl.Series("name_token_set_ratio",       name_token_set,       dtype=pl.Float32),
        pl.Series("name_partial_ratio",         name_partial,         dtype=pl.Float32),
        pl.Series("name_token_sort_ratio",      name_token_sort,      dtype=pl.Float32),
        pl.Series("name_ratio",                 name_ratio,           dtype=pl.Float32),
        pl.Series("name_latin_token_set_ratio", name_latin_token_set, dtype=pl.Float32),
        pl.Series("addr_token_set_ratio",       addr_token_set,       dtype=pl.Float32),
        pl.Series("addr_partial_ratio",         addr_partial,         dtype=pl.Float32),
        pl.Series("addr_token_sort_ratio",      addr_token_sort,      dtype=pl.Float32),
        pl.Series("addr_ratio",                 addr_ratio,           dtype=pl.Float32),
        pl.Series("addr_latin_token_set_ratio", addr_latin_token_set, dtype=pl.Float32),
    ])


# ---------------------------------------------------------------------------
# MinHash LSH index + probe (datasketch) — for fuzzy Jaccard-based candidate generation.
# Used by Phase C v4 / M26 method (liberal_v3 floor + MinHash rescue).
# ---------------------------------------------------------------------------
def _minhash_worker_task(args: tuple) -> list[tuple[str, MinHash]]:
    """Top-level worker task (must be picklable for spawn multiprocessing).

    Args is (chunk_rows, num_perm). Returns list of (entity_id, MinHash).
    """
    chunk_rows, num_perm = args
    from datasketch import MinHash as _MH  # noqa: F401 (always import in spawn worker)

    sigs: list[tuple[str, MinHash]] = []
    for eid, name_lat, addr_lat in chunk_rows:
        text = (name_lat or "") + " " + (addr_lat or "")
        if len(text) < 3:
            continue
        shingles = {text[i:i+3] for i in range(len(text) - 2)}
        m = _MH(num_perm=num_perm)
        for s in shingles:
            m.update(s.encode("utf-8"))
        sigs.append((eid, m))
    return sigs


def build_minhash_lsh_index(cand: pl.DataFrame, num_perm: int = 64,
                              threshold: float = 0.3,
                              n_workers: int = 1) -> tuple[MinHashLSH, dict[str, MinHash]]:
    """Real MinHash LSH index on (name_latin + addr_latin) char 3-grams.

    Parallelized via multiprocessing.Pool when n_workers > 1.
    For 10M docs on 8 workers, ~4× speedup vs single-threaded.

    Returns (lsh_index, sigs_dict). sigs_dict maps entity_id → MinHash sig.
    """
    print(f"   [minhash  start] num_perm={num_perm}, threshold={threshold}, workers={n_workers} ...",
          flush=True)
    t0 = time.time()
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    # Extract rows as plain Python tuples (fast pickle via mp.Pool)
    print(f"      extracting {cand.height:,} rows...", flush=True)
    t_extract = time.time()
    rows = list(cand.select(["entity_id", "name_latin", "addr_latin"]).iter_rows(named=False))
    print(f"      ... extracted in {time.time()-t_extract:.1f}s", flush=True)

    chunk_size = max(50_000, len(rows) // (n_workers * 2))
    chunks = [rows[i:i+chunk_size] for i in range(0, len(rows), chunk_size)]
    print(f"      {len(chunks)} chunks of ~{chunk_size:,} rows", flush=True)

    sigs: dict[str, MinHash] = {}
    if n_workers <= 1 or len(chunks) <= 1:
        for chunk in chunks:
            for eid, m in _minhash_worker_task((chunk, num_perm)):
                sigs[eid] = m
                lsh.insert(eid, m, check_duplication=False)
    else:
        ctx = mp.get_context("spawn")
        work_items = [(chunk, num_perm) for chunk in chunks]
        with ctx.Pool(n_workers) as pool:
            completed = 0
            for batch in pool.imap_unordered(_minhash_worker_task, work_items, chunksize=1):
                for eid, m in batch:
                    sigs[eid] = m
                    lsh.insert(eid, m, check_duplication=False)
                completed += 1
                if completed % max(1, len(chunks) // 5) == 0:
                    pct = 100 * completed // len(chunks)
                    print(f"      ... {pct}% chunks done ({time.time()-t0:.1f}s)",
                          flush=True)

    print(f"   [minhash  done] {len(sigs):,} sigs in {time.time()-t0:.1f}s", flush=True)
    return lsh, sigs


def probe_minhash_lsh_subset(s1_chunk: pl.DataFrame, lsh: MinHashLSH,
                              sigs: dict[str, MinHash],
                              num_perm: int = 64, threshold: float = 0.3) -> pl.DataFrame:
    """Query LSH with each S1's MinHash sig. Returns pairs with minhash_jaccard column."""
    pairs: list[tuple[str, str, float]] = []
    for row in s1_chunk.select(["source1_entity_id", "name_latin", "addr_latin"]).iter_rows(named=True):
        text = (row.get("name_latin") or "") + " " + (row.get("addr_latin") or "")
        if len(text) < 3:
            continue
        shingles = {text[i:i+3] for i in range(len(text) - 2)}
        m = MinHash(num_perm=num_perm)
        for s in shingles:
            m.update(s.encode("utf-8"))
        candidates = lsh.query(m)
        for cand_id in candidates:
            cand_sig = sigs.get(cand_id)
            if cand_sig is not None:
                jac = m.jaccard(cand_sig)
                if jac >= threshold:
                    pairs.append((row["source1_entity_id"], cand_id, float(jac)))
    if not pairs:
        return pl.DataFrame(schema={
            "source1_entity_id": pl.Utf8, "candidate_entity_id": pl.Utf8,
            "minhash_jaccard": pl.Float32,
        })
    return pl.DataFrame(pairs,
                       schema=["source1_entity_id", "candidate_entity_id", "minhash_jaccard"],
                       orient="row")


# ---------------------------------------------------------------------------
# Per-chunk pipeline
# ---------------------------------------------------------------------------
def process_chunk(idx: int, s1_chunk: pl.DataFrame,
                  inv_struct, token_index_df,
                  sorted_canon, ids_aligned,
                  cand_df: pl.DataFrame, args, candidate_source: str,
                  mh_lsh=None, mh_sigs=None) -> tuple[int, int, int]:
    t_chunk = time.time()

    # --- Phase 1: probe each of 3 indexes (per-S1 cap from --top-k-index) ---
    struct_df = probe_structural(s1_chunk, inv_struct, top_k=args.top_k_index)
    token_df  = probe_tokens(s1_chunk, token_index_df, top_k=args.top_k_index)
    sn_df     = probe_sorted_neighborhood(
                    s1_chunk, sorted_canon, ids_aligned,
                    window=args.sn_window)
    # Phase 1d: MinHash LSH probe (if enabled via --minhash)
    if mh_lsh is not None and mh_sigs is not None:
        mh_df = probe_minhash_lsh_subset(
            s1_chunk, mh_lsh, mh_sigs,
            num_perm=args.minhash_num_perm,
            threshold=args.minhash_threshold,
        )
    else:
        mh_df = None
    n_struct = len(struct_df)
    n_tok = len(token_df)
    n_sn = len(sn_df)
    n_mh = len(mh_df) if mh_df is not None else 0

    # --- Phase 2: union outer joins; fill nulls with 0; boolean flags ---
    merged = struct_df.join(token_df,
                            on=["source1_entity_id", "candidate_entity_id"],
                            how="full", coalesce=True)
    merged = merged.join(sn_df,
                         on=["source1_entity_id", "candidate_entity_id"],
                         how="full", coalesce=True)
    if mh_df is not None:
        merged = merged.join(mh_df,
                             on=["source1_entity_id", "candidate_entity_id"],
                             how="full", coalesce=True)
    # Ensure minhash_jaccard column exists (zero-filled when --minhash not used)
    if "minhash_jaccard" not in merged.columns:
        merged = merged.with_columns(pl.lit(0.0).cast(pl.Float32).alias("minhash_jaccard"))
    merged = merged.with_columns([
        pl.col("n_struct_keys").fill_null(0).cast(pl.Int8),
        pl.col("n_tokens_shared").fill_null(0).cast(pl.Int8),
        pl.col("sortedn_rank").fill_null(0).cast(pl.Int16),
        pl.col("minhash_jaccard").fill_null(0.0).cast(pl.Float32),
    ])
    merged = merged.with_columns([
        (pl.col("n_struct_keys") > 0).cast(pl.Int8).alias("from_struct"),
        (pl.col("n_tokens_shared") > 0).cast(pl.Int8).alias("from_token"),
        (pl.col("sortedn_rank") > 0).cast(pl.Int8).alias("from_sortedn"),
        (pl.col("minhash_jaccard") > 0.0).cast(pl.Int8).alias("from_minhash"),
    ])

    # --- Phase 3: attach S1 + M fields (vectorized polars joins). ---
    merged = attach_fields(merged, s1_chunk, cand_df)

    # --- Phase 4: HARD country filter (mandatory; STATUS.md "Hard-fail bugs #5").
    # MUST reassign `merged` — without this, cross-country pairs pass into
    # Phase E and the LightGBM classifier wastes capacity on easy negatives.
    merged = merged.filter(
        pl.col("s1_country").is_not_null()
        & pl.col("m__country").is_not_null()
        & (pl.col("s1_country") == pl.col("m__country"))
    )
    n_after_country = len(merged)

    # --- Phase 4b: compute n_floors (count of liberal_v3 clauses satisfied) ---
    # Done BEFORE the floor filter so we can use n_floors in the score.
    # n_floors is the count of clauses each candidate passes; top-K by
    # n_floors-weighted score keeps true matches at the front.
    # The 8 base clauses match the OR filter below.
    # (compute addr_road_eq + addr_city_eq on the fly since not yet attached)
    # Note: addr_road_eq and addr_city_eq are computed in compute_polars_features,
    # but we need them here for n_floors. Inline them.
    merged = merged.with_columns([
        # Inline structural eq for road and city (used by Floor C)
        (pl.col("s1_addr_road").fill_null("__n__") == pl.col("m__addr_road").fill_null("__n__"))
            .cast(pl.Int8).alias("_road_eq"),
        (pl.col("s1_addr_city").fill_null("__n__") == pl.col("m__addr_city").fill_null("__n__"))
            .cast(pl.Int8).alias("_city_eq"),
    ])
    # Now compute n_floors as sum of 8 indicator columns
    n_floors_expr = (
        # Floor A: n_struct_keys >= 2
        ((pl.col("n_struct_keys") >= 2).cast(pl.Int8)) +
        # Floor B: n_tokens_shared >= 1
        ((pl.col("n_tokens_shared") >= 1).cast(pl.Int8)) +
        # Floor C: road == road AND city == city (both non-empty)
        (((pl.col("_road_eq") == 1)
          & (pl.col("_city_eq") == 1)).cast(pl.Int8)) +
        # Floor D: sortedn_rank > 0 AND n_tokens_shared >= 1
        (((pl.col("sortedn_rank") > 0) & (pl.col("sortedn_rank") <= 10)
          & (pl.col("n_tokens_shared") >= 1)).cast(pl.Int8)) +
        # Floor E: name_partial_ratio >= 80
        ((pl.col("name_partial_ratio").fill_null(0.0) >= 80).cast(pl.Int8)) +
        # Floor F: name_token_jaccard >= 0.5
        ((pl.col("name_token_jaccard").fill_null(0.0) >= 0.5).cast(pl.Int8)) +
        # Floor G: addr_token_jaccard >= 0.4
        ((pl.col("addr_token_jaccard").fill_null(0.0) >= 0.4).cast(pl.Int8)) +
        # Floor L: minhash_jaccard >= 0.3
        ((pl.col("minhash_jaccard").fill_null(0.0) >= 0.3).cast(pl.Int8))
    ).cast(pl.Int8).alias("n_floors")
    merged = merged.with_columns([n_floors_expr])
    # Drop the temp _road_eq / _city_eq columns (we used them only for n_floors)
    merged = merged.drop(["_road_eq", "_city_eq"])

    # --- Phase 5: quality-tier OR filter (liberal_v3; M26 method + 3 new clauses) ---
    # Floor A: n_struct_keys >= 2
    # Floor B: n_tokens_shared >= 1 (relaxed from 2 for cross-script/typo pairs)
    # Floor C: s1_road == m_road AND s1_city == m_city (both non-empty)
    # Floor D: 0 < sortedn_rank <= 10 AND n_tokens_shared >= 1
    # Floor E: name_partial_ratio >= 80 (rapidfuzz)
    # Floor F: name_token_jaccard >= 0.5
    # Floor G: addr_token_jaccard >= 0.4 (NEW for v3.1)
    # Floor H: cross_script_pair AND tokens AND city_eq
    # Floor K: name_ratio >= 75
    # Floor L: minhash_jaccard >= 0.3 (MinHash LSH rescue; only populated if --minhash)
    # NEW Phase 2 floors:
    # Floor M: name_token_set_ratio >= 90 (rapidfuzz)
    # Floor N: cross_script AND name_token_jaccard >= 0.4
    # Floor O: addr_first_word_eq == 1 AND addr_last_word_eq == 1 AND tokens >= 1
    quality_pass_expr = (
        (pl.col("n_struct_keys") >= 2)
        | (pl.col("n_tokens_shared") >= 1)
        | (
            pl.col("s1_addr_road").is_not_null()
            & pl.col("m__addr_road").is_not_null()
            & (pl.col("s1_addr_road") != "")
            & (pl.col("m__addr_road") != "")
            & (pl.col("s1_addr_road") == pl.col("m__addr_road"))
            & pl.col("s1_addr_city").is_not_null()
            & pl.col("m__addr_city").is_not_null()
            & (pl.col("s1_addr_city") != "")
            & (pl.col("m__addr_city") != "")
            & (pl.col("s1_addr_city") == pl.col("m__addr_city"))
        )
        | (
            (pl.col("sortedn_rank") > 0)
            & (pl.col("sortedn_rank") <= 10)
            & (pl.col("n_tokens_shared") >= 1)
        )
        | (pl.col("name_partial_ratio").fill_null(0.0) >= 80)
        | (pl.col("name_token_jaccard").fill_null(0.0) >= 0.5)
        | (pl.col("addr_token_jaccard").fill_null(0.0) >= 0.4)
        | (
            (pl.col("cross_script_pair").fill_null(0) == 1)
            & (pl.col("n_tokens_shared").fill_null(0) >= 1)
            & (pl.col("addr_city_eq").fill_null(0) == 1)
        )
        | (pl.col("name_ratio").fill_null(0.0) >= 75)
        | (pl.col("minhash_jaccard").fill_null(0.0) >= 0.3)
        # NEW Phase 2 floors (name_token_set_ratio removed — not in df at floor time)
        | (
            (pl.col("cross_script_pair").fill_null(0) == 1)
            & (pl.col("name_token_jaccard").fill_null(0.0) >= 0.4)
        )
        | (
            (pl.col("addr_first_word_eq").fill_null(0) == 1)
            & (pl.col("addr_last_word_eq").fill_null(0) == 1)
            & (pl.col("n_tokens_shared").fill_null(0) >= 1)
        )
    )
    merged = merged.filter(quality_pass_expr)
    n_after_quality = len(merged)

    # --- Phase 6: composite score (n_floors-weighted; for top-K tiebreak) ---
    # n_floors dominates so top-K contains true matches at front.
    # Score breakdown:
    #   0.40 * n_floors / 8     - 40% weight on multi-clause density
    #   0.20 * min(n_tokens_shared, 5)/5 - 20% weight on token overlap
    #   0.15 * min(n_struct_keys, 3)/3 - 15% weight on structural match
    #   0.10 * sortedn_rank proximity
    #   0.10 * sortedn_rank > 0 indicator
    #   0.05 * minhash_jaccard > 0
    merged = merged.with_columns([
        (
            0.40 * pl.col("n_floors").cast(pl.Float32) / 8.0
            + 0.20 * pl.min_horizontal(pl.col("n_tokens_shared").fill_null(0), 5).cast(pl.Float32) / 5.0
            + 0.15 * pl.min_horizontal(pl.col("n_struct_keys").fill_null(0), 3).cast(pl.Float32) / 3.0
            + 0.10 * pl.when(pl.col("sortedn_rank") > 0)
                  .then(1.0 / (1.0 + pl.col("sortedn_rank").cast(pl.Float32)))
                  .otherwise(0.0)
            + 0.10 * (pl.col("sortedn_rank").gt(0).cast(pl.Float32))
            + 0.05 * (pl.col("minhash_jaccard").gt(0).cast(pl.Float32))
        ).cast(pl.Float32).alias("block_score"),
    ])

    # --- Phase 7: top-K cap per S1 (default 50; STATUS.md HARD CONSTRAINT) ---
    merged = (merged
        .with_columns(pl.col("block_score").rank(method="ordinal", descending=True)
                          .over("source1_entity_id").alias("_r"))
        .filter(pl.col("_r") <= args.top_k)
        .drop("_r"))
    n_after_cap = len(merged)

    # --- Phase 8: cheap polars + rapidfuzz features ---
    merged = compute_polars_features(merged)
    merged = compute_fuzzy_features(merged)

    # --- Phase 9: candidate_source literal + canonical column order ---
    # Schema: 42 columns (was 40 in v3; M26 wiring adds 2: addr_token_jaccard + from_minhash)
    merged = merged.with_columns(pl.lit(candidate_source).alias("candidate_source"))
    canonical = [
        "source1_entity_id", "candidate_entity_id", "candidate_source",
        "n_struct_keys", "n_tokens_shared", "sortedn_rank",
        "from_struct", "from_token", "from_sortedn", "from_minhash",
        "n_floors", "block_score",
        "s1_country", "m__country",
        "country_eq", "name_first_token_eq", "name_token_jaccard", "addr_token_jaccard",
        "name_n_chars_diff", "cross_script_pair",
        "addr_first_word_eq", "addr_last_word_eq", "addr_city_eq",
        "addr_house_number_eq", "addr_state_eq", "addr_road_eq",
        "addr_zip_eq", "addr_unit_eq", "addr_suburb_eq",
        "s1_name_missing", "m_name_missing", "s1_addr_missing", "m_addr_missing",
        "name_token_set_ratio", "name_partial_ratio", "name_token_sort_ratio",
        "name_ratio", "name_latin_token_set_ratio",
        "addr_token_set_ratio", "addr_partial_ratio", "addr_token_sort_ratio",
        "addr_ratio", "addr_latin_token_set_ratio",
        "minhash_jaccard",
    ]
    merged = merged.select([c for c in canonical if c in merged.columns])

    # --- Phase 10: write chunk parquet ---
    chunk_path = CHUNK_DIR / f"block_{args.suffix}_{idx:04d}.parquet"
    merged.write_parquet(chunk_path, compression="zstd", compression_level=3)

    elapsed = time.time() - t_chunk
    n_s1 = s1_chunk.height
    print(
        f"  chunk {idx + 1:4d}/{args.n_chunks}  S1_rows={n_s1:,}  "
        f"struct={n_struct:,}  tok={n_tok:,}  sn={n_sn:,}  "
        f"after_country={n_after_country:,}  after_quality={n_after_quality:,}  "
        f"after_cap={n_after_cap:,}  ({elapsed:.1f}s)",
        flush=True,
    )

    del struct_df, token_df, sn_df, merged
    gc.collect()
    return n_after_cap, n_after_quality, n_after_country


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--candidate-source", required=True, choices=["S2", "S3", "S2_TEST", "S3_TEST"])
    p.add_argument("--top-k", type=int, default=400,
                   help="Final per-S1 cap (default 400 for M26 method; was 50 for v3)")
    p.add_argument("--top-k-index", type=int, default=50,
                   help="Per-index candidates per S1 (before union + quality filter)")
    p.add_argument("--chunk-size", type=int, default=10000)
    p.add_argument("--bucket-cap", type=int, default=500,
                   help="Max cand ids per structural-key bucket")
    p.add_argument("--token-cap", type=int, default=500,
                   help="Max cand ids per token bucket")
    p.add_argument("--sn-window", type=int, default=50,
                   help="Sorted-token neighborhood window ±W (default 50 -> 101 candidates)")
    p.add_argument("--max-chunks", type=int, default=None,
                   help="(testing) process only the first N chunks")
    p.add_argument("--dry-run", action="store_true",
                   help="Process a 5% slice and skip final concat")
    p.add_argument("--suffix", default=None,
                   help="Output suffix (default: candidate source)")
    p.add_argument("--start-chunk", type=int, default=-1,
                   help="Skip chunks with idx < N (default: auto-detect from _chunks/)")
    # M26 wiring: MinHash LSH
    p.add_argument("--minhash", action="store_true",
                   help="Enable MinHash LSH index + probe (M26 method)")
    p.add_argument("--minhash-threshold", type=float, default=0.3,
                   help="MinHash Jaccard threshold (default 0.3 for M26)")
    p.add_argument("--minhash-num-perm", type=int, default=64,
                   help="MinHash num_perm (default 64)")
    p.add_argument("--minhash-workers", type=int, default=None,
                   help="MinHash build workers (default: from detect_resources)")
    args = p.parse_args()

    candidate_source = args.candidate_source
    args.suffix = args.suffix or candidate_source

    print("=" * 70, flush=True)
    print(f"Phase C v3 + D: S1 vs {candidate_source}", flush=True)
    print(f"  top_k={args.top_k}, top_k_index={args.top_k_index}, chunk_size={args.chunk_size:,}, "
          f"bucket_cap={args.bucket_cap}, token_cap={args.token_cap}, "
          f"sn_window={args.sn_window}", flush=True)
    if args.dry_run:
        print("  *** DRY RUN: 5% slice, no final concat ***", flush=True)
    print("=" * 70, flush=True)

    # ---- 1. Load sources ----
    s1, cand = load_sources(candidate_source)
    s1 = s1.rename({"entity_id": "source1_entity_id"})

    # ---- 1b. Defensive: clean any stale chunks from a prior partial run ----
    # Without this, a previously interrupted real run with the same --suffix
    # would leak old chunk files into the final concat (wrong row count, possibly
    # mixed-schema). Dry-run is skipped so users can inspect smoke chunks.
    if not args.dry_run:
        stale = sorted(CHUNK_DIR.glob(f"block_{args.suffix}_*.parquet"))
        if stale:
            print(f"[cleanup] removing {len(stale)} stale chunk files for suffix={args.suffix}",
                  flush=True)
            for f in stale:
                try:
                    f.unlink()
                except OSError:
                    pass

    # ---- 2. Build 3 indexes ----
    print("[indexes] building v3 indexes on candidate side ...", flush=True)
    t0 = time.time()
    inv_struct = build_structural_indexes(cand, cap=args.bucket_cap)
    token_index = build_token_index(cand, cap=args.token_cap)
    sorted_canon, ids_aligned = build_sorted_neighborhood(cand)
    print(f"[indexes] all built in {time.time() - t0:.1f}s", flush=True)

    # ---- 2b. Build MinHash LSH index (M26 method, --minhash flag) ----
    # Built BEFORE we del cand because we need name_latin + addr_latin columns.
    mh_lsh = None
    mh_sigs = None
    if args.minhash:
        mh_workers = args.minhash_workers if args.minhash_workers is not None else _RES["n_workers"]
        mh_lsh, mh_sigs = build_minhash_lsh_index(
            cand,
            num_perm=args.minhash_num_perm,
            threshold=args.minhash_threshold,
            n_workers=mh_workers,
        )
        print(f"[indexes] MinHash LSH ready ({len(mh_sigs):,} sigs)", flush=True)

    # ---- 3. Slim cand view (just the fields attach_fields needs) ----
    # Keep `cand` as a polars DF; vectorized joins replace the old Python
    # dict-of-dicts (which used ~3 GB RAM on a 5 M-row candidate). Projecting
    # only the columns `attach_fields` reads via M_RENAME keeps cand_view
    # to ~300 MB.
    cand_view_cols = [k for k in M_RENAME.keys() if k != "entity_id"]
    cand_view = cand.select(["entity_id"] + cand_view_cols)
    del cand  # free original DataFrame (we only need cand_view + indexes)

    # ---- 4. Slice (dry-run only) ----
    if args.dry_run:
        slice_n = max(1, int(0.05 * s1.height))
        s1_slice = s1.head(slice_n)
    else:
        s1_slice = s1

    total = s1_slice.height
    n_chunks = (total + args.chunk_size - 1) // args.chunk_size
    if args.max_chunks:
        n_chunks = min(n_chunks, args.max_chunks)
    args.n_chunks = n_chunks
    print(f"[process] {total:,} s1 entities, {n_chunks} chunks of {args.chunk_size:,}", flush=True)

    # ---- 5a. Resume support: skip chunks already on disk ----
    existing = set()
    for f in CHUNK_DIR.glob(f"block_{args.suffix}_*.parquet"):
        try:
            idx = int(f.stem.rsplit("_", 1)[-1])
            existing.add(idx)
        except ValueError:
            pass

    if args.start_chunk >= 0:
        start_chunk = args.start_chunk
        print(f"[resume] --start-chunk={start_chunk} (explicit override)", flush=True)
    else:
        start_chunk = (max(existing) + 1) if existing else 0
        if existing:
            print(f"[resume] auto-detected {len(existing):,} existing chunks "
                  f"(max idx={max(existing):,}); starting at chunk {start_chunk:,}",
                  flush=True)
        else:
            print(f"[resume] no existing chunks found; starting from chunk 0", flush=True)

    # ---- 5. Stream chunks ----
    total_pairs = 0
    skipped = 0
    for i in range(n_chunks):
        if i < start_chunk or i in existing:
            skipped += 1
            continue
        start = i * args.chunk_size
        end = min(start + args.chunk_size, total)
        s1_chunk = s1_slice[start:end]
        n_pairs, n_quality, n_country = process_chunk(
            i, s1_chunk,
            inv_struct, token_index,
            sorted_canon, ids_aligned,
            cand_view, args, candidate_source,
            mh_lsh=mh_lsh, mh_sigs=mh_sigs,
        )
        total_pairs += n_pairs

    if skipped:
        print(f"[resume] skipped {skipped:,} already-on-disk chunks", flush=True)

    print(f"[done] {n_chunks} chunks, {total_pairs:,} candidate pairs total", flush=True)

    if args.dry_run:
        print("[dry-run] skipping final concat. Per-chunk parquets in artifacts/_chunks/.", flush=True)
        print("           Re-run without --dry-run for the real run.", flush=True)
        return 0

    # ---- 6. Final concat ----
    print("[concat] combining chunk parquets ...", flush=True)
    t0 = time.time()
    chunk_files = sorted(CHUNK_DIR.glob(f"block_{args.suffix}_*.parquet"))
    final = pl.concat([pl.read_parquet(f) for f in chunk_files], how="vertical_relaxed")
    out_path = ARTIFACTS / f"block_{args.suffix}_features.parquet"
    final.write_parquet(out_path, compression="zstd", compression_level=3)
    print(f"        {final.height:,} pairs, {final.width} cols, {time.time() - t0:.1f}s", flush=True)
    print(f"        wrote {out_path}", flush=True)

    # ---- 7. Cleanup intermediates ----
    for f in chunk_files:
        try:
            f.unlink()
        except OSError:
            pass
    try:
        CHUNK_DIR.rmdir()
    except OSError:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
