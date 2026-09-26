"""Phase C+D: hybrid blocking (8 structural + char-trigram) + 27-feature extraction.

For S1 ↔ {S2, S3} candidate generation, this script:

  1. Builds 8 cheap structural inverted indexes on the candidate side.
  2. Builds a char-trigram inverted index on (name_latin + ' ' + addr_latin)
     for fuzzy blocking (each trigram → list of S2 ids, capped at 100).
  3. For each 10K-S1 chunk:
       - structural probe (8 polars inner-joins) → candidates_A
       - trigram probe (per-S1 trigram union) → candidates_B
       - union + dedup + re-rank by combined score, top-25 per S1
       - attach S1 + S2 fields
       - compute 27 features (booleans in polars; 10 fuzzy in Python)
       - write per-chunk parquet
  4. Concat per-chunk parquets into final output, delete intermediates.

RAM budget: ~2.5 GB peak on 8 GB machine.

CLI:
    python scripts/block_features.py --candidate-source S2
    python scripts/block_features.py --candidate-source S3
    python scripts/block_features.py --candidate-source S2 --dry-run
"""
from __future__ import annotations

import argparse
import gc
import glob
import io
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import numpy as np
import polars as pl
from rapidfuzz import fuzz

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
CHUNK_DIR = ARTIFACTS / "_chunks"
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

S1_NORM = ARTIFACTS / "s1_norm_train.parquet"
S2_NORM = ARTIFACTS / "s2_norm_train.parquet"
S3_NORM = ARTIFACTS / "s3_norm_train.parquet"

# 17 fields loaded from each parquet (entity_id + 16 features)
FEATURE_FIELDS = [
    "country", "name_clean", "name_latin", "name_tokens",
    "name_dev_ratio", "name_missing",
    "addr_clean", "addr_latin",
    "addr_zip", "addr_state", "addr_city",
    "addr_first_word", "addr_last_word",
    "addr_house_number", "addr_road", "addr_unit", "addr_suburb",
    "addr_missing",
]

# 8 cheap structural blocking keys (matches project's K1-K6, K8, K9 minus K7)
KEY_EXTRACTORS: dict[str, pl.Expr] = {
    "city":      pl.col("addr_city").str.strip_chars().str.to_lowercase(),
    "state":     pl.col("addr_state").str.strip_chars().str.to_lowercase(),
    "road":      pl.col("addr_road").str.strip_chars().str.to_lowercase(),
    "house":     pl.col("addr_house_number").str.strip_chars().str.to_lowercase(),
    "name_fw":   pl.col("name_clean").str.split(" ").list.first()
                       .fill_null("").str.strip_chars().str.to_lowercase(),
    "addr_fw":   pl.col("addr_first_word").str.strip_chars().str.to_lowercase(),
    "city_state": pl.concat_str(
        [pl.col("addr_city"), pl.lit("|"), pl.col("addr_state")],
        separator="", ignore_nulls=False,
    ).str.strip_chars().str.to_lowercase(),
    "house_road": pl.concat_str(
        [pl.col("addr_house_number"), pl.lit("|"), pl.col("addr_road")],
        separator="", ignore_nulls=False,
    ).str.strip_chars().str.to_lowercase(),
}


# ---------------------------------------------------------------------------
def _s(x):
    """Coerce nullable/NaN string to '' (used in the rapidfuzz loop)."""
    if x is None:
        return ""
    if isinstance(x, float) and x != x:
        return ""
    return str(x)


# ---------------------------------------------------------------------------
def load_sources(candidate_source: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    print(f"[load] s1_norm_train ...", flush=True)
    t0 = time.time()
    s1 = pl.read_parquet(S1_NORM, columns=["entity_id"] + FEATURE_FIELDS)
    print(f"        {s1.height:,} rows, {time.time() - t0:.1f}s", flush=True)

    cand_path = S2_NORM if candidate_source == "S2" else S3_NORM
    print(f"[load] {cand_path.name} ...", flush=True)
    t0 = time.time()
    cand = pl.read_parquet(cand_path, columns=["entity_id"] + FEATURE_FIELDS)
    print(f"        {cand.height:,} rows, {time.time() - t0:.1f}s", flush=True)
    return s1, cand


# ---------------------------------------------------------------------------
def build_inverted_indexes(cand: pl.DataFrame, cap: int) -> dict[str, dict[str, list[str]]]:
    """For each blocking key, build {key_value: list[entity_id]}, capped at `cap` per bucket."""
    inv: dict[str, dict[str, list[str]]] = {}
    for key, expr in KEY_EXTRACTORS.items():
        t0 = time.time()
        df = (cand.with_columns(expr.alias("_k"))
                  .filter(pl.col("_k").is_not_null() & (pl.col("_k") != ""))
                  .select(["entity_id", "_k"])
                  .unique(subset=["_k", "entity_id"]))
        grouped = (df.group_by("_k")
                     .agg(pl.col("entity_id"))
                     .with_columns(pl.col("entity_id").list.slice(0, cap).alias("capped")))
        inv[key] = dict(zip(grouped["_k"].to_list(), grouped["capped"].to_list()))
        n_keys = len(inv[key])
        n_total = sum(len(v) for v in inv[key].values())
        print(f"  [{key:>11s}] {n_keys:>7,} unique keys, {n_total:>9,} s2 ids mapped in {time.time() - t0:.1f}s", flush=True)
    return inv


# ---------------------------------------------------------------------------
def build_trigram_index(cand: pl.DataFrame, cap: int = 100) -> dict[str, list[str]]:
    """Char-trigram inverted index on (name_latin + ' ' + addr_latin).

    Returns dict[trigram -> list[entity_id]] capped at `cap` ids per trigram.
    """
    print(f"[trigram] building on {cand.height:,} candidate docs (cap={cap}) ...", flush=True)
    t0 = time.time()
    index: dict[str, list[str]] = {}
    for i, row in enumerate(cand.iter_rows(named=True)):
        nl = row.get("name_latin") or ""
        al = row.get("addr_latin") or ""
        text = ((nl + " " + al).strip()) or " "
        if len(text) < 3:
            text = (text + "  ")[:3]
        # unique trigrams only (avoid one-doc-dominates-a-trigram)
        trigrams: set[str] = set(text[k:k + 3] for k in range(len(text) - 2))
        for tg in trigrams:
            lst = index.get(tg)
            if lst is None:
                index[tg] = [row["entity_id"]]
            elif len(lst) < cap:
                lst.append(row["entity_id"])
        if (i + 1) % 500_000 == 0:
            print(f"        processed {i + 1:,} / {cand.height:,}", flush=True)
    n_trigrams = len(index)
    n_total = sum(len(v) for v in index.values())
    print(f"        {n_trigrams:,} unique trigrams, {n_total:,} (trigram, id) pairs in {time.time() - t0:.1f}s", flush=True)
    return index


def probe_trigram(chunk: pl.DataFrame, index: dict, top_k: int) -> pl.DataFrame:
    """Per-S1: extract trigrams → union candidates → rank by trigram-overlap count."""
    cand_count: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in chunk.iter_rows(named=True):
        s1_id = row["source1_entity_id"]
        nl = row.get("name_latin") or ""
        al = row.get("addr_latin") or ""
        text = ((nl + " " + al).strip()) or " "
        if len(text) < 3:
            text = (text + "  ")[:3]
        seen: set[str] = set()
        for k in range(len(text) - 2):
            tg = text[k:k + 3]
            if tg in seen:
                continue
            seen.add(tg)
            lst = index.get(tg)
            if lst:
                for cid in lst:
                    cand_count[s1_id][cid] += 1

    pairs: list[tuple[str, str, int]] = []
    for s1_id, cands in cand_count.items():
        sorted_cands = sorted(cands.items(), key=lambda x: -x[1])
        for cid, cnt in sorted_cands[:top_k]:
            pairs.append((s1_id, cid, cnt))

    if not pairs:
        return pl.DataFrame(schema={
            "source1_entity_id": pl.Utf8,
            "candidate_entity_id": pl.Utf8,
            "n_trigrams": pl.Int16,
        })
    return pl.DataFrame(
        pairs,
        schema=["source1_entity_id", "candidate_entity_id", "n_trigrams"],
        orient="row",
    )


# ---------------------------------------------------------------------------
def probe_structural(chunk: pl.DataFrame, inv: dict, top_k: int = 25) -> pl.DataFrame:
    """8 cheap structural probes via Python dict lookup. Cap top-K per S1.

    Faster than polars inner-joins because we never materialize the giant
    bucket DataFrames for popular keys.
    """
    # Compute all 8 keys in one vectorized polars call
    keys_exprs = [(pl.format("{}", expr) if False else expr).alias(f"_bk_{name}")
                  for name, expr in KEY_EXTRACTORS.items()]
    chunk_with_keys = chunk.with_columns(keys_exprs)
    rows = chunk_with_keys.select(
        ["source1_entity_id"] + [f"_bk_{n}" for n in KEY_EXTRACTORS]
    ).to_dicts()

    cand_count: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        s1_id = row["source1_entity_id"]
        for key in KEY_EXTRACTORS:
            val = row[f"_bk_{key}"]
            if val is None or val == "":
                continue
            bucket = inv[key].get(val)
            if not bucket:
                continue
            for cid in bucket:
                cand_count[s1_id][cid] += 1

    pairs: list[tuple[str, str, int]] = []
    for s1_id, cands in cand_count.items():
        sorted_cands = sorted(cands.items(), key=lambda x: -x[1])
        for cid, cnt in sorted_cands[:top_k]:
            pairs.append((s1_id, cid, cnt))

    if not pairs:
        return pl.DataFrame(schema={
            "source1_entity_id": pl.Utf8,
            "candidate_entity_id": pl.Utf8,
            "n_struct_keys": pl.Int8,
        })
    return pl.DataFrame(
        pairs,
        schema=["source1_entity_id", "candidate_entity_id", "n_struct_keys"],
        orient="row",
    )


def probe_minhash(chunk: pl.DataFrame, lsh, cand_ids_set: set[str],
                  num_perm: int) -> pl.DataFrame:
    """DEPRECATED: replaced by probe_trigram. Kept for reference only."""
    raise NotImplementedError("probe_minhash deprecated; use probe_trigram")


# ---------------------------------------------------------------------------
def attach_fields(pairs: pl.DataFrame, chunk: pl.DataFrame, cand_dict: dict) -> pl.DataFrame:
    """Attach S1 fields (from chunk) and S2 fields (from cand_dict) for feature computation.

    cand_dict maps candidate entity_id → tuple of 17 fields. Built once in main().
    """
    # S1 fields: dict[source1_entity_id -> tuple]
    s1_records = chunk.select([
        "source1_entity_id", "country", "name_clean", "name_latin",
        "name_dev_ratio", "name_missing",
        "addr_clean", "addr_latin", "addr_missing",
        "addr_zip", "addr_state", "addr_city",
        "addr_first_word", "addr_last_word",
        "addr_house_number", "addr_road", "addr_unit", "addr_suburb",
    ]).to_dicts()
    s1_dict = {r["source1_entity_id"]: r for r in s1_records}

    # Materialize pairs as plain Python lists
    pair_rows = pairs.to_dicts()

    S1_KEYS = ["country", "name_clean", "name_latin", "name_dev_ratio", "name_missing",
               "addr_clean", "addr_latin", "addr_missing", "addr_zip", "addr_state",
               "addr_city", "addr_first_word", "addr_last_word",
               "addr_house_number", "addr_road", "addr_unit", "addr_suburb"]

    out_rows: list[dict] = []
    for r in pair_rows:
        s1_id = r["source1_entity_id"]
        cand_id = r["candidate_entity_id"]
        s1 = s1_dict.get(s1_id)
        m = cand_dict.get(cand_id)
        new = {k: v for k, v in r.items()}  # copy pair row (n_struct_keys, n_trigrams, etc.)
        if s1:
            for k in S1_KEYS:
                new["s1__" + k] = s1.get(k)
        # Rename "s1__country" → "s1_country" (legacy naming)
        if s1 and "s1__country" in new:
            new["s1_country"] = new.pop("s1__country")
        if m:
            new["m__country"] = m["country"]
            new["m__name_clean"] = m["name_clean"]
            new["m__name_latin"] = m["name_latin"]
            new["m__name_dev_ratio"] = m["name_dev_ratio"]
            new["m__name_missing"] = m["name_missing"]
            new["m__addr_clean"] = m["addr_clean"]
            new["m__addr_latin"] = m["addr_latin"]
            new["m__addr_missing"] = m["addr_missing"]
            new["m__addr_zip"] = m["addr_zip"]
            new["m__addr_state"] = m["addr_state"]
            new["m__addr_city"] = m["addr_city"]
            new["m__addr_first_word"] = m["addr_first_word"]
            new["m__addr_last_word"] = m["addr_last_word"]
            new["m__addr_house_number"] = m["addr_house_number"]
            new["m__addr_road"] = m["addr_road"]
            new["m__addr_unit"] = m["addr_unit"]
            new["m__addr_suburb"] = m["addr_suburb"]
        out_rows.append(new)

    return pl.from_dicts(out_rows, infer_schema_length=10000)


# ---------------------------------------------------------------------------
def compute_polars_features(df: pl.DataFrame) -> pl.DataFrame:
    """Cheap boolean / structural features via polars vectorised expressions."""
    return df.with_columns([
        (pl.col("s1_country") == pl.col("m__country")).cast(pl.Int8).fill_null(0).alias("country_eq"),

        # name_first_token_eq (fill nulls first; str.split requires String dtype)
        (pl.col("s1__name_clean").fill_null("").str.split(" ").list.first()
         == pl.col("m__name_clean").fill_null("").str.split(" ").list.first())
        .cast(pl.Int8).fill_null(0).alias("name_first_token_eq"),

        # name_token_jaccard: |A ∩ B| / |A ∪ B| (vectorised)
        (
            pl.col("s1__name_clean").fill_null("").str.split(" ")
              .list.set_intersection(pl.col("m__name_clean").fill_null("").str.split(" "))
              .list.len().cast(pl.Float32)
            / pl.col("s1__name_clean").fill_null("").str.split(" ")
              .list.set_union(pl.col("m__name_clean").fill_null("").str.split(" "))
              .list.len().cast(pl.Float32).fill_null(1.0)
        ).alias("name_token_jaccard"),

        # name_n_chars_diff (cast lengths to Int32 BEFORE subtracting to avoid u32 overflow)
        (pl.col("s1__name_clean").fill_null("").str.len_chars().cast(pl.Int32)
         - pl.col("m__name_clean").fill_null("").str.len_chars().cast(pl.Int32)).abs()
        .alias("name_n_chars_diff"),

        # cross_script_pair: dev_ratio > 0.2 on one side, ≤ 0.2 on the other
        ((pl.col("s1__name_dev_ratio").fill_null(0.0) > 0.2)
         != (pl.col("m__name_dev_ratio").fill_null(0.0) > 0.2))
        .cast(pl.Int8).alias("cross_script_pair"),

        # 8 structured address eq
        (pl.col("s1__addr_first_word").fill_null("__null__")
         == pl.col("m__addr_first_word").fill_null("__null__"))
        .cast(pl.Int8).alias("addr_first_word_eq"),
        (pl.col("s1__addr_last_word").fill_null("__null__")
         == pl.col("m__addr_last_word").fill_null("__null__"))
        .cast(pl.Int8).alias("addr_last_word_eq"),
        (pl.col("s1__addr_city").fill_null("__null__")
         == pl.col("m__addr_city").fill_null("__null__"))
        .cast(pl.Int8).alias("addr_city_eq"),
        (pl.col("s1__addr_house_number").fill_null("__null__")
         == pl.col("m__addr_house_number").fill_null("__null__"))
        .cast(pl.Int8).alias("addr_house_number_eq"),
        (pl.col("s1__addr_state").fill_null("__null__")
         == pl.col("m__addr_state").fill_null("__null__"))
        .cast(pl.Int8).alias("addr_state_eq"),
        (pl.col("s1__addr_road").fill_null("__null__")
         == pl.col("m__addr_road").fill_null("__null__"))
        .cast(pl.Int8).alias("addr_road_eq"),
        (pl.col("s1__addr_zip").fill_null("__null__")
         == pl.col("m__addr_zip").fill_null("__null__"))
        .cast(pl.Int8).alias("addr_zip_eq"),
        (pl.col("s1__addr_unit").fill_null("__null__")
         == pl.col("m__addr_unit").fill_null("__null__"))
        .cast(pl.Int8).alias("addr_unit_eq"),
        (pl.col("s1__addr_suburb").fill_null("__null__")
         == pl.col("m__addr_suburb").fill_null("__null__"))
        .cast(pl.Int8).alias("addr_suburb_eq"),

        # 4 missingness flags (passthrough from norm parquet; coerce to Int8)
        pl.col("s1__name_missing").cast(pl.Int8).fill_null(0).alias("s1_name_missing"),
        pl.col("m__name_missing").cast(pl.Int8).fill_null(0).alias("m_name_missing"),
        pl.col("s1__addr_missing").cast(pl.Int8).fill_null(0).alias("s1_addr_missing"),
        pl.col("m__addr_missing").cast(pl.Int8).fill_null(0).alias("m_addr_missing"),
    ])


# ---------------------------------------------------------------------------
def compute_fuzzy_features(df: pl.DataFrame) -> pl.DataFrame:
    """10 rapidfuzz metrics per pair, computed in a Python loop."""
    rows = df.to_dicts()
    n = len(rows)

    name_token_set      = [-1.0] * n
    name_partial        = [-1.0] * n
    name_token_sort     = [-1.0] * n
    name_ratio          = [-1.0] * n
    name_latin_token_set = [-1.0] * n
    addr_token_set      = [-1.0] * n
    addr_partial        = [-1.0] * n
    addr_token_sort     = [-1.0] * n
    addr_ratio          = [-1.0] * n
    addr_latin_token_set = [-1.0] * n

    for i, r in enumerate(rows):
        a_name = _s(r.get("s1__name_clean"))
        b_name = _s(r.get("m__name_clean"))
        a_nl   = _s(r.get("s1__name_latin"))
        b_nl   = _s(r.get("m__name_latin"))
        a_addr = _s(r.get("s1__addr_clean"))
        b_addr = _s(r.get("m__addr_clean"))
        a_al   = _s(r.get("s1__addr_latin"))
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
        pl.Series("name_token_set_ratio", name_token_set, dtype=pl.Float32),
        pl.Series("name_partial_ratio", name_partial, dtype=pl.Float32),
        pl.Series("name_token_sort_ratio", name_token_sort, dtype=pl.Float32),
        pl.Series("name_ratio", name_ratio, dtype=pl.Float32),
        pl.Series("name_latin_token_set_ratio", name_latin_token_set, dtype=pl.Float32),
        pl.Series("addr_token_set_ratio", addr_token_set, dtype=pl.Float32),
        pl.Series("addr_partial_ratio", addr_partial, dtype=pl.Float32),
        pl.Series("addr_token_sort_ratio", addr_token_sort, dtype=pl.Float32),
        pl.Series("addr_ratio", addr_ratio, dtype=pl.Float32),
        pl.Series("addr_latin_token_set_ratio", addr_latin_token_set, dtype=pl.Float32),
    ])


# ---------------------------------------------------------------------------
def process_chunk(idx: int, s1_chunk: pl.DataFrame, inv, trigram_index,
                 cand_dict: dict, args, candidate_source: str) -> int:
    t_chunk = time.time()

    # Probe
    struct_df = probe_structural(s1_chunk, inv, top_k=args.top_k_tfidf)
    tg_df     = probe_trigram(s1_chunk, trigram_index, args.top_k_tfidf)

    # Union + dedup + re-rank
    merged = struct_df.join(tg_df,
                            on=["source1_entity_id", "candidate_entity_id"],
                            how="full", coalesce=True)
    merged = merged.with_columns([
        pl.col("n_struct_keys").fill_null(0).cast(pl.Int8),
        pl.col("n_trigrams").fill_null(0).cast(pl.Int16),
    ])
    # Score: structural 50% (capped at 3 keys), trigram-overlap 50% (capped at 30)
    merged = merged.with_columns(
        (pl.col("n_struct_keys").cast(pl.Float32) * 0.5 / 3.0
         + pl.col("n_trigrams").cast(pl.Float32) * 0.5 / 30.0).alias("block_score")
    )
    n_before_cap = len(merged)
    merged = (merged.with_columns(
        pl.col("block_score").rank(method="ordinal", descending=True)
          .over("source1_entity_id").alias("_r"))
        .filter(pl.col("_r") <= args.top_k)
        .drop("_r"))
    n_after_cap = len(merged)

    # Attach fields (Python dict lookup, no polars joins against 5M cand)
    merged = attach_fields(merged, s1_chunk, cand_dict)

    # Polars features
    merged = compute_polars_features(merged)

    # Python fuzzy features
    merged = compute_fuzzy_features(merged)

    # candidate_source
    merged = merged.with_columns(pl.lit(candidate_source).alias("candidate_source"))

    # Reorder columns to the canonical schema
    canonical = [
        "source1_entity_id", "candidate_entity_id", "candidate_source",
        "n_struct_keys", "n_trigrams", "block_score",
        "s1_country", "m__country",
        "country_eq", "name_first_token_eq", "name_token_jaccard",
        "name_n_chars_diff", "cross_script_pair",
        "addr_first_word_eq", "addr_last_word_eq", "addr_city_eq",
        "addr_house_number_eq", "addr_state_eq", "addr_road_eq",
        "addr_zip_eq", "addr_unit_eq", "addr_suburb_eq",
        "s1_name_missing", "m_name_missing", "s1_addr_missing", "m_addr_missing",
        "name_token_set_ratio", "name_partial_ratio", "name_token_sort_ratio",
        "name_ratio", "name_latin_token_set_ratio",
        "addr_token_set_ratio", "addr_partial_ratio", "addr_token_sort_ratio",
        "addr_ratio", "addr_latin_token_set_ratio",
    ]
    merged = merged.select([c for c in canonical if c in merged.columns])

    # Write per-chunk parquet
    chunk_path = CHUNK_DIR / f"block_{args.suffix}_{idx:04d}.parquet"
    merged.write_parquet(chunk_path, compression="zstd", compression_level=3)

    elapsed = time.time() - t_chunk
    n_s1 = s1_chunk.height
    print(f"  chunk {idx + 1:4d}/{args.n_chunks}  S1_rows={n_s1:,}  "
          f"struct={len(struct_df):,} tg={len(tg_df):,} after_cap={n_after_cap:,}  ({elapsed:.1f}s)",
          flush=True)

    del struct_df, tg_df, merged
    gc.collect()
    return n_after_cap


# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate-source", required=True, choices=["S2", "S3"])
    p.add_argument("--top-k", type=int, default=25)
    p.add_argument("--top-k-tfidf", type=int, default=50,
                   help="Trigram-blocker candidates per S1 (before final cap)")
    p.add_argument("--chunk-size", type=int, default=10000)
    p.add_argument("--bucket-cap", type=int, default=500)
    p.add_argument("--trigram-cap", type=int, default=100,
                   help="Max S2 ids per trigram bucket (caps RAM of fuzzy index)")
    p.add_argument("--max-chunks", type=int, default=None,
                   help="(testing) process only N chunks")
    p.add_argument("--dry-run", action="store_true",
                   help="Process a 5% slice and skip final concat")
    p.add_argument("--suffix", default=None,
                   help="Output suffix (default: candidate source)")
    args = p.parse_args()

    candidate_source = args.candidate_source
    args.suffix = args.suffix or candidate_source

    print("=" * 70, flush=True)
    print(f"Phase C+D: S1 vs {candidate_source}", flush=True)
    print(f"  top_k={args.top_k}, top_k_tfidf={args.top_k_tfidf}, "
          f"chunk_size={args.chunk_size:,}, bucket_cap={args.bucket_cap}, "
          f"trigram_cap={args.trigram_cap}", flush=True)
    if args.dry_run:
        print("  *** DRY RUN: 5% slice, no final concat ***", flush=True)
    print("=" * 70, flush=True)

    # Load sources
    s1, cand = load_sources(candidate_source)

    # Rename entity_id → source1_entity_id in s1 chunk, keep cand as-is
    s1 = s1.rename({"entity_id": "source1_entity_id"})

    # Build structural inverted indexes (one-time)
    print("[indexes] building 8 structural inverted indexes on candidate side ...", flush=True)
    t0 = time.time()
    inv = build_inverted_indexes(cand, args.bucket_cap)
    print(f"        total: {time.time() - t0:.1f}s", flush=True)

    # Build char-trigram inverted index (one-time)
    trigram_index = build_trigram_index(cand, args.trigram_cap)

    # Build candidate lookup dict for fast field attachment (avoid 5M-row polars join per chunk)
    print("[cand_dict] building candidate lookup dict ...", flush=True)
    t0 = time.time()
    cand_dict: dict[str, dict] = {}
    for r in cand.iter_rows(named=True):
        cand_dict[r["entity_id"]] = r
    print(f"        {len(cand_dict):,} entries in {time.time() - t0:.1f}s", flush=True)
    del cand  # free the polars frame; we only need cand_dict now

    # Determine slice
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

    # Stream chunks
    total_cands = 0
    for i in range(n_chunks):
        start = i * args.chunk_size
        end = min(start + args.chunk_size, total)
        s1_chunk = s1_slice[start:end]
        n_cands = process_chunk(i, s1_chunk, inv, trigram_index,
                                cand_dict, args, candidate_source)
        total_cands += n_cands

    print(f"[done] {n_chunks} chunks, {total_cands:,} candidate pairs total", flush=True)

    if args.dry_run:
        print("[dry-run] skipping final concat; per-chunk parquets in artifacts/_chunks/", flush=True)
        return 0

    # Concat all chunk parquets
    print("[concat] combining chunk parquets ...", flush=True)
    t0 = time.time()
    chunk_files = sorted(CHUNK_DIR.glob(f"block_{args.suffix}_*.parquet"))
    final = pl.concat([pl.read_parquet(f) for f in chunk_files], how="vertical_relaxed")
    out_path = ARTIFACTS / f"block_{args.suffix}_features.parquet"
    final.write_parquet(out_path, compression="zstd", compression_level=3)
    print(f"        {final.height:,} pairs, {final.width} cols, {time.time() - t0:.1f}s", flush=True)
    print(f"        wrote {out_path}", flush=True)

    # Cleanup intermediate chunk files
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