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
import os
import re
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
def build_structural_indexes(cand: pl.DataFrame, cap: int) -> dict[str, dict[str, list[str]]]:
    """Index 1: 8 structural inverted indexes; each bucket capped at `cap`."""
    print(f"[index-1] building 8 structural inverted indexes (cap={cap}) ...", flush=True)
    inv: dict[str, dict[str, list[str]]] = {}
    for name, expr in KEY_EXTRACTORS.items():
        t0 = time.time()
        df = (cand.with_columns(expr.alias("_k"))
                  .filter(pl.col("_k").is_not_null() & (pl.col("_k") != ""))
                  .select(["entity_id", "_k"])
                  .unique(subset=["_k", "entity_id"]))
        grouped = (df.group_by("_k")
                     .agg(pl.col("entity_id"))
                     .with_columns(pl.col("entity_id").list.slice(0, cap).alias("capped")))
        inv[name] = dict(zip(grouped["_k"].to_list(), grouped["capped"].to_list()))
        n_keys = len(inv[name])
        n_total = sum(len(v) for v in inv[name].values())
        print(f"   [{name:>11s}] {n_keys:>7,} unique keys, {n_total:>9,} ids in {time.time() - t0:.1f}s", flush=True)
    return inv


def build_token_index(cand: pl.DataFrame, cap: int) -> dict[str, list[tuple[str, str]]]:
    """Index 2: word-token inverted index (name_clean + addr_clean).

    Returns dict[token -> list[(entity_id, country)]] capped at `cap` per token.
    Storing country inside each entry avoids a second lookup at probe time.
    """
    print(f"[index-2] building word-token inverted index (cap={cap}) ...", flush=True)
    t0 = time.time()
    index: dict[str, list[tuple[str, str]]] = {}
    for i, row in enumerate(cand.iter_rows(named=True)):
        text = (row.get("name_clean") or "") + " " + (row.get("addr_clean") or "")
        toks = set(_tokenize(text))
        if not toks:
            continue
        ent_id = row["entity_id"]
        country = row.get("country") or ""
        for tok in toks:
            lst = index.get(tok)
            if lst is None:
                index[tok] = [(ent_id, country)]
            elif len(lst) < cap:
                lst.append((ent_id, country))
        if (i + 1) % 500_000 == 0:
            print(f"        processed {i + 1:,} / {cand.height:,}", flush=True)
    n_tokens = len(index)
    n_total = sum(len(v) for v in index.values())
    print(f"        {n_tokens:,} unique tokens, {n_total:,} (token,id) pairs in {time.time() - t0:.1f}s",
          flush=True)
    return index


def build_sorted_neighborhood(cand: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Index 3: sorted-token neighborhood.

    Returns (sorted_canon, ids_aligned) — numpy object arrays of equal length N
    (one per unique canonical). Sort order in `sorted_canon` is the lexicographic
    order of canonical_strings; ids_aligned[k] is the cand entity_id that maps
    to sorted_canon[k]. Country is NOT stored here because we look it up via
    `cand_dict` in `attach_fields` (avoids 5M wasted numpy entries).
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
    """Probe 8 structural keys per S1; return pairs with `n_struct_keys`."""
    keys_exprs = [expr.alias(f"_bk_{name}") for name, expr in KEY_EXTRACTORS.items()]
    rows = chunk.with_columns(keys_exprs).select(
        ["source1_entity_id"] + [f"_bk_{n}" for n in KEY_EXTRACTORS]
    ).to_dicts()

    cand_count: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        s1_id = row["source1_entity_id"]
        for name in KEY_EXTRACTORS:
            val = row[f"_bk_{name}"]
            if val is None or val == "":
                continue
            bucket = inv[name].get(val)
            if not bucket:
                continue
            for cid in bucket:
                cand_count[s1_id][cid] += 1

    pairs: list[tuple[str, str, int]] = []
    for s1_id, cands in cand_count.items():
        ranked = sorted(cands.items(), key=lambda x: -x[1])
        for cid, cnt in ranked[:top_k]:
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


def probe_tokens(chunk: pl.DataFrame, token_index: dict, top_k: int) -> pl.DataFrame:
    """Probe token inverted index per S1; return pairs with `n_tokens_shared`."""
    cand_count: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    s1_rows = chunk.select(["source1_entity_id", "name_clean", "addr_clean"]).to_dicts()
    for row in s1_rows:
        s1_id = row["source1_entity_id"]
        text = (row.get("name_clean") or "") + " " + (row.get("addr_clean") or "")
        toks = set(_tokenize(text))
        if not toks:
            continue
        for tok in toks:
            bucket = token_index.get(tok)
            if bucket:
                for cid, _ctry in bucket:
                    cand_count[s1_id][cid] += 1

    pairs: list[tuple[str, str, int]] = []
    for s1_id, cands in cand_count.items():
        ranked = sorted(cands.items(), key=lambda x: -x[1])
        for cid, cnt in ranked[:top_k]:
            pairs.append((s1_id, cid, cnt))

    if not pairs:
        return pl.DataFrame(schema={
            "source1_entity_id": pl.Utf8,
            "candidate_entity_id": pl.Utf8,
            "n_tokens_shared": pl.Int8,
        })
    return pl.DataFrame(
        pairs,
        schema=["source1_entity_id", "candidate_entity_id", "n_tokens_shared"],
        orient="row",
    )


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
S1_KEYS_NO_COUNTRY = [
    "name_clean", "name_latin", "name_dev_ratio", "name_missing",
    "addr_clean", "addr_latin", "addr_missing",
    "addr_zip", "addr_state", "addr_city",
    "addr_first_word", "addr_last_word",
    "addr_house_number", "addr_road", "addr_unit", "addr_suburb",
]

M_KEYS = [
    "country", "name_clean", "name_latin", "name_dev_ratio", "name_missing",
    "addr_clean", "addr_latin", "addr_missing",
    "addr_zip", "addr_state", "addr_city",
    "addr_first_word", "addr_last_word",
    "addr_house_number", "addr_road", "addr_unit", "addr_suburb",
]


def attach_fields(pairs: pl.DataFrame, chunk: pl.DataFrame,
                  cand_dict: dict) -> pl.DataFrame:
    """Attach S1 + candidate fields for feature computation.

    - s1 fields prefixed `s1_`     (e.g., s1_country, s1_name_clean)
    - candidate fields prefixed `m__` (e.g., m__country, m__name_clean)
    """
    s1_records = chunk.select(["source1_entity_id"] + ["country"] + S1_KEYS_NO_COUNTRY).to_dicts()
    s1_dict = {r["source1_entity_id"]: r for r in s1_records}
    pair_rows = pairs.to_dicts()

    out_rows: list[dict] = []
    for r in pair_rows:
        s1_id = r["source1_entity_id"]
        cand_id = r["candidate_entity_id"]
        s1 = s1_dict.get(s1_id)
        m = cand_dict.get(cand_id)
        new = dict(r)
        if s1:
            for k in ["country"] + S1_KEYS_NO_COUNTRY:
                new["s1_" + k] = s1.get(k)
        else:
            # Defensive: keep s1_* columns populated as None so downstream
            # filters/expressions don't throw ColumnNotFoundError.
            for k in ["country"] + S1_KEYS_NO_COUNTRY:
                new.setdefault("s1_" + k, None)
        # Always populate m__* keys (None if cand is missing) so the
        # filter & feature expressions have stable schema.
        for k in M_KEYS:
            new["m__" + k] = m[k] if m else None
        out_rows.append(new)

    # Infer schema from ALL rows (not just first 10000). With the v3 union
    # of 3 indexes producing ~100k+ pairs per chunk, columns like
    # addr_house_number can have mixed int/str representations and
    # infer_schema_length=10000 misses the str-only rows at the tail.
    return pl.from_dicts(out_rows, infer_schema_length=len(out_rows))


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
# Per-chunk pipeline
# ---------------------------------------------------------------------------
def process_chunk(idx: int, s1_chunk: pl.DataFrame,
                  inv_struct, token_index,
                  sorted_canon, ids_aligned,
                  cand_dict: dict, args, candidate_source: str) -> tuple[int, int, int]:
    t_chunk = time.time()

    # --- Phase 1: probe each of 3 indexes (per-S1 cap from --top-k-index) ---
    struct_df = probe_structural(s1_chunk, inv_struct, top_k=args.top_k_index)
    token_df  = probe_tokens(s1_chunk, token_index, top_k=args.top_k_index)
    sn_df     = probe_sorted_neighborhood(
                    s1_chunk, sorted_canon, ids_aligned,
                    window=args.sn_window)
    n_struct = len(struct_df)
    n_tok = len(token_df)
    n_sn = len(sn_df)

    # --- Phase 2: union outer joins; fill nulls with 0; boolean flags ---
    merged = struct_df.join(token_df,
                            on=["source1_entity_id", "candidate_entity_id"],
                            how="full", coalesce=True)
    merged = merged.join(sn_df,
                         on=["source1_entity_id", "candidate_entity_id"],
                         how="full", coalesce=True)
    merged = merged.with_columns([
        pl.col("n_struct_keys").fill_null(0).cast(pl.Int8),
        pl.col("n_tokens_shared").fill_null(0).cast(pl.Int8),
        pl.col("sortedn_rank").fill_null(0).cast(pl.Int16),
    ])
    merged = merged.with_columns([
        (pl.col("n_struct_keys") > 0).cast(pl.Int8).alias("from_struct"),
        (pl.col("n_tokens_shared") > 0).cast(pl.Int8).alias("from_token"),
        (pl.col("sortedn_rank") > 0).cast(pl.Int8).alias("from_sortedn"),
    ])

    # --- Phase 3: attach S1 + M fields (uses cand_dict; ~no extra RAM) ---
    merged = attach_fields(merged, s1_chunk, cand_dict)

    # --- Phase 4: HARD country filter (mandatory; STATUS.md "Hard-fail bugs #5").
    # MUST reassign `merged` — without this, cross-country pairs pass into
    # Phase E and the LightGBM classifier wastes capacity on easy negatives.
    merged = merged.filter(
        pl.col("s1_country").is_not_null()
        & pl.col("m__country").is_not_null()
        & (pl.col("s1_country") == pl.col("m__country"))
    )
    n_after_country = len(merged)

    # --- Phase 5: quality-tier OR filter (4 floors; STATUS.md §Phase C v3) ---
    # floor A: n_struct_keys >= 2
    # floor B: n_tokens_shared >= 2
    # floor C: s1_road == m_road AND s1_city == m_city (both non-empty)
    # floor D: 0 < sortedn_rank <= 10 AND n_tokens_shared >= 1
    quality_pass_expr = (
        (pl.col("n_struct_keys") >= 2)
        | (pl.col("n_tokens_shared") >= 2)
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
    )
    merged = merged.filter(quality_pass_expr)
    n_after_quality = len(merged)

    # --- Phase 6: composite score (for top-50 tiebreak only) ---
    # struct_score capped at 3 because city/state/country trivially match.
    # Cast to Float32 explicitly so the parquet column is Float32 (otherwise
    # the Python float literals would promote the result to Float64, bloating
    # the column 2× on disk).
    merged = merged.with_columns([
        (
            0.20 * pl.col("n_struct_keys").cast(pl.Float32) / 3.0
            + 0.45 * pl.col("n_tokens_shared").cast(pl.Float32) / 5.0
            + 0.20 * pl.col("sortedn_rank").gt(0).cast(pl.Float32)
            + 0.15 * pl.when(pl.col("sortedn_rank") > 0)
                  .then(1.0 / (1.0 + pl.col("sortedn_rank").cast(pl.Float32)))
                  .otherwise(0.0)
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
    merged = merged.with_columns(pl.lit(candidate_source).alias("candidate_source"))
    canonical = [
        "source1_entity_id", "candidate_entity_id", "candidate_source",
        "n_struct_keys", "n_tokens_shared", "sortedn_rank",
        "from_struct", "from_token", "from_sortedn",
        "block_score",
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
    p.add_argument("--top-k", type=int, default=50,
                   help="Final per-S1 cap (HARD 50 per STATUS.md; never lower without recall re-check)")
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

    # ---- 2. Build 3 indexes ----
    print("[indexes] building v3 indexes on candidate side ...", flush=True)
    t0 = time.time()
    inv_struct = build_structural_indexes(cand, cap=args.bucket_cap)
    token_index = build_token_index(cand, cap=args.token_cap)
    sorted_canon, ids_aligned = build_sorted_neighborhood(cand)
    print(f"[indexes] all built in {time.time() - t0:.1f}s", flush=True)

    # ---- 3. Build cand lookup dict (so attach_fields is O(pairs) not O(N) joins) ----
    print("[cand_dict] building candidate lookup dict ...", flush=True)
    t0 = time.time()
    cand_dict: dict[str, dict] = {}
    for r in cand.iter_rows(named=True):
        cand_dict[r["entity_id"]] = r
    print(f"        {len(cand_dict):,} entries in {time.time() - t0:.1f}s", flush=True)
    del cand  # free the polars frame; we only need cand_dict

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
            cand_dict, args, candidate_source,
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
