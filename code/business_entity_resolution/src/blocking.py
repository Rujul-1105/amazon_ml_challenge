"""Phase C — blocking / candidate generation (train-only, low-RAM).

Sequential-by-stage architecture to fit in 15 GB RAM:

  BUILD stage (one index at a time, persist to disk, free RAM):
    Stage 1: 9 inverted dicts per country         → inv_country={c}.parquet
    Stage 2: TF-IDF sparse matrix per country     → tfidf_country={c}.npz + vec.pkl
    Stage 3: Faiss ANN index per country (CPU)     → faiss_country={c}.faiss
    Stage 4: MinHash LSH per country              → mhls_name, mhls_addr per country

  QUERY stage (per (country, source)):
    Pass 1: read inverted dicts (cheap) → for each S1, write partial candidates
    Pass 2: read TF-IDF sparse → top-K cosine → append to partials
    Pass 3: read MinHash name → query → append
    Pass 4: read MinHash addr → query → append
    Pass 5: merge partials per S1, cap at top-100 by QRatio, write final parquet

Run:
    python3 -m src.blocking
"""
from __future__ import annotations

import gc
import logging
import pickle
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import polars as pl

from . import config as C
from .io_utils import read_parquet, write_parquet

log = logging.getLogger(__name__)

STOP_WORDS = {
    "the", "a", "an", "and", "of", "in", "at", "on",
    "near", "behind", "opposite", "opp", "next",
}


def first_word(text: str) -> str:
    if not text:
        return ""
    for tok in text.split():
        if tok and len(tok) > 1:
            return tok
    return ""


def last_word(text: str) -> str:
    if not text:
        return ""
    toks = [t for t in text.split() if t and len(t) > 1 and t not in STOP_WORDS]
    return toks[-1] if toks else ""


def _read_free_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except Exception:  # noqa: BLE001
        return 8.0
    return 8.0


# ---------------------------------------------------------------------------
# STAGE 1: Build inverted indexes per country (cheap, ~1 GB)
# ---------------------------------------------------------------------------

# 9 inverted dicts per country (built from union of S2 + S3 for that country)
# Output: artifacts/inv_country={c}.parquet with columns
#   (key_type, key_value, entity_id, source_flag)
# Each row is one mapping from a (key_type, key_value) to a (source, entity) pair.

def build_inverted_indexes_for_country(country: str, s2_c: pl.DataFrame, s3_c: pl.DataFrame,
                                     out_path: Path):
    log.info("[Stage 1] Building 9 inverted indexes for country=%s ...", country)
    t0 = time.time()
    rows = []  # list of dicts

    def add_index(label: str, val: str, eid: str, sf: str):
        v = (val or "").lower().strip()
        if not v:
            return
        rows.append({
            "key_type": label,
            "key_value": v,
            "entity_id": eid,
            "source_flag": sf,
        })

    for df, source_flag in ((s2_c, "S2"), (s3_c, "S3")):
        ids = df["entity_id"].to_list()
        cities = df["addr_city"].to_list()
        states = df["addr_state"].to_list()
        roads = df["addr_road"].to_list()
        houses = df["addr_house_number"].to_list()
        names = df["name_clean"].to_list()
        addrs = df["addr_latin"].to_list()
        for i in range(len(ids)):
            eid = ids[i]
            add_index("city", cities[i], eid, source_flag)
            add_index("state", states[i], eid, source_flag)
            add_index("road", roads[i], eid, source_flag)
            add_index("house", houses[i], eid, source_flag)
            nfw = first_word(names[i] or "")
            if nfw:
                add_index("nfw", nfw, eid, source_flag)
            afw = first_word(addrs[i] or "")
            if afw:
                add_index("afw", afw, eid, source_flag)
            alw = last_word(addrs[i] or "")
            if alw:
                add_index("alw", alw, eid, source_flag)
            city = (cities[i] or "").lower().strip()
            state = (states[i] or "").lower().strip()
            if city and state:
                add_index("city_state", f"{city}|{state}", eid, source_flag)
            house = (houses[i] or "").lower().strip()
            road = (roads[i] or "").lower().strip()
            if house and road:
                add_index("house_road", f"{house}|{road}", eid, source_flag)
    df = pl.from_dicts(rows)
    write_parquet(df, out_path)
    log.info("[Stage 1] Wrote %s (%d rows, %.1fs)", out_path, len(df), time.time() - t0)
    return df


# ---------------------------------------------------------------------------
# STAGE 2: TF-IDF sparse matrix per country (sparse, no Faiss)
# ---------------------------------------------------------------------------

def build_tfidf_for_country(country: str, cand_combined: pl.DataFrame,
                            base_dir: Path):
    """Build TF-IDF sparse matrix on (name_latin + ' ' + addr_latin) for that country.

    Saves: base_dir/tfidf_country={c}.npz (sparse), base_dir/tfidf_country={c}_meta.parquet
    """
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.preprocessing import normalize

    free_gb = _read_free_gb()
    if free_gb < 4.0:
        n_features = 1 << 11  # 2048
    elif free_gb < 7.0:
        n_features = 1 << 12  # 4096
    else:
        n_features = 1 << 13  # 8192

    log.info("[Stage 2] Building TF-IDF (n_features=%d) for country=%s (%d docs, free RAM=%.1f GB)",
             n_features, country, len(cand_combined), free_gb)
    t0 = time.time()
    name_latin = cand_combined["name_latin"].to_list()
    addr_latin = cand_combined["addr_latin"].to_list()
    docs = []
    for n, a in zip(name_latin, addr_latin):
        s = ((n or "") + " " + (a or "")).strip()
        docs.append(s or " ")

    vect = HashingVectorizer(
        analyzer="char_wb",
        ngram_range=(1, 3),
        n_features=n_features,
        alternate_sign=False,
        norm="l2",
    )
    X = vect.transform(docs).astype(np.float32)
    log.info("  TF-IDF matrix: shape=%s, nnz=%d (%.1fs)",
             X.shape, X.nnz, time.time() - t0)

    # Save sparse matrix
    npz_path = base_dir / f"tfidf_country={country}.npz"
    from scipy.sparse import save_npz
    save_npz(npz_path, X)
    # Save meta (entity_id + source_flag)
    meta = cand_combined.select(["entity_id", "source_flag"]).clone()
    write_parquet(meta, base_dir / f"tfidf_country={country}_meta.parquet")
    # Save vectorizer config (so we can reload it)
    with open(base_dir / f"tfidf_country={country}_vec.pkl", "wb") as f:
        pickle.dump({
            "n_features": n_features,
            "analyzer": "char_wb",
            "ngram_range": (1, 3),
            "alternate_sign": False,
            "norm": "l2",
        }, f)
    log.info("[Stage 2] Wrote %s + meta + vec.pkl (%.1fs)",
             npz_path, time.time() - t0)
    del X, docs
    gc.collect()


# ---------------------------------------------------------------------------
# STAGE 3: MinHash LSH per country, per source column
# ---------------------------------------------------------------------------

def build_minhash_for_country(country: str, cand_combined: pl.DataFrame,
                              base_dir: Path):
    """Build MinHash LSH for name_latin AND addr_latin separately."""
    from datasketch import MinHash, MinHashLSH

    free_gb = _read_free_gb()
    # 128 perms × 6M records × ~8 bytes per LSH bucket index ≈ ~5 GB
    # 64 perms saves ~2.5 GB; prefer 64 on tight RAM, 128 otherwise.
    num_perm = 128 if free_gb > 6.0 else 64
    threshold = C.MINHASH_THRESHOLD

    for column in ("name_latin", "addr_latin"):
        log.info("[Stage 3] Building MinHash LSH on %s for country=%s (perms=%d, threshold=%.2f, free RAM=%.1f GB)",
                 column, country, num_perm, threshold, free_gb)
        t0 = time.time()
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        cands = cand_combined["entity_id"].to_list()
        texts = cand_combined[column].to_list()

        def shingles(text: str):
            t = text or ""
            if len(t) < 3:
                return [t] if t else []
            return [t[i:i + 3] for i in range(len(t) - 2)]

        for i in range(len(cands)):
            sh = shingles(texts[i])
            if not sh:
                continue
            m = MinHash(num_perm=num_perm)
            for s in sh:
                m.update(s.encode("utf-8"))
            lsh.insert(cands[i], m)
        # Save
        out_path = base_dir / f"minhash_country={country}_col={column}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump({
                "lsh": lsh,
                "num_perm": num_perm,
                "threshold": threshold,
                "column": column,
            }, f)
        log.info("  Wrote %s (%.1fs)", out_path, time.time() - t0)
        del lsh
        gc.collect()


# ---------------------------------------------------------------------------
# QUERY helpers
# ---------------------------------------------------------------------------

def _query_inverted_index_for_country(
    country: str,
    source: str,
    inv_path: Path,
    s1_rows: List[dict],
    out_path: Path,
):
    """Query the inverted indexes for all S1 records. Write partial candidates per (S1, cand)."""
    log.info("[Pass 1] Querying inverted indexes for %s/%s ...", country, source)
    t0 = time.time()
    df = read_parquet(inv_path, columns=["key_type", "key_value", "entity_id", "source_flag"])
    df = df.filter(pl.col("source_flag") == source)
    # Build per-key dictionaries on this source
    by_key: Dict[str, Dict[str, set]] = {kt: defaultdict(set) for kt in
                                           ("city", "state", "road", "house",
                                            "nfw", "afw", "alw",
                                            "city_state", "house_road")}
    for row in df.iter_rows(named=True):
        by_key[row["key_type"]][row["key_value"]].add(row["entity_id"])
    log.info("  Loaded %d key buckets (%.1fs)", len(by_key), time.time() - t0)

    # Key weights for ranking
    WEIGHTS = {
        "city": 1.0, "state": 1.0, "city_state": 1.0,
        "house": 0.95, "house_road": 1.0,
        "road": 0.9,
        "nfw": 0.75, "afw": 0.75, "alw": 0.6,
    }

    out_rows = []
    for s1 in s1_rows:
        city = (s1.get("addr_city") or "").lower().strip()
        state = (s1.get("addr_state") or "").lower().strip()
        road = (s1.get("addr_road") or "").lower().strip()
        house = (s1.get("addr_house_number") or "").lower().strip()
        nfw = first_word(s1.get("name_clean") or "").lower().strip()
        afw = first_word(s1.get("addr_latin") or "").lower().strip()
        alw = last_word(s1.get("addr_latin") or "").lower().strip()

        cands: Dict[str, float] = {}
        def put(cids, w):
            for cid in cids:
                if cands.get(cid, 0.0) < w:
                    cands[cid] = w
        if city:
            put(by_key["city"].get(city, set()), 1.0)
        if state:
            put(by_key["state"].get(state, set()), 1.0)
        if road:
            put(by_key["road"].get(road, set()), 0.9)
        if house:
            put(by_key["house"].get(house, set()), 0.95)
        if nfw:
            put(by_key["nfw"].get(nfw, set()), 0.75)
        if afw:
            put(by_key["afw"].get(afw, set()), 0.75)
        if alw:
            put(by_key["alw"].get(alw, set()), 0.6)
        if city and state:
            put(by_key["city_state"].get(f"{city}|{state}", set()), 1.0)
        if house and road:
            put(by_key["house_road"].get(f"{house}|{road}", set()), 1.0)
        # Score per (cand_id, source) — take max over keys
        for cid, sc in cands.items():
            out_rows.append({"s1_id": s1["entity_id"], "cand_id": cid, "src_idx_score": float(sc)})

    out_df = pl.from_dicts(out_rows)
    write_parquet(out_df, out_path)
    log.info("  Wrote partial inverted candidates: %s (%d rows, %.1fs)",
             out_path, len(out_df), time.time() - t0)
    del df, by_key, out_rows, out_df
    gc.collect()


def _query_tfidf_for_country(
    country: str,
    source: str,
    base_dir: Path,
    cand_combined: pl.DataFrame,
    s1_rows: List[dict],
    out_path: Path,
    topk: int,
):
    from scipy.sparse import load_npz
    from sklearn.preprocessing import normalize

    log.info("[Pass 2] TF-IDF top-%d for %s/%s ...", topk, country, source)
    t0 = time.time()
    X = load_npz(base_dir / f"tfidf_country={country}.npz").astype(np.float32)
    cand_meta = read_parquet(base_dir / f"tfidf_country={country}_meta.parquet")
    cand_ids = cand_meta["entity_id"].to_numpy()
    cand_sources = cand_meta["source_flag"].to_numpy()
    # Filter only candidates from this source
    src_mask = cand_sources == source
    X_src = X[src_mask]
    cand_ids_src = cand_ids[src_mask]
    log.info("  Loaded TF-IDF (%d src candidates, %.1fs)",
             X_src.shape[0], time.time() - t0)

    out_rows = []
    for s1 in s1_rows:
        name = s1.get("name_latin") or ""
        addr = s1.get("addr_latin") or ""
        q = f"{name} {addr}".strip() or " "
        # Manual hash to match the HashingVectorizer
        from sklearn.feature_extraction.text import HashingVectorizer
        with open(base_dir / f"tfidf_country={country}_vec.pkl", "rb") as f:
            cfg = pickle.load(f)
        vect = HashingVectorizer(
            analyzer=cfg["analyzer"],
            ngram_range=cfg["ngram_range"],
            n_features=cfg["n_features"],
            alternate_sign=cfg["alternate_sign"],
            norm=cfg["norm"],
        )
        qv = vect.transform([q]).astype(np.float32)
        qv = normalize(qv, norm="l2", copy=False)
        # Sparse-sparse dot: (1 × nnz) · (nnz × N) = (1 × N)
        scores = (qv @ X_src.T).toarray().ravel().astype(np.float32)
        # Top-k indices
        if topk >= len(scores):
            top_idx = np.argsort(-scores)
        else:
            top_idx = np.argpartition(-scores, -topk)[-topk:]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
        s1_id = s1["entity_id"]
        for i in top_idx:
            if scores[i] > 0:
                out_rows.append({
                    "s1_id": s1_id,
                    "cand_id": str(cand_ids_src[i]),
                    "tfidf_score": float(scores[i]),
                })
    out_df = pl.from_dicts(out_rows)
    write_parquet(out_df, out_path)
    log.info("  Wrote %s (%d rows, %.1fs)",
             out_path, len(out_df), time.time() - t0)
    del X, X_src, out_rows, out_df
    gc.collect()


def _query_minhash_for_country(
    country: str,
    source: str,
    column: str,
    base_dir: Path,
    cand_combined: pl.DataFrame,
    s1_rows: List[dict],
    out_path: Path,
):
    from datasketch import MinHash

    log.info("[Pass 3] MinHash query on %s for %s/%s ...", column, country, source)
    t0 = time.time()
    with open(base_dir / f"minhash_country={country}_col={column}.pkl", "rb") as f:
        data = pickle.load(f)
    lsh = data["lsh"]
    num_perm = data["num_perm"]
    cand_set = set(cand_combined.filter(pl.col("source_flag") == source)["entity_id"].to_list())

    out_rows = []
    for s1 in s1_rows:
        text = (s1.get(column) or "")
        if len(text) < 3:
            continue
        m = MinHash(num_perm=num_perm)
        for i in range(len(text) - 2):
            m.update(text[i:i+3].encode("utf-8"))
        results = lsh.query(m)
        s1_id = s1["entity_id"]
        for cid in results:
            if cid.startswith(f"{source}-"):
                out_rows.append({
                    "s1_id": s1_id,
                    "cand_id": cid,
                    f"{column[:3]}_mh_score": 0.7,
                })
    if out_rows:
        out_df = pl.from_dicts(out_rows)
        write_parquet(out_df, out_path)
        log.info("  Wrote %s (%d rows, %.1fs)", out_path, len(out_df), time.time() - t0)
    else:
        log.info("  No candidates from %s for %s/%s", column, country, source)


# ---------------------------------------------------------------------------
# Final cap + write
# ---------------------------------------------------------------------------

def _finalize_and_cap(
    country: str,
    source: str,
    base_dir: Path,
    cand_combined: pl.DataFrame,
    final_path: Path,
):
    """Read all partials, union, cap at top-100 by QRatio, write final."""
    from rapidfuzz import fuzz

    s1_id_col = "s1_id"
    cand_id_col = "cand_id"
    partial_paths = sorted(base_dir.glob(f"partial_country={country}_source={source}_*.parquet"))
    if not partial_paths:
        log.warning("No partials for %s/%s, skipping", country, source)
        return

    log.info("[Finalize] Unioning %d partials for %s/%s ...", len(partial_paths), country, source)
    parts = [read_parquet(p) for p in partial_paths]
    union = pl.concat(parts)
    del parts
    # Dedup: keep max score per (s1_id, cand_id)
    union = (
        union.group_by([s1_id_col, cand_id_col])
              .agg([
                  pl.col("src_idx_score").max().alias("src_idx_score"),
                  pl.col("tfidf_score").max().alias("tfidf_score"),
                  pl.col("name_mh_score").max().alias("name_mh_score"),
                  pl.col("addr_mh_score").max().alias("addr_mh_score"),
              ])
    )
    cand_lookup: Dict[str, str] = {}
    addr_lookup: Dict[str, str] = {}
    name_l_lookup: Dict[str, str] = {}
    addr_l_lookup: Dict[str, str] = {}
    for row in cand_combined.filter(pl.col("source_flag") == source).iter_rows(named=True):
        eid = row["entity_id"]
        cand_lookup[eid] = (row.get("name_clean") or "")
        addr_lookup[eid] = (row.get("addr_latin") or "")
        name_l_lookup[eid] = (row.get("name_latin") or "")
        addr_l_lookup[eid] = (row.get("addr_latin") or "")

    s1_df = read_parquet(C.ARTIFACTS_ROOT / "s1_norm_train.parquet")
    s1_country = s1_df.filter(pl.col("country") == country)
    s1_name_l = {r["entity_id"]: (r.get("name_latin") or "") for r in s1_country.iter_rows(named=True)}
    s1_addr_l = {r["entity_id"]: (r.get("addr_latin") or "") for r in s1_country.iter_rows(named=True)}

    s1_ids = list(s1_name_l.keys())
    union_dict: Dict[Tuple[str, str], dict] = {}
    for row in union.iter_rows(named=True):
        k = (row[s1_id_col], row[cand_id_col])
        d = union_dict.setdefault(k, {})
        if row.get("src_idx_score") is not None:
            d["src_idx_score"] = max(d.get("src_idx_score", 0.0), row["src_idx_score"] or 0.0)
        if row.get("tfidf_score") is not None:
            d["tfidf_score"] = max(d.get("tfidf_score", 0.0), row["tfidf_score"] or 0.0)
        if row.get("name_mh_score") is not None:
            d["name_mh_score"] = max(d.get("name_mh_score", 0.0), row["name_mh_score"] or 0.0)
        if row.get("addr_mh_score") is not None:
            d["addr_mh_score"] = max(d.get("addr_mh_score", 0.0), row["addr_mh_score"] or 0.0)

    out_rows = []
    cap = C.BLOCKING_TOPK_PER_S1
    for i, s1 in enumerate(s1_ids):
        s1_name = s1_name_l.get(s1, "")
        s1_addr = s1_addr_l.get(s1, "")
        scored = []
        for (s, cand), d in union_dict.items():
            if s != s1:
                continue
            cand_name = name_l_lookup.get(cand, "")
            cand_addr = addr_l_lookup.get(cand, "")
            qn = fuzz.QRatio(s1_name, cand_name)
            qa = fuzz.QRatio(s1_addr, cand_addr)
            score = sum(d.values()) / max(len(d), 1)
            scored.append((cand, qn, qa, score))
        scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
        scored = scored[:cap]
        for cand, qn, qa, sc in scored:
            out_rows.append({
                "s1_id": s1,
                "cand_id": cand,
                "score": float(sc),
                "qratio_name": int(qn),
                "qratio_addr": int(qa),
                "source_flag": source,
                "country": country,
            })

    out_df = pl.from_dicts(out_rows)
    write_parquet(out_df, final_path)
    log.info("[Finalize] Wrote final %s (%d rows, %d unique S1)",
             final_path, len(out_df), len(s1_ids))
    del union_dict, out_rows, out_df, union, cand_lookup, addr_lookup, name_l_lookup, addr_l_lookup
    gc.collect()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=logging.INFO,
    )

    s1 = read_parquet(C.ARTIFACTS_ROOT / "s1_norm_train.parquet")
    s2 = read_parquet(C.ARTIFACTS_ROOT / "s2_norm_train.parquet")
    s3 = read_parquet(C.ARTIFACTS_ROOT / "s3_norm_train.parquet")
    log.info("Loaded: s1=%d, s2=%d, s3=%d", len(s1), len(s2), len(s3))

    countries = [c for c in s1["country"].unique().to_list() if c]
    log.info("Countries: %s", countries)

    base_dir = C.ARTIFACTS_ROOT / "_blocking_idx"
    base_dir.mkdir(exist_ok=True)

    for country in countries:
        log.info("=" * 60)
        log.info("Country: %s  (free RAM=%.1f GB)", country, _read_free_gb())
        log.info("=" * 60)

        s1_c = s1.filter(pl.col("country") == country)
        s2_c = s2.filter(pl.col("country") == country)
        s3_c = s3.filter(pl.col("country") == country)
        cand_combined = pl.concat([s2_c, s3_c])
        log.info("S1=%d  S2=%d  S3=%d  combined=%d",
                 len(s1_c), len(s2_c), len(s3_c), len(cand_combined))

        # ====== BUILD PHASE (one index at a time, free RAM between) ======

        # Stage 1: Inverted indexes
        inv_path = base_dir / f"inv_country={country}.parquet"
        if not inv_path.exists():
            build_inverted_indexes_for_country(country, s2_c, s3_c, inv_path)
        else:
            log.info("[Stage 1] Skipping — %s already exists", inv_path)
        del s2_c, s3_c
        gc.collect()

        # Stage 2: TF-IDF sparse
        tfidf_path = base_dir / f"tfidf_country={country}.npz"
        if not tfidf_path.exists():
            build_tfidf_for_country(country, cand_combined, base_dir)
        else:
            log.info("[Stage 2] Skipping — %s already exists", tfidf_path)
        gc.collect()

        # Stage 3: MinHash LSH x2 (name + addr)
        for column in ("name_latin", "addr_latin"):
            mh_path = base_dir / f"minhash_country={country}_col={column}.pkl"
            if not mh_path.exists():
                build_minhash_for_country(country, cand_combined, base_dir)
            else:
                log.info("[Stage 3] Skipping — %s already exists", mh_path)
            gc.collect()

        # ====== QUERY PHASE: per (country, source) ======

        s1_rows = s1_c.to_dicts()

        for source in ("S2", "S3"):
            final_path = C.ARTIFACTS_ROOT / f"blocks_country={country}_source={source}.parquet"
            if final_path.exists():
                log.info("Skipping %s — final already exists", final_path)
                continue

            log.info("Blocking %s/%s: %d S1 queries (free RAM=%.1f GB)",
                     country, source, len(s1_rows), _read_free_gb())

            # Pass 1: inverted
            p1 = base_dir / f"partial_country={country}_source={source}_inv.parquet"
            if not p1.exists():
                _query_inverted_index_for_country(country, source, inv_path, s1_rows, p1)
            else:
                log.info("[Pass 1] skipping — %s exists", p1)
            gc.collect()

            # Pass 2: TF-IDF
            p2 = base_dir / f"partial_country={country}_source={source}_tfidf.parquet"
            if not p2.exists():
                _query_tfidf_for_country(
                    country, source, base_dir, cand_combined, s1_rows, p2, topk=C.BLOCKING_TOPK_PER_S1)
            else:
                log.info("[Pass 2] skipping — %s exists", p2)
            gc.collect()

            # Pass 3a: MinHash name
            p3a = base_dir / f"partial_country={country}_source={source}_name_mh.parquet"
            if not p3a.exists():
                _query_minhash_for_country(country, source, "name_latin", base_dir, cand_combined, s1_rows, p3a)
            else:
                log.info("[Pass 3a] skipping — %s exists", p3a)
            gc.collect()

            # Pass 3b: MinHash addr
            p3b = base_dir / f"partial_country={country}_source={source}_addr_mh.parquet"
            if not p3b.exists():
                _query_minhash_for_country(country, source, "addr_latin", base_dir, cand_combined, s1_rows, p3b)
            else:
                log.info("[Pass 3b] skipping — %s exists", p3b)
            gc.collect()

            # Final: union partials, cap, write
            _finalize_and_cap(country, source, base_dir, cand_combined, final_path)

        # Free country-level data
        del cand_combined, s1_c, s1_rows
        gc.collect()
        log.info("After %s free RAM=%.1f GB", country, _read_free_gb())

    return 0


if __name__ == "__main__":
    sys.exit(main())
