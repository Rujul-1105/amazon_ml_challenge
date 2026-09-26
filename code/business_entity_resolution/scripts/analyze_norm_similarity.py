"""Similarity analysis on the *normalized* parquet fields.

Compares:
  1. Raw-field similarity  (name_ratio, name_token_set_ratio, addr_ratio) on the
     raw `business_name` / `business_address` — same numbers as before.
  2. Normalized-field similarity on:
       name_clean / name_latin  (libpostal-tokened, lowercased, whitespace-sep)
       addr_clean               (libpostal-parsed, lowercased)
  3. Cheap exact-match signals on **structured address components** that
     libpostal extracted:
       addr_zip, addr_state, addr_city, addr_road, addr_house_number,
       addr_first_word

Outputs (artifacts/):
  norm_similarity_sample.tsv — 10,000 rows of pairs with both raw and
                              normalized similarity features.
  norm_similarity_insights.json — distributions + recommendations.
  norm_similarity_insights.md   — human write-up.

Run:
    python scripts/analyze_norm_similarity.py
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import polars as pl
from rapidfuzz import fuzz

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = PROJECT_ROOT / "dataset" / "train"
ARTIFACTS = PROJECT_ROOT / "artifacts"

S1_NORM = ARTIFACTS / "s1_norm_train.parquet"
S2_NORM = ARTIFACTS / "s2_norm_train.parquet"
S3_NORM = ARTIFACTS / "s3_norm_train.parquet"
GT_TSV = TRAIN_DIR / "train_ground_truth.tsv"

OUT_SAMPLE = ARTIFACTS / "norm_similarity_sample.tsv"
OUT_JSON = ARTIFACTS / "norm_similarity_insights.json"
OUT_MD = ARTIFACTS / "norm_similarity_insights.md"

SAMPLE_N = 10_000  # number of pairs to score in detail
SEED = 42
# ---------------------------------------------------------------------------


def _read_norm(path: Path, label: str) -> pl.DataFrame:
    print(f"[load] {label}: {path.name} ({path.stat().st_size / 1e9:.2f} GB)")
    t0 = time.time()
    df = pl.read_parquet(path)
    print(f"        → {df.height:,} rows in {time.time() - t0:.1f}s, {df.width} cols")
    return df


def _read_gt(path: Path) -> pl.DataFrame:
    print(f"[load] ground truth: {path.name}")
    t0 = time.time()
    df = pl.read_csv(
        path, separator="\t", encoding="utf8-lossy",
        schema_overrides={"source1_entity_id": pl.Utf8, "matched_entity_ids": pl.Utf8},
    )
    print(f"        → {df.height:,} rows in {time.time() - t0:.1f}s")
    return df


def build_pairs() -> pl.DataFrame:
    """Return a sample of (S1, matched) pairs with both normalized fields joined."""
    s1 = _read_norm(S1_NORM, "s1_norm_train")
    s2 = _read_norm(S2_NORM, "s2_norm_train")
    s3 = _read_norm(S3_NORM, "s3_norm_train")
    gt = _read_gt(GT_TSV)

    # Stamp source for S2/S3, concat
    s2 = s2.with_columns(pl.lit("S2").alias("matched_source"))
    s3 = s3.with_columns(pl.lit("S3").alias("matched_source"))
    lookup = pl.concat([s2, s3], how="vertical_relaxed").rename({
        "entity_id": "matched_entity_id",
        "country": "matched_country",
    })

    # Explode ground truth → one row per match pair
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
    print(f"        → {pairs.height:,} positive (S1, matched) pairs")

    print("[join] attaching normalized S1 metadata ...")
    s1_cols = ["entity_id", "country", "name_clean", "name_latin", "name_tokens",
               "name_missing", "addr_clean", "addr_latin",
               "addr_zip", "addr_state", "addr_city",
               "addr_first_word", "addr_last_word",
               "addr_house_number", "addr_road", "addr_unit", "addr_suburb",
               "addr_missing"]
    s1_renamed = s1.select([
        pl.col("entity_id").alias("source1_entity_id"),
        *[pl.col(c).alias(f"s1__{c}") for c in s1_cols if c != "entity_id"],
    ])
    pairs = pairs.join(s1_renamed, on="source1_entity_id", how="left")
    print(f"        → {pairs.height:,} after S1 join")

    print("[join] attaching normalized matched metadata ...")
    matched_cols = ["matched_entity_id", "matched_source", "matched_country",
                    "name_clean", "name_latin", "name_tokens", "name_missing",
                    "addr_clean", "addr_latin",
                    "addr_zip", "addr_state", "addr_city",
                    "addr_first_word", "addr_last_word",
                    "addr_house_number", "addr_road", "addr_unit", "addr_suburb",
                    "addr_missing"]
    pairs = pairs.join(
        lookup.select([pl.col(c).alias(f"m__{c}" if c != "matched_entity_id" else "matched_entity_id")
                       for c in matched_cols]),
        on="matched_entity_id", how="left",
    )
    print(f"        → {pairs.height:,} after matched join")

    # Sample for similarity computation (full join is 7.6M rows; expensive to fuzzy score all)
    sample = pairs.sample(n=min(SAMPLE_N, pairs.height), seed=SEED, with_replacement=False)
    print(f"[sample] {len(sample):,} pairs for similarity scoring")
    return sample


# ---------------------------------------------------------------------------
# Similarity scoring (Python loop on a 10k sample — fast enough with rapidfuzz)
# ---------------------------------------------------------------------------

def _s(x):
    """Coerce nullable/NaN string to ''."""
    if x is None:
        return ""
    if isinstance(x, float) and x != x:  # NaN
        return ""
    return str(x)


def score_pair(r: dict) -> dict:
    s1_name = _s(r.get("s1__name_clean"))
    m_name = _s(r.get("m__name_clean"))
    s1_name_lat = _s(r.get("s1__name_latin"))
    m_name_lat = _s(r.get("m__name_latin"))
    s1_addr = _s(r.get("s1__addr_clean"))
    m_addr = _s(r.get("m__addr_clean"))

    out = {}

    # ---- Name similarities on normalized fields ----
    if s1_name and m_name:
        out["norm_name_ratio"] = fuzz.ratio(s1_name, m_name)
        out["norm_name_token_set_ratio"] = fuzz.token_set_ratio(s1_name, m_name)
        out["norm_name_token_sort_ratio"] = fuzz.token_sort_ratio(s1_name, m_name)
        out["norm_name_partial_ratio"] = fuzz.partial_ratio(s1_name, m_name)
    else:
        out["norm_name_ratio"] = -1
        out["norm_name_token_set_ratio"] = -1
        out["norm_name_token_sort_ratio"] = -1
        out["norm_name_partial_ratio"] = -1

    if s1_name_lat and m_name_lat:
        out["norm_name_latin_ratio"] = fuzz.ratio(s1_name_lat, m_name_lat)
        out["norm_name_latin_token_set"] = fuzz.token_set_ratio(s1_name_lat, m_name_lat)
    else:
        out["norm_name_latin_ratio"] = -1
        out["norm_name_latin_token_set"] = -1

    # ---- Address similarity on normalized fields ----
    if s1_addr and m_addr:
        out["norm_addr_ratio"] = fuzz.ratio(s1_addr, m_addr)
        out["norm_addr_token_set_ratio"] = fuzz.token_set_ratio(s1_addr, m_addr)
        out["norm_addr_token_sort_ratio"] = fuzz.token_sort_ratio(s1_addr, m_addr)
    else:
        out["norm_addr_ratio"] = -1
        out["norm_addr_token_set_ratio"] = -1
        out["norm_addr_token_sort_ratio"] = -1

    # ---- Structured address component equalities (the cheap exact-match rescue signals) ----
    for f in ["addr_zip", "addr_state", "addr_city", "addr_first_word",
              "addr_last_word", "addr_house_number", "addr_road", "addr_unit", "addr_suburb"]:
        sv = _s(r.get(f"s1__{f}")).strip().lower()
        mv = _s(r.get(f"m__{f}")).strip().lower()
        out[f"{f}_eq"] = int(bool(sv) and sv == mv)
        out[f"{f}_nonempty"] = int(bool(sv) and bool(mv))

    # ---- Missing-field flags ----
    out["s1_name_missing"] = int(bool(r.get("s1__name_missing")))
    out["m_name_missing"] = int(bool(r.get("m__name_missing")))
    out["s1_addr_missing"] = int(bool(r.get("s1__addr_missing")))
    out["m_addr_missing"] = int(bool(r.get("m__addr_missing")))

    return out


def compute_insights(pdf) -> dict:
    cols_of_interest = [
        "norm_name_ratio", "norm_name_token_set_ratio", "norm_name_token_sort_ratio",
        "norm_name_partial_ratio", "norm_name_latin_ratio", "norm_name_latin_token_set",
        "norm_addr_ratio", "norm_addr_token_set_ratio", "norm_addr_token_sort_ratio",
    ]
    struct_cols = [
        "addr_zip_eq", "addr_state_eq", "addr_city_eq", "addr_first_word_eq",
        "addr_last_word_eq", "addr_house_number_eq", "addr_road_eq",
        "addr_unit_eq", "addr_suburb_eq",
    ]

    insights = {}

    # 1) Distributions of continuous similarity scores
    def stats(series, name):
        vals = [v for v in series if v is not None and v >= 0]
        if not vals:
            return {"n": 0}
        s = sorted(vals)
        return {
            "n": len(vals),
            "mean": round(sum(vals) / len(vals), 2),
            "median": round(s[len(s) // 2], 2),
            "p25": round(s[len(s) // 4], 2),
            "p75": round(s[3 * len(s) // 4], 2),
            "p90": round(s[int(0.9 * len(s))], 2),
            "share_>=85": round(sum(1 for v in vals if v >= 85) / len(vals), 4),
            "share_>=95": round(sum(1 for v in vals if v >= 95) / len(vals), 4),
        }

    for c in cols_of_interest:
        insights[c] = stats(pdf[c].tolist(), c)

    # 2) Exact-match rates on structured address components (the rescue signals)
    for c in struct_cols:
        n = pdf[c].sum()
        total = len(pdf)
        insights[f"{c}_rate"] = round(float(n) / total, 4)

    # 3) How many pairs clear *both* a strong-name and strong-address threshold?
    name_strong = pdf["norm_name_token_set_ratio"] >= 90
    addr_strong = pdf["norm_addr_token_set_ratio"] >= 80
    insights["share_name_strong_only"] = round(float(name_strong.mean()), 4)
    insights["share_addr_strong_only"] = round(float(addr_strong.mean()), 4)
    insights["share_both_strong"] = round(float((name_strong & addr_strong).mean()), 4)
    insights["share_either_strong"] = round(float((name_strong | addr_strong).mean()), 4)

    # 4) Same-address-component rescue: among pairs whose names are *weak*,
    # how often does at least one structured address component match?
    name_weak = pdf["norm_name_token_set_ratio"] < 60
    insights["name_weak_count"] = int(name_weak.sum())
    if name_weak.any():
        any_addr_eq = pdf.loc[name_weak, struct_cols].any(axis=1)
        insights["name_weak_but_any_addr_eq"] = int(any_addr_eq.sum())
        insights["name_weak_but_addr_zip_eq"] = int(pdf.loc[name_weak, "addr_zip_eq"].sum())
        insights["name_weak_but_addr_city_eq"] = int(pdf.loc[name_weak, "addr_city_eq"].sum())
        insights["name_weak_but_addr_state_eq"] = int(pdf.loc[name_weak, "addr_state_eq"].sum())
        insights["name_weak_but_addr_road_eq"] = int(pdf.loc[name_weak, "addr_road_eq"].sum())
        insights["name_weak_but_addr_house_eq"] = int(pdf.loc[name_weak, "addr_house_number_eq"].sum())

    # 5) Most-discriminative single feature: each structured eq's lift over baseline
    base_match_rate = float((pdf["norm_name_token_set_ratio"] >= 85).mean())
    insights["baseline_name_strong_rate"] = round(base_match_rate, 4)
    for c in struct_cols:
        col_rate = float(pdf[c].mean())
        # Lift = P(strong_name | component_eq) / P(strong_name)
        # Approximate via simple rates
        cond = pdf[c] == 1
        p_strong_given = float((pdf.loc[cond, "norm_name_token_set_ratio"] >= 85).mean()) if cond.any() else 0
        insights[f"lift_{c}_on_name_strong"] = round(p_strong_given / max(base_match_rate, 1e-6), 2)

    return insights


def main():
    t0 = time.time()
    sample = build_pairs()
    pdf = sample.to_pandas()

    print("[score] computing per-pair similarity ...")
    rows = []
    for i, r in enumerate(pdf.to_dict("records")):
        rows.append(score_pair(r))
        if (i + 1) % 2000 == 0:
            print(f"        {i+1:,}/{len(pdf):,}")
    scored = pl.from_dicts(rows)

    # Concat scored features back to the sample frame for the TSV
    out = pl.concat([sample, scored], how="horizontal")
    print(f"[write] {OUT_SAMPLE} ({out.height:,} rows, {out.width} cols)")
    out.write_csv(OUT_SAMPLE, separator="\t", quote_style="never")

    print("[insights] summarising ...")
    import pandas as pd
    scored_pdf = pd.DataFrame(rows)
    combined = pd.concat([pdf.reset_index(drop=True), scored_pdf], axis=1)
    insights = compute_insights(combined)
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(insights, f, indent=2, ensure_ascii=False)
    print(f"[write] {OUT_JSON}")
    print(f"[done] total wall-clock: {time.time() - t0:.1f}s")
    return insights


if __name__ == "__main__":
    main()