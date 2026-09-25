"""Exploratory data analysis on the 6 TSV files.

Outputs:
    - artifacts/eda_report.md         (markdown summary)
    - artifacts/eda_counts.csv        (row/unique counts per file)
    - artifacts/eda_country.csv       (country distributions)
    - artifacts/eda_missing.csv       (missing-value rates)

Usage:
    python3 -m src.eda
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import polars as pl
import regex

from . import config as C
from .io_utils import file_size_mb, load_ground_truth, load_source

log = logging.getLogger(__name__)

DEVANAGARI_RE = regex.compile(r"[ऀ-ॿ]+")
LATIN_RE = regex.compile(r"[A-Za-z]+")


# ---------------------------------------------------------------------------
# Per-file EDA
# ---------------------------------------------------------------------------

def eda_source(df: pl.DataFrame, name: str) -> dict:
    """Compute summary stats for one *_source{1,2,3}.tsv."""
    n = len(df)
    n_unique = df["entity_id"].n_unique()
    out = {
        "file": name,
        "rows": n,
        "unique_entity_ids": n_unique,
        "duplicate_ids": n - n_unique,
        "name_missing": int(df["business_name"].is_null().sum() + (df["business_name"] == "").sum()),
        "addr_missing": int(df["business_address"].is_null().sum() + (df["business_address"] == "").sum()),
        "country_missing": int(df["country"].is_null().sum() + (df["country"] == "").sum()),
        "name_mean_len": float(df["business_name"].str.len_chars().mean() or 0),
        "addr_mean_len": float(df["business_address"].str.len_chars().mean() or 0),
    }
    return out


def country_distribution(df: pl.DataFrame, name: str) -> pl.DataFrame:
    return (
        df.group_by("country")
          .agg(pl.len().alias("n"))
          .with_columns((pl.col("n") / pl.col("n").sum()).alias("fraction"))
          .sort("n", descending=True)
          .with_columns(pl.lit(name).alias("file"))
    )


def sample_records(df: pl.DataFrame, k: int = 5) -> pl.DataFrame:
    return df.sample(k, seed=C.SEED)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def write_markdown_report(
    counts: list[dict],
    countries: list[pl.DataFrame],
    out_path: Path,
) -> None:
    lines: list[str] = ["# EDA Report — Amazon ML Challenge 2026\n"]

    lines.append("## 1. File-level summary\n")
    lines.append("| File | Rows | Unique IDs | Dup IDs | Name missing | Addr missing | Name mean len | Addr mean len |")
    lines.append("|------|------|-----------:|--------:|-------------:|-------------:|--------------:|--------------:|")
    for r in counts:
        lines.append(
            f"| {r['file']} | {r['rows']:,} | {r['unique_entity_ids']:,} | "
            f"{r['duplicate_ids']:,} | {r['name_missing']:,} | {r['addr_missing']:,} | "
            f"{r['name_mean_len']:.1f} | {r['addr_mean_len']:.1f} |"
        )

    lines.append("\n## 2. Country distribution per file\n")
    for c in countries:
        lines.append(f"### {c['file'][0]}\n")
        lines.append("| Country | Count | Fraction |")
        lines.append("|---------|------:|---------:|")
        for row in c.iter_rows(named=True):
            lines.append(f"| {row['country']} | {row['n']:,} | {row['fraction']*100:.2f}% |")
        lines.append("")

    lines.append("## 3. Ground truth stats\n")
    gt = load_ground_truth(C.TRAIN_GT)
    n_total = len(gt)
    n_singletons = int((gt["matched_entity_ids"].is_null() | (gt["matched_entity_ids"] == "")).sum())
    n_with_matches = n_total - n_singletons
    lines.append(f"- Total S1 entities: {n_total:,}")
    lines.append(f"- Singletons (no matches): {n_singletons:,} ({100*n_singletons/n_total:.2f}%)")
    lines.append(f"- With matches: {n_with_matches:,}")

    # Match-count distribution
    n_match_counts = (
        gt.with_columns(
            pl.when(pl.col("matched_entity_ids").is_null() | (pl.col("matched_entity_ids") == ""))
              .then(0)
              .otherwise(pl.col("matched_entity_ids").str.count_matches(",") + 1)
              .alias("n_matches")
        )
        .group_by("n_matches")
        .agg(pl.len().alias("count"))
        .sort("n_matches")
    )
    lines.append("\n### Match-count distribution\n")
    lines.append("| # matches | # S1 entities |")
    lines.append("|----------:|--------------:|")
    for row in n_match_counts.iter_rows(named=True):
        lines.append(f"| {row['n_matches']} | {row['count']:,} |")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote EDA markdown: %s", out_path)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=logging.INFO,
    )

    files = [
        ("train_source1", C.TRAIN_SOURCE1),
        ("train_source2", C.TRAIN_SOURCE2),
        ("train_source3", C.TRAIN_SOURCE3),
        ("test_source1",  C.TEST_SOURCE1),
        ("test_source2",  C.TEST_SOURCE2),
        ("test_source3",  C.TEST_SOURCE3),
    ]

    counts: list[dict] = []
    countries: list[pl.DataFrame] = []
    all_dfs: dict[str, pl.DataFrame] = {}

    for name, path in files:
        log.info("==== %s (%.0f MB) ====", name, file_size_mb(path))
        df = load_source(path)
        counts.append(eda_source(df, name))
        countries.append(country_distribution(df, name))
        all_dfs[name] = df

        # Spot-check sample
        log.info("Random sample of 5 rows from %s:", name)
        for row in sample_records(df, 5).iter_rows(named=True):
            log.info("  %s | %r | %r | %r",
                     row["entity_id"],
                     (row["business_name"] or "")[:60],
                     (row["business_address"] or "")[:60],
                     row["country"])

    # Persist artifacts
    counts_df = pl.DataFrame(counts)
    counts_df.write_csv(C.ARTIFACTS_ROOT / "eda_counts.csv")
    log.info("Wrote %s", C.ARTIFACTS_ROOT / "eda_counts.csv")

    countries_df = pl.concat(countries, how="vertical")
    countries_df.write_csv(C.ARTIFACTS_ROOT / "eda_country.csv")
    log.info("Wrote %s", C.ARTIFACTS_ROOT / "eda_country.csv")

    # Missing rate
    miss = []
    for name, df in all_dfs.items():
        for col in ("business_name", "business_address", "country"):
            n_missing = int((df[col].is_null() | (df[col] == "")).sum())
            miss.append({"file": name, "column": col,
                         "missing_count": n_missing,
                         "missing_rate": n_missing / len(df)})
    pl.DataFrame(miss).write_csv(C.ARTIFACTS_ROOT / "eda_missing.csv")
    log.info("Wrote %s", C.ARTIFACTS_ROOT / "eda_missing.csv")

    write_markdown_report(counts, countries, C.ARTIFACTS_ROOT / "eda_report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
