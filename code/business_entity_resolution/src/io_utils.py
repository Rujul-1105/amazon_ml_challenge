"""Robust TSV loaders and chunk helpers for the ER pipeline.

All TSV readers use `encoding="utf8-lossy"` so a stray byte (e.g. an ampersand
in a French business name mixed with Latin-1) never aborts a 5M-row load.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterator, Optional

import polars as pl

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema definitions
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


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_source(path: Path | str) -> pl.DataFrame:
    """Load one *_source{1,2,3}.tsv file as a polars DataFrame."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing source file: {path}")
    log.info("Loading %s ...", path)
    df = pl.read_csv(
        path,
        separator="\t",
        encoding="utf8-lossy",
        schema_overrides=SOURCE_SCHEMA,
        null_values=["", "NA", "null", "NaN"],
        ignore_errors=False,
        low_memory=False,
    )
    n = len(df)
    u = df["entity_id"].n_unique()
    log.info("  %d rows, %d unique entity_ids", n, u)
    if n != u:
        log.warning("  Duplicate entity_ids detected: %d unique vs %d rows", u, n)
    return df


def load_ground_truth(path: Path | str) -> pl.DataFrame:
    """Load train_ground_truth.tsv. Empty `matched_entity_ids` -> empty string."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing ground-truth file: {path}")
    log.info("Loading ground truth %s ...", path)
    df = pl.read_csv(
        path,
        separator="\t",
        encoding="utf8-lossy",
        schema_overrides=GT_SCHEMA,
        null_values=["", "NA", "null"],
        ignore_errors=False,
        low_memory=False,
    )
    n = len(df)
    log.info("  %d ground-truth rows", n)
    return df


def explode_ground_truth(gt: pl.DataFrame) -> pl.DataFrame:
    """Return one row per (s1_id, matched_id) pair.

    Singletons become rows with matched_id = None (filtered out by callers
    that need only positive pairs; callers that need the singleton signal
    should keep them).
    """
    return (
        gt
        .with_columns(
            pl.col("matched_entity_ids")
              .str.split(",")
              .alias("matched_list")
        )
        .explode("matched_list")
        .rename({"matched_list": "matched_id"})
        .with_columns(
            pl.col("matched_id").str.strip_chars().alias("matched_id")
        )
        .filter(
            pl.col("matched_id").is_not_null()
            & (pl.col("matched_id") != "")
        )
        .select("source1_entity_id", "matched_id")
    )


def iter_chunks(df: pl.DataFrame, chunk_size: int) -> Iterator[pl.DataFrame]:
    """Iterate over a polars DataFrame in fixed-size row chunks."""
    n = len(df)
    for start in range(0, n, chunk_size):
        yield df[start : start + chunk_size]


def write_parquet(df: pl.DataFrame, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path, compression="zstd", compression_level=3)
    log.info("Wrote %s (%d rows)", path, len(df))


def read_parquet(path: Path | str, columns: Optional[list[str]] = None) -> pl.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing parquet: {path}")
    return pl.read_parquet(path, columns=columns)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def file_size_mb(path: Path | str) -> float:
    return os.path.getsize(path) / (1024 * 1024)
