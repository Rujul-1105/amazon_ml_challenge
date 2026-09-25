"""Centralized configuration for the Business Entity Resolution pipeline.

Paths, hyper-parameters, country lists, and feature flags live here so every
module reads from a single source of truth.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Development policy
# ---------------------------------------------------------------------------

# All development (EDA, normalization, blocking, feature engineering, training,
# validation, threshold tuning, error analysis) MUST operate on training files
# only. The test files in dataset/test/ are not to be read, inspected, or
# processed locally. The test set is touched ONCE, at the very end, by the
# inference script `predict.py` (or pipeline.py --mode submit) to generate
# output TSVs for the leaderboard submission.
#
# Set TRAIN_ONLY = False ONLY when running the final inference / submission
# generation step. Never set it to False during development, validation, or
# model selection — that would defeat the purpose of having a held-out test set.
TRAIN_ONLY: bool = True


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path("/home/rujul/projects/a/amzn_ml").resolve()
DATA_ROOT: Path = PROJECT_ROOT / "data_set" / "student_resource"
DATASET_ROOT: Path = DATA_ROOT / "dataset"

# Final output goes to BOTH the project root output/ (for the official zip)
# and code/business_entity_resolution/output/ (so the code package is
# self-contained per the submission rules).
OUTPUT_ROOT: Path = PROJECT_ROOT / "output"
PKG_OUTPUT_ROOT: Path = PROJECT_ROOT / "code" / "business_entity_resolution" / "output"

# Code package itself
PKG_ROOT: Path = PROJECT_ROOT / "code" / "business_entity_resolution"
SRC_ROOT: Path = PKG_ROOT / "src"
ARTIFACTS_ROOT: Path = PKG_ROOT / "artifacts"
NOTEBOOKS_ROOT: Path = PKG_ROOT / "notebooks"
SCRIPTS_ROOT: Path = PKG_ROOT / "scripts"

# Source files
TRAIN_DIR: Path = DATASET_ROOT / "train"
TEST_DIR: Path = DATASET_ROOT / "test"

TRAIN_SOURCE1: Path = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2: Path = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3: Path = TRAIN_DIR / "train_source3.tsv"
TRAIN_GT: Path = TRAIN_DIR / "train_ground_truth.tsv"

TEST_SOURCE1: Path = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2: Path = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3: Path = TEST_DIR / "test_source3.tsv"

# Helper scripts
VALIDATOR: Path = DATA_ROOT / "utils" / "validate_submission.py"
DOC_TEMPLATE: Path = DATA_ROOT / "Documentation_template.md"

# ---------------------------------------------------------------------------
# Countries
# ---------------------------------------------------------------------------

# Training countries seen in ground truth
TRAIN_COUNTRIES: List[str] = ["US", "India", "France"]  # open-set
# Test will additionally include France — never hard-code to {US, India}
# Treat country as an open-set string label.

# ---------------------------------------------------------------------------
# Validation / train/val split
# ---------------------------------------------------------------------------

VAL_FRAC: float = 0.10  # fraction of S1 IDs held out for validation
SEED: int = 42

# ---------------------------------------------------------------------------
# Blocking
# ---------------------------------------------------------------------------

BLOCKING_TOPK_PER_S1: int = 100  # cap on candidates per S1
BLOCKING_TARGET_RECALL: float = 0.92

# TF-IDF parameters (per country)
TFIDF_NGRAM_RANGE: Tuple[int, int] = (1, 3)
TFIDF_MIN_DF: int = 2
TFIDF_MAX_DF: float = 0.95
TFIDF_MAX_FEATURES: int = 1 << 21

# MinHash LSH
MINHASH_THRESHOLD: float = 0.7
MINHASH_PERM: int = 128
MINHASH_SHINGLE_K: int = 3  # char n-gram size

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class LightGBMParams:
    objective: str = "binary"
    metric: str = "binary_logloss"
    learning_rate: float = 0.05
    num_leaves: int = 127
    max_depth: int = -1
    min_data_in_leaf: int = 200
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.9
    bagging_freq: int = 1
    scale_pos_weight: float = 5.0
    verbose: int = -1
    n_jobs: int = 18
    seed: int = SEED

# ---------------------------------------------------------------------------
# Per-country, per-source default thresholds (tune on val)
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS: Dict[Tuple[str, str], float] = {
    ("US", "S2"): 0.55,
    ("US", "S3"): 0.55,
    ("India", "S2"): 0.50,
    ("India", "S3"): 0.55,
    ("France", "S2"): 0.65,
    ("France", "S3"): 0.65,
}

# ---------------------------------------------------------------------------
# Singleton detector
# ---------------------------------------------------------------------------

SINGLETON_PROB_THRESHOLD: float = 0.5  # if P(singleton) > this AND no candidate exceeds τ+0.15, output empty

# ---------------------------------------------------------------------------
# Graph refinement
# ---------------------------------------------------------------------------

GRAPH_TRIANGLE_MIN_PROB: float = 0.85
GRAPH_DAMPING: float = 0.9
GRAPH_MAX_NEW_EDGE_RATE: float = 0.03  # if closure adds >3% new edges, leakage

# ---------------------------------------------------------------------------
# Cascade (high-precision auto-commit)
# ---------------------------------------------------------------------------

CASCADE_NAME_LEV_THRESHOLD: float = 0.92
CASCADE_NAME_JARO_THRESHOLD: float = 0.96
CASCADE_NAME_TOKEN_SET_THRESHOLD: float = 0.95
CASCADE_NAME_JACCARD_THRESHOLD: float = 0.7

# ---------------------------------------------------------------------------
# Hard-negative mining
# ---------------------------------------------------------------------------

HARD_NEG_RATIO: float = 0.30  # 30% of negatives are hard (70% random)
NEG_POS_RATIO: int = 5       # 5 negatives per positive

# ---------------------------------------------------------------------------
# Filesystem
# ---------------------------------------------------------------------------

def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


ensure_dirs(OUTPUT_ROOT, PKG_OUTPUT_ROOT, ARTIFACTS_ROOT, NOTEBOOKS_ROOT, SCRIPTS_ROOT)
