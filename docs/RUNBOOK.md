# RUNBOOK

**Authoritative step-by-step commands.** Use this on every new machine
or new Claude session. Read `docs/STATUS.md` first — it lists current
state, hard-fail bugs to avoid, and the v3 blocker design rationale.

## Table of contents

1. First-time setup (one time per machine)
2. Each new Claude session — start with these reads
3. Phase C v3 — blocking (S2/S3 separately; run them on different machines if you want)
4. Phase C2 — combine S2+S3 and validate recall
5. Phase E — LightGBM training (needs ≥32 GiB RAM)
6. Phase F — inference (test blocker cached once)
7. Phase G + H — graph refinement, packaging

---

## 1. First-time setup (one time per machine)

**a. Python packages** (only `pip install` step needed; libpostal is already
installed on the AWS box the user has been using):

```bash
python3 -m pip install --user polars pandas numpy scikit-learn \
    lightgbm faiss-cpu networkx rapidfuzz datasketch pyarrow \
    unidecode regex indic-transliteration joblib tqdm
```

**b. Smoke check** before any blocking/training run:

```bash
python -c "import polars, numpy, lightgbm, rapidfuzz; print('OK')"
ls code/business_entity_resolution/artifacts/s{1,2,3}_norm_train.parquet
# All three files must exist. If missing, re-run Phase B (libpostal normalize).
```

---

## 2. Each new Claude session — mandatory first reads

Before doing **anything**, the next Claude session MUST read these three files in this order:

1. `CLAUDE.md` (at the project root) — what we're building, current phase status, hardware realities.
2. `docs/STATUS.md` — per-phase detailed state, **v3 blocker design spec, hard-fail bugs, expected recalls**.
3. `docs/RUNBOOK.md` (this file) — concrete commands.

Then verify what's already done:

```bash
ls code/business_entity_resolution/artifacts/
# Expect: s{1,2,3}_norm_train.parquet (Phase B output)
# Expect: block_S2_features.parquet OR NOT (depending on whether v3 S2 has run)
# If block_features parquets present for both S2 and S3 → Phase C done, go to Phase E.
# If only one direction done → run the missing one.
```

Then check disk and RAM:

```bash
df -h code/business_entity_resolution/artifacts/  # need ~30 GB free
free -h                                          # need 8 GB minimum for blocking
```

---

## 3. Phase C v3 — blocking

**This script already exists at `code/business_entity_resolution/scripts/block_features.py`**.
If it doesn't (e.g., fresh repo clone), copy from the version committed to git.

### 3a. S2-only run (on this machine)

```bash
cd code/business_entity_resolution

# Smoke test on 5% slice, max 3 chunks. Validates code path & RAM.
python scripts/block_features.py \
    --candidate-source S2 --dry-run --max-chunks 3
# Expect: ~5 min wall-clock, peak RAM <4 GB, NO final parquet written.

# Full S2 run
python scripts/block_features.py \
    --candidate-source S2
# Writes: artifacts/block_S2_features.parquet (~50M rows × 36 cols, ~3 GB on disk).
# Wall-clock on AWS t3.medium: ~50-80 min. On 8-core box: ~25-40 min.
```

### 3b. S3 run (on the SAME machine after S2 finishes, OR a different machine)

```bash
python scripts/block_features.py \
    --candidate-source S3
# Writes: artifacts/block_S3_features.parquet (similar size).
```

### 3c. Test blocker caching (Phase F prep, run ONLY when needed)

```bash
# Test normalization MUST be re-run first because Phase B only ran on train.
# Set TRAIN_ONLY=False at the top of src/config.py (one-time change).
# Then run normalize_submit_files() in src/normalize.py to produce
# artifacts/s{1,2,3}_norm_test.parquet.
python -m src.normalize  # with TRAIN_ONLY=False — one-time cost ~10 min on AWS.

# Block on test data with the SAME v3 design.
python scripts/block_features.py \
    --candidate-source S2 --suffix S2_TEST
python scripts/block_features.py \
    --candidate-source S3 --suffix S3_TEST
# Writes artifacts/block_S{2,3}_TEST_features.parquet (each ~1.5 GB).
# Wall-clock each: ~30 min on AWS t3.medium.
# IMPORTANT: do this ONCE and cache. Subsequent model iterations just
# load these parquets in `predict.py` — no re-blocking.
```

### 3d. CLI flags reference

| Flag | Default | Purpose |
| --- | --- | --- |
| `--candidate-source` | (required) | `S2`, `S3`, or `S3_TEST` etc. (any string ending the output suffix) |
| `--top-k` | `50` | Per-S1 cap (was 25 in v2; do NOT lower without checking recall) |
| `--chunk-size` | `10000` | S1 rows per chunk (raise to lower RAM; lower to fit tighter boxes) |
| `--bucket-cap` | `500` | Max S2 ids per structural-key bucket |
| `--sn-window` | `50` | Sorted-token neighborhood window ±W |
| `--trigram-cap` | N/A | (v3 has no char trigrams; flag ignored or removed) |
| `--max-chunks` | `None` | Run only the first N chunks (for testing) |
| `--dry-run` | `False` | Process a 5% slice and skip final concat |
| `--suffix` | (auto) | Output suffix; `--suffix S2_TEST` writes `block_S2_TEST_features.parquet` |
| `--top-k-tfidf` | (n/a) | v3 doesn't have a separate "tfidf" pre-cap |

### 3e. RAM diagnosis

If you OOM:

- Lower `--chunk-size` to 5000 → halves working-set RAM, ~2× slower per chunk.
- Lower `--bucket-cap` to 200 → halves inverted-index RAM.
- Lower `--sn-window` to 25 → halves sorted-neighborhood work.

---

## 4. Phase C2 — combine S2+S3 + validate recall

### 4a. Combine S2 + S3 blocked parquets

```bash
cd code/business_entity_resolution
python scripts/combine_block_features.py
# Writes: artifacts/block_features.parquet (~110 M pairs × 36 cols, ~3 GB).
```

### 4b. Validate recall (mandatory sanity check)

```bash
python scripts/validate_block_recall.py \
    --blocks artifacts/block_S2_features.parquet artifacts/block_S3_features.parquet
# (or pass --blocks artifacts/block_features.parquet after combine)
# Expect: overall recall ≥ 0.85. Exit code 0 if recall target met, else 2.
```

If recall < 0.80, do NOT proceed to Phase E. Debug:

1. Print per-bucket recall (`U.S.|S2`, `U.S.|S3`, `India|S2`, `India|S3`).
2. If `U.S.|S2` recall is high but `India|S2` low → sorted-token neighborhood may need a bigger window for transliterated names; consider raising `--sn-window` to 100.
3. If both axes are uniformly low → quality-tier filter is too strict; lower floor B from `n_tokens_shared ≥ 2` to `≥ 1`.
4. Inspect `artifacts/match_insights.md` and `artifacts/norm_similarity_insights.md` for the data-driven thresholds.

---

## 5. Phase E — LightGBM training

**Hardware target: ≥32 GiB RAM / 16 cores / 50 GB SSD.** Do **not** try to train on AWS t3.medium (8 GiB) — it will OOM.

### 5a. Prepare training data

```bash
cd code/business_entity_resolution
python scripts/prepare_training_data.py
# Reads: block_features.parquet (~110 M rows) + dataset/train/train_ground_truth.tsv.
# Writes: artifacts/training_data.parquet (~110 M rows × 37 cols, ~3 GB).
# Wall-clock: ~5-10 min.
```

### 5b. Train LightGBM

```bash
python scripts/train_classifier.py
# 3-stage search:
#   1) Coarse grid (manual, 20 M-row sample)         ~30 min
#   2) Optuna refinement (TPE, 30 trials)             ~2 h
#   3) Final fit at best params (50-80 M rows)          ~1.5-2 h
# Writes: artifacts/lgbm_classifier.txt.
# Wall-clock total on 16-core / 32 GB: ~4-5 h.
```

### 5c. Tune per-(country, source) thresholds

```bash
python scripts/threshold.py
# Sweeps val set; saves artifacts/per_country_source_threshold.json.
# Expected output: {"US|S2": ~0.45, "US|S3": ~0.50, "India|S2": ~0.40, "India|S3": ~0.45}.
```

### 5d. Train singleton detector (separate small LightGBM)

```bash
python scripts/singleton_detector.py
# Aggregates per-S1 features (max P, mean P, count, count > 0.5).
# Trains separate binary classifier (is_singleton).
# Writes: artifacts/singleton_classifier.txt.
```

---

## 6. Phase F — inference on test data

**Test blocker cached** (see §3c). Iterate cheaply:

```bash
cd code/business_entity_resolution

# ONE-TIME-SETUP (if not already done):
#   - TRAIN_ONLY=False in src/config.py
#   - artifacts/s{1,2,3}_norm_test.parquet exist
#   - artifacts/block_S{2,3}_TEST_features.parquet exist

# Apply model + threshold + singleton to test data:
python scripts/predict.py
# Reads: block_S{2,3}_TEST_features.parquet + lgbm_classifier.txt + threshold JSON.
# Writes: output/matching_results.tsv + output/candidate_pairs.tsv.
# Wall-clock: 5-10 min per iteration.

# Validate submission format:
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test --check-ids
# Exit 0 = submission format OK.
```

**Key speedup trick**: after the first test-blocker run, every model
iteration is **5-10 min** — load parquets, LightGBM predict, threshold,
format. Do NOT re-block on test data unless you change the blocker.

---

## 7. Phase G + H — graph refinement and packaging

(Not yet written; detail to come after Phase F is verified.)

```bash
# Phase G (post Phase F):
python scripts/graph_refine.py
# Closes triangles where min(p_AB, p_BC) ≥ 0.85 with damping 0.9.

# Phase H (final submission):
python scripts/predict.py  # re-run with refined predictions
python utils/validate_submission.py ... (same as Phase F)
zip -r submission.zip output/ code/business_entity_resolution/ Documentation_template.md
```

---

## 8. Common pitfalls (recap from STATUS.md)

- **Polars `.str.split(...)`** on a nullable String column: wrap with `fill_null("")` first.
- **`str.len_chars().abs().cast(Int32)` overflows**: cast to Int32 BEFORE abs.
- **Don't `.select(["source1_entity_id"])`** before adding blocking key columns (loses addr_city etc.).
- **Country hard filter is mandatory**: drop cross-country candidates at probe time.
- **Quality-tier filter is OR, not AND**: a candidate is kept if it passes at least one of the four floors.
- **Structural contribution is capped at 3** in composite score (`min(n_struct_keys, 3) / 3`), not 8.
- **AWS t3.medium is enough for blocking** but NOT for training. Move to a ≥32 GiB box for Phase E.

---

## 9. Memory + time budget (quick reference)

| Stage | Min CPU | Min RAM | Disk | Wall-clock |
| --- | --- | --- | --- | --- |
| Phase C v3 (per direction) | 2 vCPU | **8 GiB** | 30 GB | 50-80 min on AWS t3.medium |
| Phase C2 (combine + validate) | 2 vCPU | 8 GiB | 5 GB | <5 min |
| Phase E (LightGBM) | 8 cores | **32 GiB** | 50 GB | 3-5 h |
| Phase F (one-time test block) | 2 vCPU | 8 GiB | +3 GB | ~50 min/direction |
| Phase F (model iteration) | 2 vCPU | 8 GiB | 0 | 5-10 min |
| Phase G | 4 cores | 16 GiB | +1 GB | ~15 min |
| Phase H (packaging) | 2 vCPU | 8 GiB | +1 GB | <5 min |
