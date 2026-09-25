# Project STATUS

Current state of each phase. Updated after every completed milestone.

## Phase A — env / packages ✅

- Installed: polars 1.44, pandas 3.0, numpy 2.5, scikit-learn 1.9, lightgbm 4.7, faiss-cpu 1.15, networkx 3.7, datasketch 2.0, rapidfuzz 3.14, pyarrow 25.0.1, indic-transliteration, regex, unidecode.
- Did not install: annoy (build failure), python-recordlinkage (no Python 3.14 wheel).
- Folder structure created: `code/business_entity_resolution/{src,artifacts,docs,output,scripts}/`.
- Artifacts: `eda_country.csv`, `eda_counts.csv`, `eda_missing.csv`, `eda_report.md`.

## Phase B — normalization ✅

- **libpostal** (C library + 2 GB language models) built and installed to `/home/rujul/local/`.
- `src/normalize.py` rewritten to call `postal.parser.parse_address` for each record. **Deliberately dropped `expand_address`** (~2× speedup).
- Ran with 4 workers, setsid+nohup detached, RAM-aware (n_workers from `/proc/meminfo`).
- Outputs (train-only):
  - `artifacts/s1_norm_train.parquet`  — 2.2 M rows, 219 MB, 21 cols
  - `artifacts/s2_norm_train.parquet`  — 5.0 M rows, 525 MB, 21 cols
  - `artifacts/s3_norm_train.parquet`  — 5.3 M rows, 544 MB, 21 cols

Schema per record: `entity_id, country, name_clean, name_latin, name_tokens, name_ngram_key, name_dev_ratio, addr_clean, addr_latin, addr_first_word, addr_last_word, addr_ngram_key, addr_house_number, addr_road, addr_unit, addr_suburb, addr_zip, addr_state, addr_city, name_missing, addr_missing`.

**Test files (`test_source*.tsv`) NOT touched** — train-only policy in effect.

## Phase C — blocking ⏸ resume here

Methodology doc: `code/business_entity_resolution/docs/phase_c_blocking.md` (final 12-key design with first-word rationale).

### Three failed attempts on the 15 GB box

| # | Build plan | Where killed |
|---|---|---|
| 1 | Faiss IndexFlatIP + MinHash 128 perms simultaneously | mid-TF-IDF build |
| 2 | Drop MinHash, HashingVectorizer `n_features=4096`, Faiss IndexIVFFlat | at TF-IDF build |
| 3 | Sequential-by-stage (build each index → persist → free RAM) | at startup before Stage 1 wrote anything |

`_blocking_idx/` directory created but empty. No `blocks_*.parquet` written.

Final `src/blocking.py` already implements the sequential-by-stage pattern + idempotent guards. It is ready to run on new hardware (≥16 GB RAM).

### What we are NOT changing

- 12-key design (K1–K12 inverted dicts + TF-IDF + MinHash) — frozen, see methodology doc.
- Train-only policy — blocks built only on `s{1,2,3}_norm_train.parquet`.
- Test files still untouched — they enter the picture only in Phase H inference.

## Phase D — feature engineering ⏸

Not started. ~40 pairwise string-similarity features per (S1, candidate) pair. Joblib-parallelized, chunked to keep RAM ≤8 GB. Sequence per pair: name Jaccard, char-ngram Jaccard, rapidfuzz (4 ratios), LCS, length ratio, numeric overlap, is_cross_script, addr_* similarity, addr_zip_exact, addr_state_exact, addr_city_exact, candidate_rank, top-1/2/3 scores, score_spread, qratio_name, qratio_addr.

## Phase E — model training ⏸

Not started. `src/train_classifier.py` (LightGBM binary, `objective=binary, lr=0.05, num_leaves=127`) + `src/singleton_detector.py` (separate small LGBM on per-S1 aggregate features). Train/val split by S1 ID (90/10), cluster-aware.

## Phase F — inference + threshold ⏸

Not started. `src/predict.py` + `src/threshold.py` (per-(country, source) τ) + `src/cascade.py` (deterministic high-precision auto-commit) + `src/f05.py` (already exists) reused. Apply singleton detector + graph refinement (Phase G) before writing `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

## Phase G — graph refinement ⏸

Not started. `src/graph_refine.py`: build edge graph from predicted matches, close triangles only if `min(p_AB, p_BC) ≥ 0.85` with damping 0.9. Re-apply singleton detector after closure. Safety: never add cross-country edges, never edges for singleton candidates.

## Phase H — packaging ⏸

Not started. Generate `output/matching_results.tsv` and `output/candidate_pairs.tsv` from the trained model on test data. This is the only step that reads `test_*.tsv`. Run `python3 data_set/student_resource/utils/validate_submission.py --check-ids`. Build `<team>_submission.zip` per the problem rules.
