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

## Phase C — blocking 🔄 in progress (S2 running)

The original 12-key design (K1–K9 inverted dicts + TF-IDF K10 + MinHash K11/K12) failed 3× on the 15 GB box per the historical notes below. On the new hardware (8 GB RAM), we re-designed to a **hybrid blocker** that fits in RAM:

- **8 cheap structural inverted indexes** (K1–K9 minus K7 addr_last_word): city, state, road, house_number, name_first_word, addr_first_word, city|state, house|road. Each is a `dict[key_value → list[entity_id]]` with per-bucket cap 500.
- **Char-trigram inverted index** for fuzzy blocking (the cheap substitute for TF-IDF/MinHash on 8 GB RAM). 41,787 unique trigrams, 2.2 M (trigram, id) pairs.
- MinHash and TF-IDF dropped (each is OOM-prone on 8 GB). The trigram index catches most of the typo/abbreviation/word-order cases that TF-IDF would have.

Single streaming script:
`code/business_entity_resolution/scripts/block_features.py`

**Smoke test passed** (re-run on a 5% slice of S1):
- Trigram index built in ~280 s, cand_dict (5 M entries) in ~48 s
- 2 chunks of 10K S1 processed in ~225 s total
- 500 K candidate pairs, 36 cols, schema correct (country_eq mean
  0.976, addr_token_set_ratio mean 60.5)

**S2 run in flight**: launched in background; expected ~5–6 h
wall-clock (220 chunks × ~100 s). Output:
`code/business_entity_resolution/artifacts/block_S2_features.parquet`
(~50 M rows × 36 cols).

### Historical context (the original 3 failed attempts on the 15 GB box)

| # | Build plan | Where killed |
|---|---|---|
| 1 | Faiss IndexFlatIP + MinHash 128 perms simultaneously | mid-TF-IDF build |
| 2 | Drop MinHash, HashingVectorizer `n_features=4096`, Faiss IndexIVFFlat | at TF-IDF build |
| 3 | Sequential-by-stage (build each index → persist → free RAM) | at startup before Stage 1 wrote anything |

The strategy was frozen because the hardware was insufficient. The
final `src/blocking.py` was overwritten by the new hybrid design
in `scripts/block_features.py`; the 12-key methodology doc remains
at `docs/phase_c_blocking.md` for reference.

## Phase D — feature engineering ✅ done

Done as part of Phase C in `scripts/block_features.py`. Each candidate
pair carries 27 features:

- **Polars-side (cheap):** `country_eq`, `name_first_token_eq`,
  `name_token_jaccard`, `name_n_chars_diff`, `cross_script_pair`;
  `addr_first_word_eq`, `addr_last_word_eq`, `addr_city_eq`,
  `addr_house_number_eq`, `addr_state_eq`, `addr_road_eq`,
  `addr_zip_eq`, `addr_unit_eq`, `addr_suburb_eq`;
  `s1_name_missing`, `m_name_missing`, `s1_addr_missing`,
  `m_addr_missing`.
- **Python/rapidfuzz (10):** `name_token_set_ratio`, `name_partial_ratio`,
  `name_token_sort_ratio`, `name_ratio`, `name_latin_token_set_ratio`;
  `addr_token_set_ratio`, `addr_partial_ratio`, `addr_token_sort_ratio`,
  `addr_ratio`, `addr_latin_token_set_ratio`.

Methodology and (pre-blocking) sample-stat rationale are documented
in `code/business_entity_resolution/artifacts/norm_similarity_insights.md`.

Train-only policy in effect throughout Phase C/D (test files not
touched; they enter the picture only in Phase H inference).

## Phase E — model training ⏸

Not started. `src/train_classifier.py` (LightGBM binary, `objective=binary, lr=0.05, num_leaves=127`) + `src/singleton_detector.py` (separate small LGBM on per-S1 aggregate features). Train/val split by S1 ID (90/10), cluster-aware.

## Phase F — inference + threshold ⏸

Not started. `src/predict.py` + `src/threshold.py` (per-(country, source) τ) + `src/cascade.py` (deterministic high-precision auto-commit) + `src/f05.py` (already exists) reused. Apply singleton detector + graph refinement (Phase G) before writing `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

## Phase G — graph refinement ⏸

Not started. `src/graph_refine.py`: build edge graph from predicted matches, close triangles only if `min(p_AB, p_BC) ≥ 0.85` with damping 0.9. Re-apply singleton detector after closure. Safety: never add cross-country edges, never edges for singleton candidates.

## Phase H — packaging ⏸

Not started. Generate `output/matching_results.tsv` and `output/candidate_pairs.tsv` from the trained model on test data. This is the only step that reads `test_*.tsv`. Run `python3 data_set/student_resource/utils/validate_submission.py --check-ids`. Build `<team>_submission.zip` per the problem rules.
