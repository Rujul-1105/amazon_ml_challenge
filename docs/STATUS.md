# Project STATUS

Current state of each phase. Update after every milestone. **Future
sessions: read this file FIRST before doing anything.**

---

## Phase A — env / packages ✅

- Installed: polars 1.44, pandas 3.0, numpy 2.5, scikit-learn 1.9, lightgbm 4.7, faiss-cpu 1.15, networkx 3.7, datasketch 2.0, rapidfuzz 3.14, pyarrow 25.0.1, indic-transliteration, regex, unidecode.
- Did not install: annoy (build failure), python-recordlinkage (no Python 3.14 wheel).
- Also installed: pickle, scipy (for sparse matrix ops), joblib.
- Folder structure: `code/business_entity_resolution/{src,artifacts,docs,output,scripts}/`.
- Artifacts: `eda_country.csv`, `eda_counts.csv`, `eda_missing.csv`, `eda_report.md` (EDA CSVs, from prior session — useful for sanity-check only).

## Phase B — normalization ✅

- **libpostal** (C library + 2 GB language models) built and installed.
- `src/normalize.py` calls `postal.parser.parse_address` per record.
- Outputs (train-only, **intact and required for all of Phase C onwards**):
  - `artifacts/s1_norm_train.parquet` — 2.2 M rows, ~219 MB, 21 cols
  - `artifacts/s2_norm_train.parquet` — 5.0 M rows, ~525 MB, 21 cols
  - `artifacts/s3_norm_train.parquet` — 5.3 M rows, ~544 MB, 21 cols
- Schema per record: `entity_id, country, name_clean, name_latin, name_tokens, name_ngram_key, name_dev_ratio, addr_clean, addr_latin, addr_first_word, addr_last_word, addr_ngram_key, addr_house_number, addr_road, addr_unit, addr_suburb, addr_zip, addr_state, addr_city, name_missing, addr_missing`.
- **Test files NOT touched** — train-only policy.

## Phase C — blocking 🔄 (v3 LOCKED-IN; v2 S2 already produced, needs replacement)

### v1 (the abandoned "12-key" design)

`src/blocking.py` implements the full 12-key plan (9 inverted + TF-IDF +
2 MinHash + Faiss). Three failed attempts on the 15 GB box, all OOM:
| # | Plan | Where killed |
| --- | --- | --- |
| 1 | Faiss IndexFlatIP + MinHash 128 perms simultaneously | mid-TF-IDF build |
| 2 | Drop MinHash, HashingVectorizer `n_features=4096`, Faiss IndexIVFFlat | at TF-IDF build |
| 3 | Sequential-by-stage (build each index → persist → free RAM) | at startup before Stage 1 wrote anything |

Dropped — too memory-hungry for 8 GB boxes.

### v2 (already produced — INSUFFICIENT RECALL)

`scripts/block_features.py` ran with v2 design:
- **8 cheap structural inverted indexes** (cap 500/bucket)
- **Char-trigram inverted index** (cap 100/bucket)
- Python dict lookups for probing
- Per-S1 cap = 25 candidates

**Validation (run `scripts/validate_block_recall.py` on `block_S2_features.parquet`):**
- Mean recall: **20.4%** ← catastrophic
- Median recall: **0.0%**
- 50.7% of S1 entities have ZERO true matches recalled
- 80.2% of S1 recall < 50%
- S2 produced `block_S2_features.parquet` (55 M rows × 36 cols, 1.55 GB) — **will be overwritten by v3**.

**Why v2 failed (root cause):**
- Char-trigrams are too noisy: ` Co`, `inc`, `ent`, `ati` etc. appear in 15-30% of docs; their 100-slot buckets fill with random businesses.
- Per-key cap of 500/bucket kills recall for popular keys (e.g., `city="Chicago"` has 50K+ S2 entities but only 0.5% are kept).
- Per-S1 cap of 25 is too tight — many true matches get pushed out by marginal-rank candidates.

### v3 (LOCKED-IN — current design)

**Goal**: push recall from 20% to 85%+ on the train ground truth, with strict 50-candidate-per-S1 cap (≤ 220 M total pairs).

**Three indexes, hard threshold UNION, quality-tier OR filter, top-50 cap.**

#### Three indexes

1. **Structural** (8 keys, unchanged): `dict[key_value → list[id]]`, cap 500/bucket.
   - Keys: `city`, `state`, `road`, `house`, `name_fw` (first word of name), `addr_fw`, `city_state` (compound), `house_road` (compound).

2. **Word-token inverted index** (replaces char-trigrams).
   - Tokens extracted from `name_clean + " " + addr_clean`.
   - `tokenize(text)` → set of lowercase alphanumerics, drop:
     - length < 3 chars
     - pure-digit
     - **STOP_TOKENS** (curated list):
       - English stop-words: `the, a, an, and, or, of, in, at, on, to, for, with, by, from, as, is, are, was, were, be, been, it, its, this, that, these, those`
       - Business legal forms: `inc, incorporated, ltd, limited, llc, llp, corp, corporation, company, co, companies, pvt, private, plc, gmbh, sa, srl, group, holdings, partners, associates, enterprises, international, global, world`
       - Common nouns: `no, number, de, la, el, los, las, san, santa, new, old, north, south, east, west, central, city, state, india, usa, us, uk`
   - Cap 500/bucket.

3. **Sorted-token neighborhood** (replaces trigrams + rescues typos/reorders).
   - `canonical_tokens(text)` → " ".join(sorted(tokenize(text))).
   - Examples: `"Apex Construction Inc"` → `"apex construction"`; `"APEX CONSTRUCTON Inc"` → `"apex constructon"` (1 edit-distance away).
   - Build: `np.argsort(candidates_canonical)` — O(N log N) numpy.
   - Probe: `np.searchsorted` to find S1's canonical-pos in O(log N), take window **±50** → 101 candidates per S1.

#### Hard country pre-filter (free precision boost)

`scripts/block_features.py` MUST drop any candidate where `cand.country != s1.country`. Country is 100% same on true matches (verified).

#### Per-chunk probe order (mandatory)

For each 10K S1 chunk:
1. Compute S1 features.
2. Probe Index 1 (structural) → structural_candidates.
3. Probe Index 2 (tokens) → token_candidates.
4. Probe Index 3 (sorted-neighborhood) → sortedn_candidates.
5. **Union all three** with per-pair metadata:
   - `n_struct_keys` (Int8)
   - `n_tokens_shared` (Int8)
   - `sortedn_rank` (Int16, 0 if not from SN)
   - `from_struct`, `from_token`, `from_sortedn` (3 booleans)
6. **Quality-tier filter (OR gate)**: keep pair iff it clears at least ONE of these 4 floors (user-confirmed OR logic):

   | Floor | Source | Hard constraint |
   | --- | --- | --- |
   | A | structural | `n_struct_keys ≥ 2` |
   | B | tokens | `n_tokens_shared ≥ 2` |
   | C | structural compound | `road == cand_road` AND `city == cand_city` AND both non-empty |
   | D | sorted-near + token | `sortedn_rank ≤ 10` AND `n_tokens_shared ≥ 1` |

7. **Composite score** (per-pair, used only for top-50 cap tiebreak):
   ```python
   struct_score = min(n_struct_keys, 3) / 3.0      # capped at 3 (city/state/country trivially match)
   sortedn_prox = 1.0 / (1 + sortedn_rank)         # closer is better
   composite = (
       0.20 * struct_score
       + 0.45 * min(n_tokens_shared, 5) / 5.0
       + 0.20 * (1.0 if from_sortedn else 0.0)
       + 0.15 * sortedn_prox
   )
   ```
   **Why structural capped at 3**: city/state/country are not discriminative on
   their own (always same / often same across many businesses). Road,
   house_number, name_first_word, and compounds are informative.
8. **Final top-50 cap per S1** by composite_score desc.

#### Output schema (36 cols)

`source1_entity_id, candidate_entity_id, candidate_source, n_struct_keys, n_tokens_shared, sortedn_rank, from_struct, from_token, from_sortedn, block_score, s1_country, m__country, country_eq, name_first_token_eq, name_token_jaccard, name_n_chars_diff, cross_script_pair, addr_first_word_eq, addr_last_word_eq, addr_city_eq, addr_house_number_eq, addr_state_eq, addr_road_eq, addr_zip_eq, addr_unit_eq, addr_suburb_eq, s1_name_missing, m_name_missing, s1_addr_missing, m_addr_missing, name_token_set_ratio, name_partial_ratio, name_token_sort_ratio, name_ratio, name_latin_token_set_ratio, addr_token_set_ratio, addr_partial_ratio, addr_token_sort_ratio, addr_ratio, addr_latin_token_set_ratio`

(`n_tokens_shared` and `sortedn_rank` are new vs. v2; the rest are unchanged from v2's 27 features + metadata columns.)

#### RAM and time

| Component | RAM (MB) |
| --- | --- |
| 8 structural indexes (cap 500/bucket) | ~600 |
| Word-token index (5M × ~10 tokens × dict) | ~800 |
| Sorted canonical strings (5M × ~30 chars) + np.array | ~400 |
| Per-chunk working set (10K S1 × 50 candidates × 32 cols) | ~200 |
| **Peak** | **~2.5 GB** |

Time per direction: ~30-60 min on 4-8 cores; ~50-80 min on AWS t3.medium (2 vCPU / 8 GiB). The user's AWS box is sufficient.

#### Expected recall

| Index alone | Approx recall |
| --- | --- |
| Structural (≥1 key) | ~30% |
| Token (≥1 shared token, post-stop-filter) | ~75% |
| Sorted-token neighborhood (window ±50) | ~95% |
| **All 3 with hard-threshold OR + top-50** | **85-92%** |

#### Implementation specifics (LITERAL — copy into `block_features.py`)

```python
import re
import numpy as np
import polars as pl

# --- Stop tokens (curated; expand if needed) ---
STOP_TOKENS = {
    "the","a","an","and","or","of","in","at","on","to","for","with","by","from",
    "as","is","are","was","were","be","been","it","its","this","that","these","those",
    "inc","incorporated","ltd","limited","llc","llp","corp","corporation","company",
    "co","companies","pvt","private","plc","gmbh","sa","srl",
    "group","holdings","partners","associates","enterprises",
    "international","global","world",
    "no","number","de","la","el","los","las","san","santa",
    "new","old","north","south","east","west","central",
    "city","state","india","usa","us","uk",
}

# --- Structural keys (unchanged from v2) ---
KEY_EXTRACTORS = {
    "city":     pl.col("addr_city").str.strip_chars().str.to_lowercase(),
    "state":    pl.col("addr_state").str.strip_chars().str.to_lowercase(),
    "road":     pl.col("addr_road").str.strip_chars().str.to_lowercase(),
    "house":    pl.col("addr_house_number").str.strip_chars().str.to_lowercase(),
    "name_fw":  pl.col("name_clean").str.split(" ").list.first()
                    .fill_null("").str.strip_chars().str.to_lowercase(),
    "addr_fw":  pl.col("addr_first_word").str.strip_chars().str.to_lowercase(),
    "city_state":  pl.concat_str([pl.col("addr_city"), pl.lit("|"),
                                   pl.col("addr_state")],
                                  separator="", ignore_nulls=False)
                      .str.strip_chars().str.to_lowercase(),
    "house_road":  pl.concat_str([pl.col("addr_house_number"), pl.lit("|"),
                                   pl.col("addr_road")],
                                  separator="", ignore_nulls=False)
                      .str.strip_chars().str.to_lowercase(),
}

def tokenize(text):
    """Alphanumeric tokens, len>=3, not in STOP_TOKENS, not pure-digit."""
    if not text:
        return set()
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in toks
            if len(t) >= 3 and t not in STOP_TOKENS and not t.isdigit()}

def canonical_tokens(text):
    return " ".join(sorted(tokenize(text)))

# --- Quality-tier filter (one of four OR) ---
def passes_quality_floor(n_struct_keys, n_tokens_shared, sortedn_rank,
                         s1_road, m_road, s1_city, m_city):
    if n_struct_keys >= 2:
        return True                                                       # floor A
    if n_tokens_shared >= 2:
        return True                                                       # floor B
    if (s1_road and m_road and s1_road == m_road
        and s1_city and m_city and s1_city == m_city):
        return True                                                       # floor C
    if sortedn_rank > 0 and sortedn_rank <= 10 and n_tokens_shared >= 1:
        return True                                                       # floor D
    return False

# --- Composite score (for top-50 tiebreak) ---
def composite_score(n_struct_keys, n_tokens_shared, sortedn_rank):
    struct_score = min(n_struct_keys, 3) / 3.0
    sortedn_prox = 1.0 / (1 + sortedn_rank) if sortedn_rank > 0 else 0.0
    return (
        0.20 * struct_score
        + 0.45 * min(n_tokens_shared, 5) / 5.0
        + 0.20 * (1.0 if sortedn_rank > 0 else 0.0)
        + 0.15 * sortedn_prox
    )
```

## Phase D — feature engineering ✅

Same as v2 — 27 features (cheap polars booleans + 10 rapidfuzz ratios). Same per-pair output structure. Already computed inside `block_features.py`. **Nothing to migrate separately** — Phase C v3 run will emit features as it produces candidates.

## Phase E — LightGBM classifier ⏸ pending

**Hardware target**: **32 GiB RAM / 16 cores / 50 GB SSD** (8 GB box CANNOT train).

**Hyperparams**:
- `objective: binary, metric: [binary_logloss, auc], num_leaves: 63, min_data_in_leaf: 1000, learning_rate: 0.05, n_estimators: 1500 (early stop 50), feature_fraction: 0.8, bagging_fraction: 0.8, bagging_freq: 5, lambda_l1: 0.1, lambda_l2: 0.1, scale_pos_weight: DYNAMIC (n_neg/n_pos after sampling), max_bin: 255, bin_construct_sample_cnt: 200_000, n_jobs: -1, seed: 42`.

**Search strategy** (3 ML techniques):
1. Coarse grid: `num_leaves ∈ {31, 63, 127}` × `learning_rate ∈ {0.03, 0.05, 0.1}` on 20 M-row sample → ~30 min.
2. Optuna: 30 trials, TPE → ~2 h.
3. Final fit at best params on 50-80 M rows → ~1.5-2 h.

**Sampling**: all positives (~6.4 M) + 50% random + 50% hard-negatives (mined from quick baseline) at neg:pos 5:1 to 10:1.

**Save**: `artifacts/lgbm_classifier.txt`.

## Phase F — inference ⏸ pending

**Test blocker ONCE, cache, iterate cheaply**:

```bash
# ONE-TIME (~30-60 min on AWS t3.medium):
python scripts/block_features.py --candidate-source S2 --suffix S2_TEST
python scripts/block_features.py --candidate-source S3 --suffix S3_TEST
# Writes artifacts/block_S{2,3}_TEST_features.parquet (each ~1.5 GB).
```

After cache, every model iteration is just `predict.py` in **5-10 min**:
- Load 2 parquets (~30 s)
- LightGBM predict (~3-5 min)
- Per-bucket threshold + format (~30 s)

**Per-(country, source) threshold tuning**: sweep on val for max F_0.5.

## Phase G — graph refinement ⏸ pending

Build edge graph from predicted matches; close triangles only if `min(p_AB, p_BC) ≥ 0.85` with damping 0.9. Re-apply singleton detector.

## Phase H — packaging ⏸ pending

```bash
python scripts/predict.py
python utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
zip -r submission.zip output/ code/business_entity_resolution/ Documentation_template.md
```

---

## Hard-fail bugs to AVOID (learned the hard way)

These come up because the previous sessions iterated fast:

1. **Polars `.str.split(" ")` requires `fill_null("")` before** — otherwise null columns throw `SchemaError: invalid series dtype: expected String, got null`. Wrap: `pl.col("s1__name_clean").fill_null("").str.split(" ")`.

2. **`pl.from_dicts` over many rows** can produce inconsistent types — use `infer_schema_length=10000` to avoid spurious dtype mismatches.

3. **`str.len_chars()` returns u32**. Wrapping in `.abs().cast(Int32)` overflows. Cast to Int32 BEFORE abs: `(...str.len_chars().cast(pl.Int32)).abs()`.

4. **Don't drop columns before computing blocking keys**. `chunk.with_columns(expr).filter(...)` works only if the columns referenced by `expr` are still present. Don't `.select(["source1_entity_id"])` before adding key columns.

5. **Country hard filter is mandatory** even when probes return candidates — country equality is a free precision filter that eliminates ~2.5% of cross-country noise.

6. **AWS t3.medium (2 vCPU / 8 GiB) is enough for Phase C v3 blocking** — but NOT for Phase E (LightGBM training). Move Phase E to a 32+ GiB box.

7. **Hard-threshold QUALITY filter is OR not AND**: a candidate is kept if it passes **at least one** of the four floors. Do NOT make it AND — that would lose pairs strong on one axis but weak on others.

8. **structural contribution is capped at min(n_struct_keys, 3) / 3** — not /8.0. Because city/state/country trivially match (always same-country, often same-state, often same-city), so even matching all 3 of them should contribute at most 1.0, not 3/8=0.375.