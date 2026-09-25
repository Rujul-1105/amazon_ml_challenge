# Phase C — Blocking Methodology

**Goal:** for each of 2.2M S1 entities, produce ≤ 100 candidate S2/S3 records to
score in Phase D. Naïve pair count is ~22 billion; we need a ~100,000×
reduction without losing recall.

**Target:** blocking recall ≥ 0.92 on a 10% cluster-aware held-out slice.

**Train-only:** reads only `s{1,2,3}_norm_train.parquet` +
`train_ground_truth.tsv`. Test files not touched (policy enforced in code).

---

## 1. Why bucket by first word of name, not by complete name

Business name fields for the same real-world entity almost always share the
**brand token** but differ on the legal suffix:

```
"Apex Construction Inc"     →  first word: apex
"Apex Pvt Ltd"               →  first word: apex
"Apex"                       →  first word: apex
"Apex Industries Limited"    →  first word: apex
"APEX Corp"                  →  first word: apex
```

A first-word inverted index buckets all four together (high recall within the
bucket) and lets Phase D's pairwise classifier filter the inevitable false
positives using city/state/road features.

Bucketing by **complete name** would give near-zero overlaps across the legal
suffix variants and miss them entirely. Typos and reordering would also slip
through.

### Comparison (illustrative for our dataset)

| Strategy | Approx. block recall | Bucket-size characteristics |
|---|---|---|
| First word only (K5) | ~50% standalone | ~20–2,000 per unique value |
| First word + TF-IDF + MinHash (this plan) | **~92%+** | first-word gives the biggest bucket; ANN candidates add recall |
| Complete name only | ~10% standalone | mostly size 1, near-zero overlap |
| Complete name + TF-IDF + MinHash | ~80% | misses brand-token shortcut |

This is the canonical recipe in entity-resolution literature (Bilenko & Mooney
2003, Christen 2012). The first-word block is the workhorse; ANN indexes
(MinHash K11 and TF-IDF K10) rescue recall on legal-suffix variants, typos,
and reordering.

---

## 2. The 12 blocking keys (final, after data inspection)

We build **9 structured inverted dicts** (high precision, cheap lookups) and
**3 approximate-NN indexes** (high recall, higher cost) per country. For each
S1 entity we union their candidates and cap at top-100 by string-similarity
score.

### 9 inverted indexes (built once per country, per source)

| K | Inverted-index key (extracted from each S2/S3 record) | Source field | S2 fill | S3 fill |
|---|---|---|---|---|
| **K1** | `city` | libpostal `addr_city` | 82% | 78% |
| **K2** | `state` | libpostal `addr_state` | 78% | 68% |
| **K3** | `road` | libpostal `addr_road` | 86% | 84% |
| **K4** | `house_number` | libpostal `addr_house_number` | 87% | 87% |
| **K5** | `name_first_word` | first whitespace-token of `name_clean` | 100% | 100% |
| **K6** | `addr_first_word` | first non-stop word of `addr_latin` | varies | varies |
| **K7** | `addr_last_word` | last non-stop word of `addr_latin` | varies | varies |
| **K8** | `city + "|" + state` | compound (city_state) | 65% | 60% |
| **K9** | `house_number + "|" + road` | compound ("same address") | 75% | 70% |

### 3 ANN indexes (built once per country over all S2+S3 text)

| K | Method | Source |
|---|---|---|
| **K10** | TF-IDF char 1–3 grams + Faiss CPU `IndexFlatIP` (or `IndexIVFFlat` if RAM tight) top-100 cosine NN per query | `(name_latin + " " + addr_latin)` |
| **K11** | MinHash LSH on char 3-gram shingles, threshold 0.7, 128 perms | `name_latin` |
| **K12** | MinHash LSH on char 3-gram shingles, threshold 0.7, 128 perms | `addr_latin` |

### What we explicitly do NOT use as a blocking key

| Field | Why dropped |
|---|---|
| `addr_zip` | 2% fill in train — too sparse; would cover <2% of true matches alone |
| `addr_unit` | 9–13% fill — too narrow |
| `name_ngram_key` | Replaced by `name_first_word` (more stable) |
| `addr_ngram_key` | Replaced by `addr_first_word` (more stable) |
| `country` | Used as a *stratification* axis, not a key (every pair is within-country) |

---

## 3. Per-S1 query flow (per country)

```
For each S1 entity:
    1. Compute 9 structured keys from S1's normalized fields:
       K1=city, K2=state, K3=road, K4=house_number, K5=name_first_word,
       K6=addr_first_word, K7=addr_last_word, K8=city+state, K9=house+road
    2. Look up each key in its respective inverted dict → collect all
       S2/S3 record IDs that share any key with S1.
    3. Encode S1's (name_latin + " " + addr_latin) → query Faiss TF-IDF
       index → top-100 by cosine.
    4. Build MinHash of S1's name_latin and addr_latin → query MinHash LSH
       → all candidates above similarity 0.7.
    5. Union all candidates, dedup by (cand_id, source_flag, country).
       Take the per-candidate max score across keys.
    6. Cap at top-100 by rapidfuzz.fuzz.QRatio(name_latin) then QRatio(addr_latin).
    7. Empty set → potential singleton (Phase E singleton detector).
```

**Estimated throughput:** ~5–10 candidates per S1 for typical records,
up to 100 for dense "Apex"-like buckets. Total scored-pair count
~150–200M (down from 22B naïve).

---

## 4. Compute & memory budget (per country)

| Component | Memory | Build time | Query time |
|---|---|---|---|
| 9 inverted dicts | <2 GB | <5 min | <1 sec / 10k S1 |
| TF-IDF char 1–3 grams (CSR sparse, ~5M docs × ~64k hash buckets) | ~1.3 GB | 5–10 min | <10 sec / 100k S1 (batched) |
| Faiss `IndexFlatIP` | ~1.3 GB (hash-bounded dim) | 5–10 min | ~5 sec / 100k S1 |
| MinHash LSH (128 perms) | ~6–10 GB (sketched records) | 10–15 min | ~5 sec / 100k S1 |

**Total per country:** ~10–15 GB peak; we run 2 countries sequentially
(US first, then India) to stay below the 15 GB cap.

**Fallbacks if RAM tight:**

- Reduce TF-IDF `n_features` from 2^16 → 2^14 (16× less memory).
- Replace Faiss `IndexFlatIP` with `IndexIVFFlat` (nlist=512, nprobe=16) — 5–10×
  compression at small accuracy cost.
- Reduce MinHash perms from 128 → 64 — still high-quality LSH.

---

## 5. Per-country, per-source output schema

| File | Rows expected | Time |
|---|---|---|
| `artifacts/blocks_country=US_source=S2.parquet` | ~50–100M candidate pairs | 10–15 min build + query |
| `artifacts/blocks_country=US_source=S3.parquet` | ~50–100M candidate pairs | 10–15 min build + query |
| `artifacts/blocks_country=India_source=S2.parquet` | ~30–60M candidate pairs | 10–15 min |
| `artifacts/blocks_country=India_source=S3.parquet` | ~30–60M candidate pairs | 10–15 min |

Columns: `(s1_id, cand_id, score, source_flag, country)`.

**Total storage:** ~2 GB across the 4 parquets (snappy compressed).

---

## 6. Validation

```python
# pseudo-code
held_out_s1_ids = sample 10% of S1 IDs (stratified by cluster size)
true_matches   = load from train_ground_truth
pred_candidates = blocks.loc[held_out_s1_ids]

for s1_id in held_out_s1_ids:
    true = true_matches.get(s1_id, set())
    pred = pred_candidates.get(s1_id, set())
    recall[s1_id] = len(true & pred) / max(1, len(true))

overall_recall = mean(recall.values())
```

**Per-bucket report:** `{(country, source): recall}` for US-S2, US-S3, India-S2,
India-S3.

**Targets:**
- Overall recall ≥ 0.92
- Per-bucket recall ≥ 0.85
- Mean candidates per S1 ≤ 100

**Tuning actions if recall is below target:**
1. Relax MinHash threshold (0.7 → 0.6).
2. Increase Faiss top-K (100 → 150, then 200).
3. Add additional structured keys (e.g., `name_last_word`, `addr_clean` first 3 tokens).
4. Lower TF-IDF `min_df` to expand vocabulary.

---

## 7. Why not use more sophisticated methods

We don't use:

- **Sorted-neighborhood blocking** (sort records by a key, slide window).
  Higher implementation cost; we already get most of the same coverage via
  the K1–K9 union + K10 cosine.

- **Learned blocking** (DeepER, etc.). Requires labeled pairs for training;
  we have ground truth but saved as `(s1, [matches])` lists, not as
  positive/negative pair labels. Phase D is the right place to spend that
  labeled data, not blocking.

- **Graph-based candidate generation.** Powerful but memory-heavy on this
  dataset (10M nodes). Reserved for Phase G graph refinement on the much
  smaller predicted-pair graph.

---

## 8. References

- Bilenko, M. & Mooney, R. J. (2003). "Adaptive Blocking: Learning to Scale Up
  Record Linkage." ICDM.
- Christen, P. (2012). *Data Matching* (Springer).
- McCallum, A., Bilmes, S. & Ouyang, M. (2011). "The CRFs for Blocking
  Workshop." (MinHash LSH for entity resolution.)
- Leskovec, R. & Rajaraman, A. (2014). *Mining of Massive Datasets* —
  chapter on locality-sensitive hashing (MinHash derivation).
