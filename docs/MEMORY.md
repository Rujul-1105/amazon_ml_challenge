# MEMORY.md — Session Handoff & Strategic Notes

**Read this on session start.** Pairs with `STATUS.md` (phase state) and
`RUNBOOK.md` (commands). This file is the "what we've learned and what
we should do next" companion — the kind of context that doesn't live in
code or git history.

---

## 0. TL;DR — what we know right now

- **Problem**: Amazon ML Challenge 2026 — Business Entity Resolution.
  Predict matches for each S1 entity against S2 ∪ S3 sources. Metric is
  macro-averaged **F_0.5** (precision-weighted). Train: US + India. Test:
  US + India + France.
- **Where we are**: Phase C (blocking) done; Phase E (LightGBM) is next.
- **Where we're stuck**: blocking recall at **0.6802** mean on the combined
  S2+S3 train holdout. Target is 0.85+ (per `STATUS.md`); pushing past 0.95
  is the user's stretch goal.
- **Three next-session priorities**:
  1. **Improve recall** past 0.85, ideally 0.95+.
  2. **Speed up the blocker** with dynamic RAM/core-aware parallel
     processing — current per-direction run is ~50-80 min on AWS t3.medium
     and ~25-40 min on an 8-core box; can probably halve that.
  3. **Then** run Phase E (LightGBM) on the higher-recall blocks.

---

## 1. Architectural decisions locked in (DO NOT CHANGE WITHOUT DISCUSSION)

### v3 blocking design — locked in `STATUS.md` §Phase C v3

The blocker probes **three inverted indexes**, unions the results, applies
a hard country pre-filter + a 4-floor OR quality-tier filter, and caps
each S1 at 50 candidates by a composite score.

| Index | What it does | Cap | Expected recall contribution |
|---|---|---|---|
| 1. Structural (8 keys) | `city, state, road, house, name_fw, addr_fw, city_state, house_road` — dict[key_value → list[id]] | 500/bucket | ~30% |
| 2. Word-token | `tokenize(name_clean + " " + addr_clean)` → set of len≥3, not-stop, non-digit alphanumerics → `dict[token → list[id]]` | 500/bucket | ~75% |
| 3. Sorted-token neighborhood | `canonical_tokens(text) = " ".join(sorted(tokenize(text)))`. Build `np.argsort`; probe via `np.searchsorted` with window ±50 | 101 candidates per S1 | ~95% |

**STOP_TOKENS list** (`scripts/block_features.py` ~line 111) is the curated
filter: English stop-words + business legal forms (`inc, ltd, llc, corp,
co, gmbh, sa, srl, plc, pvt, group, holdings, partners, enterprises,
international, global, world`) + common nouns (`no, number, de, la, el,
new, old, north, south, east, west, central, city, state, india, usa,
us, uk`).

### Hard country pre-filter
**Mandatory**: drop `cand.country != s1.country` candidates at probe time.
Country is 100% same on true matches — this is a free precision filter.
Implemented in `process_chunk` after `attach_fields`, before quality
filter. The pipeline **must not** propagate cross-country pairs to Phase E.

### 4-floor OR quality-tier filter (one of four must pass)

```python
def passes_quality_floor(n_struct_keys, n_tokens_shared, sortedn_rank,
                         s1_road, m_road, s1_city, m_city):
    if n_struct_keys >= 2:                                       # A
        return True
    if n_tokens_shared >= 2:                                     # B
        return True
    if (s1_road and m_road and s1_road == m_road
        and s1_city and m_city and s1_city == m_city):           # C
        return True
    if 0 < sortedn_rank <= 10 and n_tokens_shared >= 1:         # D
        return True
    return False
```

In polars expression form, see `process_chunk` in `block_features.py`.

**Why OR, not AND**: a pair that's strong on one axis but weak on the
others is still likely a true match (e.g., "name reordered, road+city
match"). AND would lose these. This was a hard lesson learned from v2.

### Composite score (for top-50 tiebreak only)
```python
struct_score = min(n_struct_keys, 3) / 3.0            # capped at 3, NOT 8
sortedn_prox = 1.0 / (1 + sortedn_rank) if rank > 0 else 0.0
composite = 0.20 * struct_score
          + 0.45 * min(n_tokens_shared, 5) / 5.0
          + 0.20 * (1.0 if from_sortedn else 0.0)
          + 0.15 * sortedn_prox
```
**Capping structural at 3** (not 8) is critical: city/state/country
trivially match and would otherwise dominate. The 4 informative keys
(road, house_number, name_first_word, compounds) are what count.

### Output schema (40 cols)
```
source1_entity_id, candidate_entity_id, candidate_source,
n_struct_keys, n_tokens_shared, sortedn_rank,
from_struct, from_token, from_sortedn,
block_score,
s1_country, m__country,
country_eq, name_first_token_eq, name_token_jaccard,
name_n_chars_diff, cross_script_pair,
addr_first_word_eq, addr_last_word_eq, addr_city_eq,
addr_house_number_eq, addr_state_eq, addr_road_eq,
addr_zip_eq, addr_unit_eq, addr_suburb_eq,
s1_name_missing, m_name_missing, s1_addr_missing, m_addr_missing,
name_token_set_ratio, name_partial_ratio, name_token_sort_ratio,
name_ratio, name_latin_token_set_ratio,
addr_token_set_ratio, addr_partial_ratio, addr_token_sort_ratio,
addr_ratio, addr_latin_token_set_ratio
```

---

## 2. Code locations — full map

| Path | Purpose |
|---|---|
| `code/business_entity_resolution/scripts/block_features.py` | Phase C — v3 blocker. Run with `--candidate-source S2` / `S3` / `S2_TEST` / `S3_TEST` |
| `code/business_entity_resolution/scripts/combine_block_features.py` | Original (RAM-hungry) combine — read both into memory |
| `code/business_entity_resolution/scripts/combine_block_features_lazy.py` | **NEW** RAM-safe combine — lazy polars + `sink_parquet` streaming write |
| `code/business_entity_resolution/scripts/validate_block_recall.py` | Original (RAM-hungry) recall validator — Python sets per pair |
| `code/business_entity_resolution/scripts/validate_block_recall_lazy.py` | **NEW** RAM-safe validator — all-lazy polars query, single streaming collect |
| `code/business_entity_resolution/src/blocking.py` | **Abandoned v1 12-key design** — NOT used. Don't waste time here |
| `code/business_entity_resolution/src/normalize.py` | Phase B — libpostal `parse_address` driver |
| `code/business_entity_resolution/src/transliterate.py` | Devanagari ↔ Latin helper used by normalize |
| `code/business_entity_resolution/src/f05.py` | Macro F_0.5 scorer (for Phase F inference eval) |
| `code/business_entity_resolution/src/io_utils.py` | polars TSV/parquet loaders |
| `code/business_entity_resolution/src/eda.py` | EDA driver (already executed, CSVs in artifacts/) |
| `code/business_entity_resolution/artifacts/s{1,2,3}_norm_train.parquet` | Phase B output — REQUIRED for Phase C |
| `code/business_entity_resolution/artifacts/block_S{2,3}_features.parquet` | Phase C output — per-direction blocks |
| `code/business_entity_resolution/artifacts/block_features.parquet` | Phase C2 — combined S2+S3 blocks |
| `code/business_entity_resolution/artifacts/_chunks/` | Per-chunk parquets (only during a blocker run; cleaned on success) |

---

## 3. Current results (as of 2026-09-27)

### Recall on combined S2+S3 train holdout (10%, ~208K S1)

```
Mean recall:        0.6802   (target 0.85; stretch 0.95)
Median recall:      0.7500
P25 recall:         0.5000
P75 recall:         1.0000
% S1 at recall=1.0: 0.4017
Mean cands/S1:      85.3      (50 cap from S2 + 50 cap from S3, w/ overlap)

Per-country:
  India: 0.6889 (n=83,252)
  US:    0.6744 (n=125,105)
```

### Recall per direction (S3-only, run earlier today)

```
Mean recall: 0.3377   ← much lower because S2 is the bigger, easier direction
Median:      0.3333
```

### v2 → v3 trajectory

| Metric | v2 | v3 (locked-in) | Target |
|---|---|---|---|
| Mean recall | 0.204 | 0.6802 | 0.85+ |
| Median recall | 0.0 | 0.75 | 1.0 |
| % S1 at recall=0 | 50.7% | ~10% (P25=0.5 means median S1 still gets half; very few zero) | 0% |
| 50% cap | hard | reached (85.3 mean cands) | flexible |

**Bottom line**: v3 is a big improvement but still ~17 pp short of the
spec target. The candidate funnel is too narrow for the bottom 25% of S1.

---

## 4. Known bugs we hit (and how we fixed them)

1. **`pl.from_dicts` schema inference bug** — `infer_schema_length=10000`
   missed mixed-type columns (`addr_house_number` sometimes str, sometimes
   int). Fix: `infer_schema_length=len(out_rows)`. **Symptom**:
   `ComputeError: could not append value "9193" of type: str to the
   builder`. See `block_features.py` `attach_fields` (~line 447).

2. **Resume after kill** — original script had no resume; killed mid-run
   wasted all prior chunks. Fix: added `--start-chunk N` flag + auto-detect
   by globbing `_chunks/block_{suffix}_*.parquet` for existing indices.
   See `main()` (~line 798-815).

3. **Validate script OOMs on combined data** — original `validate_block_recall.py`
   builds Python `set` per (S1, cand) pair → ~20 GB at 190M pairs. Fix:
   `validate_block_recall_lazy.py` uses all-lazy polars query, single
   `collect(engine="streaming")` at the end. Peak RAM < 2 GB.

4. **Combine script OOMs** — original `combine_block_features.py` loads
   both into eager DataFrames. Fix: `combine_block_features_lazy.py` uses
   `pl.scan_parquet` + `sink_parquet` (streaming write). ~27s wall-clock
   on 5.37 GB output, peak RAM < 1 GB.

5. **Polars `.join` type strictness** — LazyFrame can only join LazyFrame,
   DataFrame only with DataFrame. Fix: `.lazy()` everything that flows
   into a join. See `validate_block_recall_lazy.py` after edits.

---

## 5. Performance — current and target

### Current blocker wall-clock (single-threaded probe loop)
- AWS t3.medium (2 vCPU, 8 GiB): ~50-80 min / direction
- 8-core box: ~25-40 min / direction
- Breakdown (per direction):
  - Load sources + cand_dict: ~30s
  - Build 8 structural indexes: ~14s
  - Build word-token index: ~46s
  - Build sorted-neighborhood index: ~43s
  - Per-chunk probe + write (221 chunks × 10K S1): ~20-30 min

### Why it's slow — three hot spots

1. **Python loop in `process_chunk`** — `attach_fields` does `cand_dict.get()`
   per pair in pure Python (~100K pairs per chunk × 221 chunks = 22M dict
   lookups). Switching to vectorized polars joins OR multiprocessing this
   loop is the single biggest win.

2. **Per-chunk serial probes** — structural / token / sortedn probes are
   independent and could run in parallel via `multiprocessing.Pool` with
   `imap_unordered`. Probe results merge cheaply.

3. **`compute_fuzzy_features` (rapidfuzz loop)** — Python loop, ~10
   rapidfuzz calls per pair, 22M pairs. CPU-bound; would benefit from
   `rapidfuzz.process.cdist` on bulk arrays OR `concurrent.futures`.

### Dynamic RAM + core detection (TODO — required for next session)

```python
import os, psutil  # psutil is in requirements.txt

def detect_resources(target_free_gb: float = 6.0, ram_per_worker_gb: float = 1.5):
    """Pick chunk_size and n_workers from /proc/meminfo and os.cpu_count()."""
    free_gb = psutil.virtual_memory().available / (1024 ** 3)
    total_gb = psutil.virtual_memory().total / (1024 ** 3)
    physical_cores = os.cpu_count() or 2
    n_workers = max(1, min(4, int((free_gb - target_free_gb) / ram_per_worker_gb)))
    n_workers = min(n_workers, physical_cores - 1)  # leave 1 for I/O

    # Chunk size: bigger chunks = less Python loop overhead, but more RAM per chunk
    # Each chunk holds ~100K candidates × 40 cols × ~50 bytes = ~200 MB at peak
    # Goal: keep per-chunk working set under 1.5 GB
    if free_gb >= 24:
        chunk_size = 25000   # big chunks, parallel probe pool
    elif free_gb >= 12:
        chunk_size = 15000
    else:
        chunk_size = 10000   # safe default for 8 GB box

    return {
        "n_workers": n_workers,
        "chunk_size": chunk_size,
        "free_gb": round(free_gb, 1),
        "total_gb": round(total_gb, 1),
        "physical_cores": physical_cores,
    }
```

**Design ideas** for the rewrite (do NOT delete the working v3 code;
branch or wrap it):

- **Multi-process chunk pipeline**:
  - Main process: load sources, build indexes (one-time, ~2 min), distribute
  - Workers: each takes a chunk of S1 ids, runs probe + attach + features,
    writes its parquet to `_chunks/`, signals done
  - Main: collects `.done` markers, no global state, easy to resume
  - Use `multiprocessing.get_context("spawn")` to avoid fork-into-large-mem
    on Linux (avoids COW blowup with polars/numpy)

- **Vectorize `attach_fields`**:
  - The Python `cand_dict.get()` per pair is the big loss. Convert the
    candidate dict to a polars DataFrame once, then use `merged.join(
    cand_df, left_on="candidate_entity_id", right_on="entity_id")` —
    single vectorized join per chunk instead of 100K Python lookups.

- **Bulk rapidfuzz**:
  - `from rapidfuzz import process, fuzz`
  - `process.cdist(s1_names, cand_names, scorer=fuzz.token_set_ratio,
    workers=-1)` returns a 2D matrix — parallelized internally.

- **Persist indexes**:
  - Currently rebuilds the 3 indexes every run (~2 min × per direction).
  - Pickle them to `artifacts/_indexes/s2.pkl`, `s3.pkl` and load if
    newer than the normalized parquet. Saves ~2 min per rerun.

- **Resume v2 — better than current**:
  - Track done-chunks in a `set[str]` pickled to disk (don't re-glob on
    every run). Write chunk done-marker BEFORE writing the parquet (so a
    kill mid-write doesn't get re-skipped).

---

## 6. Recall improvement levers — push past 0.95

The blocker needs to recall more true matches from the bottom 25% of S1
(the ones at P25 = 0.5). Here are concrete ideas, roughly in order of
expected ROI:

### High-leverage (likely 5-15 pp recall gain each)

1. **Relax Floor B from `n_tokens_shared ≥ 2` to `≥ 1`** — many true
   matches differ by one token (added "Pvt" / dropped "Inc"). Single
   shared tokens + strong structural or sortedn evidence is enough.
   Try first; if precision drops too much in Phase E, undo.

2. **Add a 4th floor to the OR — Floor E: rapidfuzz name partial ≥ 80**.
   After the cheap polars filter, pairs with high string similarity
   on name alone are usually true matches. Compute partial_ratio in
   polars or pre-bucket by rapidfuzz token_set_ratio ≥ 70 on `name_clean`.

3. **Widen the sorted-neighborhood window dynamically** — `--sn-window 50`
   catches 95% per spec, but the bottom-decile S1 may have names with
   many stop-words stripped (canonical gets short, window neighbor count
   shrinks). Compute `effective_window = max(50, 50 / len(canonical))`
   to keep candidate count stable across name lengths.

4. **Add a MinHash / Jaccard inverted index** — `datasketch` is already
   in requirements.txt. Build MinHash on `(name_latin + ' ' + addr_latin)`
   shingles of size 3 chars, threshold 0.5, probe top-30 per S1. This
   catches reordered/typo names that sortedn misses when both sides have
   different token-set cardinalities. **Skip char-trigrams** (v2 showed
   they were too noisy) — MinHash is more principled.

5. **Re-add `name_ngram_key` + `addr_ngram_key` from normalized schema**
   as a 4th structural index. These are 3-word shingles already computed
   by libpostal. They're a "free" middle ground between char-trigrams
   (too noisy) and full tokens (too specific).

### Medium-leverage (likely 2-5 pp each)

6. **Bump per-S1 cap from 50 to 75 or 100** — with quality-floor OR,
   raising the cap doesn't hurt precision much. Mean cands/S1 = 85.3
   already (50+50 across two directions), but per-direction cap is
   tighter. Test 75 per-direction; if Phase E precision holds, push to 100.

7. **Add per-(country, direction) caps** — US cap can be tighter (city
   discrimination is good); India cap can be wider (transliteration noise).

8. **Token-bucket cap = 1000 (not 500)** for high-discriminator tokens
   (e.g., unique surnames, rare business words). Use IDF-weighted cap:
   `effective_cap = min(500, max(50, 5000 / df))`. Currently popular
   tokens lose recall.

9. **Add an alias / acronym match** — "IBM" vs "International Business
   Machines" never share tokens. Pre-compute a small alias dict from the
   ground-truth pairs (carefully — leak avoidance).

### Low-leverage (likely <2 pp each, but easy)

10. **Increase `addr_first_word` priority** — many businesses match only
    on street name + number; already in Index 1 but with 500-cap, may
    lose rare street names.

11. **Add `zip + state` compound** to structural keys — US zip is very
    discriminative when combined with state.

12. **Cross-script pairing** — current `cross_script_pair` is a feature,
    not an index. Build a separate inverted index for Devanagari-tokenized
    names, mapped to their Latin canonical.

### How to test recall improvements

```bash
# Smoke test on a 5% S1 slice (much faster than full)
python scripts/block_features.py --candidate-source S2 --dry-run --max-chunks 11

# Validate on 10% holdout (lazy, safe)
python scripts/validate_block_recall_lazy.py \
    --blocks artifacts/block_S2_features.parquet artifacts/block_S3_features.parquet
```

Iterate fast: dry-run 5% → measure recall on the slice → if promising,
full run on S2 (or S3 — smaller, faster).

---

## 7. Phase E (LightGBM) preview — for when recall is good enough

The model will train on the combined `block_features.parquet` with these
hyperparameters (from `STATUS.md` §Phase E):
- `objective: binary, num_leaves: 63, learning_rate: 0.05`
- `n_estimators: 1500 (early stop 50), feature_fraction: 0.8`
- `bagging_fraction: 0.8, bagging_freq: 5, lambda_l1/l2: 0.1`
- `scale_pos_weight: DYNAMIC (n_neg/n_pos after sampling), max_bin: 255`
- `n_jobs: -1` (will use all cores — but the 32 GB box is mandatory)

Sampling: all positives + 50% random + 50% hard-negatives (mined from
a quick baseline) at neg:pos 5:1 to 10:1.

**Hardware**: ≥32 GiB RAM / 16 cores / 50 GB SSD. **Don't try on
AWS t3.medium** — it will OOM. Move Phase E to a separate box.

After Phase E: per-(country, source) threshold tuning (sweep on val for
max F_0.5). Singleton detector (separate small LightGBM on aggregated
features). Graph refinement (close triangles where min(p_AB, p_BC) ≥ 0.85
with damping 0.9). Final packaging with `validate_submission.py`.

---

## 8. Quick-start for the next session

```bash
# 1. Read these in order:
cat /home/rujul/projects/a/amzn_ml/CLAUDE.md
cat /home/rujul/projects/a/amzn_ml/docs/STATUS.md
cat /home/rujul/projects/a/amzn_ml/docs/MEMORY.md    # this file

# 2. Check current state:
ls /home/rujul/projects/a/amzn_ml/code/business_entity_resolution/artifacts/
# Expect: s{1,2,3}_norm_train.parquet + block_S{2,3}_features.parquet
#         + block_features.parquet

# 3. Run baseline recall check (lazy, safe):
cd /home/rujul/projects/a/amzn_ml/code/business_entity_resolution
python3 scripts/validate_block_recall_lazy.py \
    --blocks artifacts/block_S2_features.parquet artifacts/block_S3_features.parquet

# 4. Pick a recall-improvement lever from §6 above; implement in
#    block_features.py; test on S3-only (smaller, faster) first.

# 5. Once recall ≥ 0.85, proceed to Phase E.
```

**Compute realities** (from `STATUS.md` §Compute realities):
- Blocker fits in 8 GiB; ~2.5 GB peak
- Phase E needs ≥32 GiB (use a different box)
- No GPU needed for any phase

---

## 9. Open questions / known unknowns

- **Why are US and India recall balanced at 0.68?** — suggests the
  failure mode is structural (e.g., quality-tier filtering) not
  transliteration-specific. Worth a per-(country, source) breakdown.

- **What's the recall-vs-cap curve?** — mean cands = 85.3 already (above
  50 cap due to two directions). Per-direction cap relaxation should
  be tested.

- **Is the 4-floor OR too permissive in one direction or too strict in
  another?** — would need a confusion analysis: which floor caught each
  true positive? Per-floor hit-rate would tell us.

- **Does the quality-tier filter actually help precision?** — without it,
  Phase E would see ~3× more pairs. The composite-score top-K is the
  second line of defense. Could remove the floor entirely and let LightGBM
  handle it; might be cleaner. Worth an ablation.

- **Why is the P25 = 0.5?** — the bottom quartile of S1 only recall half
  their matches. Are these the same S1s each time? Run with seed=42,
  seed=43, seed=44 and intersect the bottom-quartile S1 ids. If they
  overlap, there's a structural reason (e.g., S1 ids with completely
  re-tokenized names post-normalization).

---

## 10. Files NOT to delete (data lineage)

- `artifacts/s{1,2,3}_norm_train.parquet` — 1.3 GB total, ~22 min to
  regenerate via libpostal. Don't re-run unless schema changes.
- `artifacts/_chunks/block_S2_*.parquet` — already cleaned (the
  `block_S2_features.parquet` is the post-concat product). If you ever
  see _chunks populated, the prior run got interrupted.
- `artifacts/block_features.parquet` — combined S2+S3, 5.37 GB. Inputs
  to Phase E. Backup before re-running combine.

---

**Last updated**: 2026-09-27 (end of v3-blocker-debugging session).
**Next session should**: pick a recall lever from §6, implement + test
on S3-only, validate on combined, then plan the multi-process blocker
rewrite from §5 before committing the changes.
