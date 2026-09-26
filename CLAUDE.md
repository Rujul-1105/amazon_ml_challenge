# CLAUDE.md — Project entry point

**Read this file FIRST** at the start of any new Claude session on this project.
It tells you what we're building, where we are, and what's next.

## What this is

Amazon ML Challenge 2026 — Business Entity Resolution.

- **Input:** 3 noisy sources of business records (S1 deduplicated reference, S2, S3) over US, India, (test-only) France. Source files live in `data_set/student_resource/dataset/{train,test}/`.
- **Task:** for every S1 entity, predict the set of matching S2/S3 entity_ids.
- **Metric:** macro-averaged **F_0.5** (precision 2× recall). Singletons matter: correct empty = 1.0, false-positive = 0.0.
- **Hard rules:** no external data lookups; final model ≤8B params MIT/Apache; output must be UTF-8 tab-separated.

Full problem statement: `data_set/student_resource/README.md` (canonical) and `ps.txt` (mirror).

## Where we are

| Phase | Status | Output |
|---|---|---|
| A — env / packages | ✅ done | — |
| B — normalization (libpostal `parse_address`) | ✅ done | `code/business_entity_resolution/artifacts/s{1,2,3}_norm_train.parquet` (~1.3 GB total) |
| **C — blocking (v3 design locked-in, S2 ran with v2)** | 🔄 **v2 S2 produced; rebuilding with v3** | v2 output: `artifacts/block_S2_features.parquet` (1.55 GB). v3 will overwrite. |
| **D — feature engineering** | ✅ done | 27 features per candidate pair |
| E — LightGBM classifier + singleton detector | ⏸ pending | needs Phase C v3 done (recall must be ≥ 0.80 before training) |
| F — inference + threshold | ⏸ pending | depends on Phase E |
| G — graph refinement | ⏸ pending | depends on Phase F |
| H — packaging (submission zip) | ⏸ pending | depends on Phase G |

**v3 blocker design — REPLACE the current `block_features.py`.** The v2 blocker (8 cheap structural keys + char-trigram inverted index) hit only **20.4% mean recall** on the train ground truth (median 0%; 50.7% of S1 entities had ZERO true matches recalled). v3 fix is locked-in — see `docs/STATUS.md` §Phase C v3 and the comments in `scripts/block_features.py`.

**Hard constraint recap (from user feedback)**:
1. Country always same — hard pre-filter before any candidate survives.
2. Stop-word + common-business-word list filters Index 2 token inverted index.
3. Index 2 (tokens) and Index 3 (sorted-token neighborhood) both kept — overlap but each adds value.
4. Per-S1 strict **50-candidate cap** (≤ 220 M rows total combined).
5. Quality-tier floor is an **OR** of 4 hard conjunctions (one passing = keep the pair).
6. Structural score is capped at `min(n_struct_keys, 3) / 3` (city/state/country trivially match; only road/house/name_fw/compounds are informative).

## How to start work

1. **Read** `docs/STATUS.md` — current state per phase, v3 design spec, hard-fail bugs.
2. **Read** `docs/RUNBOOK.md` — concrete commands (block, combine, train, predict, validate).
3. **Skim** `~/.claude/plans/deep-gliding-wombat.md` — full v3 plan with weights/ thresholds/ floors.
4. **Check** Phase C state: `ls code/business_entity_resolution/artifacts/`
   - If `s{1,2,3}_norm_train.parquet` exist but no `block_*_features.parquet` exist → run Phase C v3.
   - If `block_S{2,3}_features.parquet` both exist → proceed to Phase E.
   - **NOTE**: v2 S2 ran but produced 20% recall; v3 will overwrite it.

## Compute realities — confirmed working sizes

| Stage | Min CPU | Min RAM | Disk | Wall-clock |
| --- | --- | --- | --- | --- |
| v3 blocking (S2 or S3) | 2 vCPU | **8 GiB** | 30 GB | ~30-50 min / direction on 2 vCPU; ~20-30 min on 4 vCPU |
| Phase E (LightGBM training) | 8 cores | 32 GiB | 50 GB | 3-5 h |
| Phase F (test inference) | same as Phase C v3 | same | +1.5 GB output | 5-10 min (after test blocker cached) |

- **v3 blocking fits in 8 GB RAM**, peak ~2.5 GB. **AWS t3.medium (2 vCPU / 8 GiB)** works — expect ~50-80 min for one direction.
- **Phase E training needs ≥32 GiB RAM** (LightGBM histogram). Don't try to train on AWS t3.medium.
- **No GPU needed** — CPU-only stack.
- **Test blocker caching**: Phase F writes `block_S{2,3}_TEST_features.parquet` once (~50 min); every subsequent model iteration on test data is then 5-10 min (load + predict only).
- **Detached execution** for any stage > 5 min: `setsid nohup bash -c "..." > /tmp/<stage>.log 2>&1 &` so Claude session timeouts don't kill the job.
- **Source-of-truth data:** train files only during development. Test files touched only at Phase H.

## Code layout

```
amzn_ml/
├── data_set/student_resource/    # raw TSVs (TRAIN — read freely during dev)
│                                 # TEST files are in dataset/test/ — DO NOT READ during dev
├── code/business_entity_resolution/     # the package
│   ├── src/                    # pipeline code (one module per phase)
│   │   ├── config.py          # paths + hyper-params (TRAIN_ONLY = True)
│   │   ├── io_utils.py        # polars TSV/parquet loaders
│   │   ├── f05.py             # macro F_0.5 scorer
│   │   ├── eda.py             # EDA (already executed, CSVs in artifacts/)
│   │   ├── normalize.py       # libpostal parse_address; (B done)
│   │   ├── transliterate.py   # Devanagari ↔ Latin (utility used by normalize)
│   │   ├── blocking.py        # ⏸ resume here; sequential-by-stage build pattern
│   │   └── validate_blocking.py  # recall validation for Phase C
│   ├── artifacts/             # phase outputs (parquets, reports)
│   ├── docs/phase_c_blocking.md  # methodology rationale for the 12 blocking keys
│   ├── requirements.txt
│   └── output/                # (empty until Phase H)
├── docs/                       # status + runbook for new sessions
├── output/                     # (empty until Phase H)
└── CLAUDE.md                   # ← you are here
```

## Memory pointers

`~/.claude/projects/.../memory/` carries session-spanning facts:
- `amzn-ml-challenge-overview.md` — problem summary
- `train-only-dev-policy.md` — never read test files in dev
- `new-hardware-phase-c-handoff.md` — what's done, what failed, where to resume

## Conventions (across every phase)

1. **RAM-aware workers** — auto-cap from `/proc/meminfo`; n_workers = clamp(int((free_gb - 6.0) / 1.5), 1, 4).
2. **Detached runners** — `setsid nohup bash -c "..." > /tmp/<stage>.log 2>&1; touch /tmp/<stage>.done`.
3. **Resumability** — each stage's driver checks `if output_path.exists(): skip`.
4. **Train-only** — never read `dataset/test/*`; only at Phase H inference.
5. **Env** — `source /home/rujul/local/setup_env.sh` before any Python run.

## Why CPU-bound (and why the GPU is useless here)

This stack has **no neural networks, no embedding models, no GPU-targeted libraries**. The pipeline is exactly:

| Stage | Operations | Hardware |
|---|---|---|
| libpostal `parse_address` | CRF inference (libpostal-core.so) | CPU (no CUDA path) |
| TF-IDF (sklearn HashingVectorizer) | sparse matrix multiply (scipy) | CPU + numpy multithreading |
| Faiss-cpu IndexIVFFlat | IVF bucket lookup + distance | CPU only |
| datasketch MinHash / MinHashLSH | shingle hashing + LSH bandit | CPU only |
| rapidfuzz QRatio / Levenshtein | char-level edit-distance | CPU only |
| polars dataframe operations | filter / group / join | CPU + native SIMD |
| Phase E: LightGBM classifier | histogram splits, GBDT | CPU (`n_jobs=18`) |

Each is a multi-threaded C/Cython/Rust library that already saturates ≥16 cores on its dataset size. LightGBM at `n_jobs=18` on 16 GB RAM uses ~1.2 GB peak with sparse-binning. Adding GPU support to any of these either:

- **Doesn't exist** (libpostal, datasketch have no CUDA paths),
- **Doesn't apply** (sklearn HashingVectorizer has no GPU fork),
- **Or hurts** (LightGBM-GPU adds compile-time CUDA deps but, for our small dataset, **doesn't outperform CPU** — CPU is faster when dataset fits in L3 cache and n_jobs ≤ physical cores).

GPU + 4 GB VRAM also has a **capacity** problem: a typical embedding model (sentence-transformers/all-MiniLM-L6-v2) uses 0.5 GB params + 1 GB activation + batched inference buffers ≥ 2 GB — leaving no headroom for our other pipeline stages. We'd pay VRAM contention cost without getting recall-quality wins, because:

1. Our text fields are **short** (mean 25 chars name, 50 chars address). The cheap path is char n-gram TF-IDF + HashingVectorizer which already captures typos and abbreviations.
2. Match pairs are dominated by **clean exact-token overlaps** ("Apex Inc" ↔ "Apex Ltd" — first-word match), not dense semantic similarity.
3. Embeddings help when fields are **long descriptive text** (think: news articles, product reviews). Our fields are **structured name/address** where exact tokens matter most.

**Bottom line:** for this dataset, ER recall ceiling is determined by char n-gram TF-IDF + structured exact-match keys, not by semantic embeddings. Adding a GPU buys nothing measurable and adds operational cost. If we ever moved to long-text domains, we'd revisit.
