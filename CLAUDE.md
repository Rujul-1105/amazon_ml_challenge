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
| B — normalization (libpostal `parse_address`) | ✅ done | `code/business_entity_resolution/artifacts/s{1,2,3}_norm_train.parquet` (~1.3 GB) |
| **C — blocking** | ⏸ **resume here** | missing 4 `artifacts/blocks_country=*_source=*.parquet` files |
| D — feature engineering | ⏸ pending | — |
| E — LightGBM classifier + singleton detector | ⏸ pending | — |
| F — inference + threshold | ⏸ pending | — |
| G — graph refinement | ⏸ pending | — |
| H — packaging (submission zip) | ⏸ pending | — |

See `docs/STATUS.md` for detailed state of each phase, what failed and why, and what to retry with new hardware.

## How to start work

1. **Read** `docs/STATUS.md` (current state + known failures).
2. **Read** `docs/RUNBOOK.md` (concrete commands).
3. **Skim** the plan at `~/.claude/plans/go-through-the-problem-bright-cosmos.md` — it's the authoritative spec for all phases.
4. **Check** Phase C state: `ls code/business_entity_resolution/artifacts/`
   - If `s{1,2,3}_norm_train.parquet` exist → proceed to Phase C
   - If `blocks_country=*_source=*.parquet` exist → blocking is done, run Phase C2 validation

## Compute realities

- **Hardware target:** 18-core CPU + ≥16 GB RAM + ≥144 GB disk.
- **No GPU needed.** This is a CPU-bound stack — see "Why CPU-bound" section below.
- **No worker-heavy parallelism.** Each Python+heavy-lib worker uses ~1.0–1.5 GB peak. Cap at 4 workers on 16 GB; never exceed box's free RAM / 1.5 GB workers.
- **Detached execution** for any stage > 5 min: use `setsid nohup bash -c "..." > /tmp/<stage>.log 2>&1 &` so Claude session timeouts don't kill the job.
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
