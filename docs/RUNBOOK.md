# RUNBOOK

Concrete commands. Use this on every new machine or new Claude session.

## 0. First-time setup (one time per machine)

**a. System packages (Fedora):**

```bash
sudo dnf install -y autoconf automake libtool pkg-config curl git gcc make file
```

**b. libpostal (C library + 2 GB language models) into `/home/rujul/local/`:**

```bash
git clone https://github.com/openvenues/libpostal /tmp/libpostal
cd /tmp/libpostal && ./build.sh && make install
ldconfig
cd /tmp/libpostal && ./libpostal/scripts/download_libpostal_data.sh
```

The build patches autotools to use a local prefix (`/home/rujul/local/usr/...`) instead of `/usr/...` — see `setup_env.sh` to set the env vars right. This was needed because no sudo on the prior hardware.

`/home/rujul/local/` after install:
- `bin/`: patched autoconf, automake, libtool, libtoolize, autom4te
- `usr/bin/`: libpostal C tools
- `lib/`: libpostal.so, libpostal.a, libpostal.so.1, libpostal.so.1.0.1
- `include/`: libpostal.h, log/log.h
- `share/libpostal/`: 2 GB of parsed language data (this is the disk-cost part)
- `share/autoconf/`, `share/automake-1.18/`: the autotools data files

**c. Set up the env (always source this before Python):**

```bash
# /home/rujul/local/setup_env.sh content:
export PATH=/home/rujul/local/patched-bin:/home/rujul/local/usr/bin:$PATH
export PERL5LIB=/home/rujul/local/usr/share/autoconf:/home/rujul/local/usr/share/automake-1.18:/home/rujul/local/usr/lib64/perl5/vendor_perl:/home/rujul/local/usr/share/perl5/vendor_perl
export M4=/home/rujul/local/usr/bin/m4
export AUTOM4TE_CFG=/home/rujul/local/usr/share/autoconf/autom4te.cfg
export AC_MACRODIR=/home/rujul/local/usr/share/autoconf
export autom4te_perllibdir=/home/rujul/local/usr/share/autoconf
export AUTOMAKE_LIBDIR=/home/rujul/local/usr/share/automake-1.18
export AUTOCONF=/home/rujul/local/usr/bin/autoconf
export ACLOCAL=/home/rujul/local/usr/bin/aclocal
export AUTOHEADER=/home/rujul/local/usr/bin/autoheader
export AUTOM4TE=/home/rujul/local/usr/bin/autom4te.patched
export LD_LIBRARY_PATH=/home/rujul/local/lib:${LD_LIBRARY_PATH:-}
export LIBPOSTAL_DATA_DIR=/home/rujul/local/share/libpostal
export LIBPOSTAL_PREFIX=/home/rujul/local
export PKG_CONFIG_PATH=/home/rujul/local/lib/pkgconfig:${PKG_CONFIG_PATH:-}
```

`source /home/rujul/local/setup_env.sh` before any Python invocation that touches libpostal.

**d. Python packages (pip install --user):**

```bash
python3 -m pip install --user polars pandas numpy scikit-learn \
    lightgbm faiss-cpu networkx rapidfuzz datasketch pyarrow \
    unidecode regex indic-transliteration joblib tqdm
```

## 1. Each Claude session

Start the session by reading these three files in order:

1. `/home/rujul/projects/a/amzn_ml/CLAUDE.md` (you are here)
2. `/home/rujul/projects/a/amzn_ml/docs/STATUS.md` (current state)
3. `/home/rujul/projects/a/amzn_ml/docs/RUNBOOK.md` (concrete commands — this file)

Then check what's already done:

```bash
ls code/business_entity_resolution/artifacts/
# Expect: eda_*.csv, eda_report.md, s{1,2,3}_norm_train.parquet
# If blocks_*.parquet files exist too, blocking is done.
```

## 2. Phase C — blocking (current: hybrid 8-structural + char-trigram)

The original 12-key design (K1–K12 with TF-IDF + MinHash + Faiss) was
abandoned for 8 GB RAM. The current blocker
(`scripts/block_features.py`) is **streaming**, **RAM-bounded**, and
**combines blocking + feature engineering** in a single pass per chunk.
TF-IDF and MinHash are dropped; fuzzy blocking falls back to a
**char-trigram inverted index** with per-trigram cap = 100.

```bash
cd D:/Projects/Amazon_ML/amazon_ml_challenge

# S1 ↔ S2 (do this first)
python code/business_entity_resolution/scripts/block_features.py \
    --candidate-source S2

# S1 ↔ S3 (separate run; on a different machine is fine)
python code/business_entity_resolution/scripts/block_features.py \
    --candidate-source S3
```

**Defaults** (override via CLI flags):

| Flag | Default | Purpose |
| --- | --- | --- |
| `--top-k` | 25 | Final candidates per S1 after structural + trigram union |
| `--top-k-tfidf` | 50 | Trigram-blocker candidates per S1 (pre-cap) |
| `--chunk-size` | 10000 | S1 rows per chunk |
| `--bucket-cap` | 500 | Max S2 ids per structural key bucket |
| `--trigram-cap` | 100 | Max S2 ids per char-trigram bucket |

**Smoke test (5% slice, no final concat)** — quick verification:

```bash
python code/business_entity_resolution/scripts/block_features.py \
    --candidate-source S2 --dry-run --max-chunks 2
```

Expect: 2 chunks in ~50–225 s, ~500 K candidate pairs, 36-column
parquets in `artifacts/_chunks/`. No `block_S2_features.parquet`
written (dry-run skips final concat).

**Detached run** (so Claude session timeouts don't kill it):

```bash
setsid nohup bash -c "
  cd D:/Projects/Amazon_ML/amazon_ml_challenge
  python code/business_entity_resolution/scripts/block_features.py --candidate-source S2 \
    > /tmp/block_S2.log 2>&1
  echo 'EXIT_CODE='\$? >> /tmp/block_S2.log
  touch /tmp/block_S2.done
" </dev/null >/dev/null 2>&1 &
disown
echo 'Launched blocking job; tail /tmp/block_S2.log'
```

**Monitor every ~15 min** (use ScheduleWakeup if running in a Claude session):

```bash
tail -20 /tmp/block_S2.log     # progress
free -h                         # RAM check
ls -la code/business_entity_resolution/artifacts/   # output?
ls /tmp/block_S2.done           # completion sentinel?
```

**Expected timing on 8 GB RAM / 16 cores:**

- Index build (one-time): 8 structural indexes ~30 s + trigram index
  ~280 s + cand_dict ~50 s = ~6 min one-time
- Per chunk (10K S1): ~100 s on this hardware
- Total per direction: 220 chunks × 100 s + ~6 min = ~5–6 h
- RAM peak observed: ≤ 7.5 GB (safe on 8 GB box)

**Output schema** (`artifacts/block_{S2|S3}_features.parquet`, 36 cols):

```
source1_entity_id, candidate_entity_id, candidate_source,     # identity
n_struct_keys, n_trigrams, block_score,                       # blocker signals
s1_country, m__country, country_eq,
name_first_token_eq, name_token_jaccard,
name_n_chars_diff, cross_script_pair,
addr_first_word_eq, addr_last_word_eq, addr_city_eq,
addr_house_number_eq, addr_state_eq, addr_road_eq,
addr_zip_eq, addr_unit_eq, addr_suburb_eq,
s1_name_missing, m_name_missing, s1_addr_missing, m_addr_missing,
name_token_set_ratio, name_partial_ratio, name_token_sort_ratio,
name_ratio, name_latin_token_set_ratio,
addr_token_set_ratio, addr_partial_ratio, addr_token_sort_ratio,
addr_ratio, addr_latin_token_set_ratio                        # 27 features
```

## 3. Phase D — feature engineering

**Done as part of Phase C** — see §2 above. The 27 features listed in
the output schema are computed per candidate pair inside
`block_features.py` and emitted to the same parquet.

## 4. Combine S2 + S3 candidate parquets

Once both `--candidate-source S2` and `--candidate-source S3` runs
have finished, combine them vertically:

```bash
python code/business_entity_resolution/scripts/combine_block_features.py
```

This writes `artifacts/block_features.parquet` (~100 M rows × 36 cols).

## 5. Recall validation

Verify blocking recall vs `train_ground_truth.tsv` on a 10% holdout:

```bash
python code/business_entity_resolution/scripts/validate_block_recall.py
```

Expect overall recall **≥ 0.80** (we dropped TF-IDF/MinHash vs the
project's original 0.92 target). Exit code 0 on pass, 2 on fail.

## 6. Phase E — model training

```bash
cd D:/Projects/Amazon_ML/amazon_ml_challenge
setsid nohup bash -c "
  cd D:/Projects/Amazon_ML/amazon_ml_challenge
  python code/business_entity_resolution/scripts/train_classifier.py > /tmp/lgbm.log 2>&1
  echo 'EXIT_CODE='\$? >> /tmp/lgbm.log
  touch /tmp/lgbm.done
" </dev/null >/dev/null 2>&1 &
disown
python code/business_entity_resolution/scripts/singleton_detector.py  # separate model
```

## 7. Phase F — inference + threshold

Same detached pattern. Reads `block_features.parquet`, predicts on
`test_*` ONCE here. Per-(country, source) threshold τ.

## 8. Phase G — graph refinement

Detached. Operates on the predicted-pair graph from Phase F. Closes
triangles only if `min(p_AB, p_BC) ≥ 0.85` with damping 0.9.

## 9. Phase H — packaging (last)

```bash
cd D:/Projects/Amazon_ML/amazon_ml_challenge
python code/business_entity_resolution/scripts/predict.py

# Validate:
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test \
    --check-ids

# Package:
zip -r submission.zip output/ code/business_entity_resolution/ Documentation_template.md
```

`Documentation_template.md` is at the repo root. Fill from
`docs/STATUS.md` and the blocker methodology doc.

## Common pitfalls

- **Memory: do NOT spawn multiple Python workers.** The hybrid blocker
  in `scripts/block_features.py` already keeps RAM ≤ 7.5 GB. If
  running ad-hoc, keep n_jobs ≤ 4.
- **Detached: always use `setsid nohup` + log file + done sentinel.**
  The Claude session timer can otherwise kill jobs that take > 2 min.
- **Train-only: never `ls dataset/test/` or read those files** during
  development. They're read-only for Phase H.
- **Resumability: deleting intermediate files (e.g.
  `artifacts/_chunks/`) makes the next run rebuild everything.** Don't
  delete unless you have time.
