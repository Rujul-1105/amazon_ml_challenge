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

## 2. Phase C — blocking (resume here)

```bash
cd /home/rujul/projects/a/amzn_ml/code/business_entity_resolution

# Always source the env first
source /home/rujul/local/setup_env.sh
export PYTHONPATH=$(pwd)

# Detached so a Claude session timeout can't kill it (per §0.2 in plan)
setsid nohup bash -c "
  source /home/rujul/local/setup_env.sh
  export PYTHONPATH=$(pwd)
  python3 -m src.blocking > /tmp/blocking.log 2>&1
  echo 'EXIT_CODE='\$? >> /tmp/blocking.log
  touch /tmp/blocking.done
" </dev/null >/dev/null 2>&1 &
disown
echo "Launched detached blocking job"
```

**Monitor every 10 min** (use `ScheduleWakeup` if running in a Claude session):

```bash
tail -20 /tmp/blocking.log      # progress
free -h                         # RAM check
ls -la artifacts/              # output parquets?
ls /tmp/blocking.done           # completion sentinel?
```

Expected timing on 18-core/16 GB:
- US (3M S1 + 6M candidates): ~30 min
- India (1.7M S1 + 4M candidates): ~25 min
- Total: ~60–90 min

**Tuning for 16 GB** (RAM-aware workers, current code auto-adjusts):

```python
# In src/blocking.py, _read_free_gb() function:
# Bump these thresholds by ~1 GB to take advantage of the new box
if free_gb < 5.0:
    n_features = 1 << 12  # 4096 — was 1024 in old box
elif free_gb < 8.0:
    n_features = 1 << 14  # 16384 — was 4096
else:
    n_features = 1 << 16  # 65536 — original target (full recall)

# MinHash perm count
num_perm = 128 if free_gb > 8.0 else 64  # was 6.0 — bump floor by 2 GB
```

**Validation once `.done` appears:**

```bash
source /home/rujul/local/setup_env.sh
export PYTHONPATH=$(pwd)
python3 -m src.validate_blocking
```

Expect: `Overall recall on held-out slice: >= 0.92` with per-bucket breakdown.

## 3. Phase D — feature engineering (after C2)

```bash
setsid nohup bash -c "source /home/rujul/local/setup_env.sh && \
  export PYTHONPATH=$(pwd) && \
  python3 -m src.features > /tmp/features.log 2>&1 && \
  touch /tmp/features.done" </dev/null >/dev/null 2>&1 &
disown
```

## 4. Phase E — model training

```bash
setsid nohup bash -c "source /home/rujul/local/setup_env.sh && \
  export PYTHONPATH=$(pwd) && \
  python3 -m src.train_classifier > /tmp/lgbm.log 2>&1 && \
  touch /tmp/lgbm.done" </dev/null >/dev/null 2>&1 &
disown
python3 -m src.singleton_detector  # separate model
```

## 5. Phase F — inference + threshold

Same detached pattern. Reads `train_*` parquets, predicts on `test_*` ONCE here.

## 6. Phase G — graph refinement

Detached. Operates on the predicted-pair graph from Phase F.

## 7. Phase H — packaging (last)

```bash
python3 -m src.predict    # generate output/matching_results.tsv, output/candidate_pairs.tsv

# Validate:
python3 data_set/student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir data_set/student_resource/dataset/test \
    --check-ids

# Package:
zip -r submission.zip output/ code/business_entity_resolution/ \
    data_set/student_resource/Documentation_template.md
```

`Documentation_template.md` is in `data_set/student_resource/`. Fill it from `docs/STATUS.md` + methodology in `code/business_entity_resolution/docs/phase_c_blocking.md`.

## Common pitfalls

- **Memory: do NOT spawn multiple Python workers.** Auto-cap from `/proc/meminfo` is in `src/normalize.py` and `src/blocking.py`. If running ad-hoc, keep n_jobs ≤4.
- **Detached: always use `setsid nohup` + log file + done sentinel.** The Claude session timer can otherwise kill jobs that take >2 min.
- **Train-only: never `ls data_set/student_resource/dataset/test/` or read those files** during development. They're read-only for Phase H.
- **libpostal env: `source /home/rujul/local/setup_env.sh` every shell** before running Python that touches libpostal. The patches are path-based.
- **Resumability: deleting intermediate files (e.g. `_blocking_idx/`) makes `blocking.py` rebuild.** Don't delete unless you have time.
