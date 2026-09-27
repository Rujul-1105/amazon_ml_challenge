#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# run_blocking_v3.sh — end-to-end Phase C v4 blocking + combine + validate
#
# What this does (M26 method: liberal_v3 floor + MinHash LSH + cap=400):
#
#   0. Pre-flight: check artifacts, disk, RAM
#   1. Smoke test on S2: --dry-run --max-chunks 3  (5% slice, no final concat)
#   2. (skip smoke if user passes --no-smoke)
#   3. Full S2 run (detached, monitored via /tmp/block_S2.log)
#   4. Full S3 run (detached, monitored via /tmp/block_S3.log)
#   5. Combine S2 + S3 into block_features.parquet (lazy streaming)
#   6. Validate recall vs train_ground_truth.tsv (lazy)
#
# Use on a box with ≥12 GiB RAM and ≥50 GiB disk (M26 cap=400 writes
# ~8× more candidates than v3).
#
# Usage:
#   bash scripts/run_blocking_v3.sh                  # run everything
#   bash scripts/run_blocking_v3.sh --no-smoke       # skip smoke test
#   bash scripts/run_blocking_v3.sh --skip-s2         # S3-only
#   bash scripts/run_blocking_v3.sh --skip-s3         # S2-only
#   bash scripts/run_blocking_v3.sh --no-minhash      # run v3 floor (no MinHash)
#   bash scripts/run_blocking_v3.sh --minhash-threshold 0.4
# ------------------------------------------------------------------------------
set -euo pipefail

# ---- args ----
SMOKE=true
RUN_S2=true
RUN_S3=true
USE_MINHASH=true
DRY=""
CHUNK_SIZE="${CHUNK_SIZE:-10000}"
SN_WINDOW="${SN_WINDOW:-50}"
TOP_K="${TOP_K:-200}"
MINHASH_THRESHOLD="${MINHASH_THRESHOLD:-0.3}"
MINHASH_WORKERS="${MINHASH_WORKERS:-4}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-smoke)   SMOKE=false; shift ;;
    --skip-s2)    RUN_S2=false; shift ;;
    --skip-s3)    RUN_S3=false; shift ;;
    --no-minhash) USE_MINHASH=false; shift ;;
    --chunk-size) CHUNK_SIZE="$2"; shift 2 ;;
    --sn-window)  SN_WINDOW="$2"; shift 2 ;;
    --top-k)      TOP_K="$2"; shift 2 ;;
    --minhash-threshold) MINHASH_THRESHOLD="$2"; shift 2 ;;
    --minhash-workers)   MINHASH_WORKERS="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
ARTIFACTS="$ROOT/artifacts"
SCRIPTS="$ROOT/scripts"
mkdir -p "$ARTIFACTS"

echo "============================================================"
echo " Phase C v4 — BLOCKING (M26: MinHash + liberal_v3 + cap=$TOP_K) "
echo " ROOT   : $ROOT"
echo " ARTIF  : $ARTIFACTS"
echo " CHUNK  : $CHUNK_SIZE  TOP_K : $TOP_K  SN_WINDOW : $SN_WINDOW"
echo " MINHASH: $USE_MINHASH (threshold=$MINHASH_THRESHOLD, workers=$MINHASH_WORKERS)"
echo "============================================================"

# ---- 0. Pre-flight ----
echo
echo "[0] Pre-flight checks"
for f in s1_norm_train.parquet s2_norm_train.parquet s3_norm_train.parquet; do
  if [[ ! -f "$ARTIFACTS/$f" ]]; then
    echo "  MISSING: $ARTIFACTS/$f — Phase B not done."
    exit 1
  fi
done
GT="$ROOT/dataset/train/train_ground_truth.tsv"
if [[ ! -f "$GT" ]]; then
  echo "  WARN: $GT missing — validate_block_recall_lazy.py will fail at the end."
fi
FREE_GB=$(free -g | awk '/Mem:/ {print $7}')
echo "  Free RAM: ${FREE_GB} GiB (need >=12 for M26 blocking w/ MinHash)"
if [[ "$FREE_GB" -lt 12 ]]; then
  echo "  WARN: <12 GiB free — raise --chunk-size or lower --bucket-cap/--sn-window"
fi
DISK_FREE=$(df -BG --output=avail "$ARTIFACTS" | tail -1 | tr -dc '0-9')
echo "  Disk free: ${DISK_FREE} GiB (need >=50 for cap=400 output)"
if [[ "$DISK_FREE" -lt 50 ]]; then
  echo "  WARN: <50 GiB free — cap=400 writes ~8x more candidates than v3"
fi
[[ -d "$ARTIFACTS/_chunks" ]] && rm -rf "$ARTIFACTS/_chunks" && echo "  Cleaned prior _chunks/"

# ---- 1. Smoke test (S2 only, dry-run, max 3 chunks) ----
if $SMOKE; then
  echo
  echo "[1] SMOKE TEST — S2 dry-run, 3 chunks only"
  MINHASH_FLAG=""
  if $USE_MINHASH; then
    MINHASH_FLAG="--minhash --minhash-threshold $MINHASH_THRESHOLD --minhash-workers $MINHASH_WORKERS"
  fi
  python "$SCRIPTS/block_features.py" \
      --candidate-source S2 \
      --dry-run --max-chunks 3 \
      --chunk-size "$CHUNK_SIZE" --sn-window "$SN_WINDOW" --top-k "$TOP_K" \
      $MINHASH_FLAG \
      --suffix SMOKE
  echo "  smoke OK (no parquet written)"
  # clean chunk leftovers
  [[ -d "$ARTIFACTS/_chunks" ]] && rm -rf "$ARTIFACTS/_chunks"
fi

# ---- 2. Full S2 ----
if $RUN_S2; then
  echo
  echo "[2] FULL S2 — blocking (~30-60 min with MinHash)"
  LOG="/tmp/block_S2.log"
  MINHASH_FLAG=""
  if $USE_MINHASH; then
    MINHASH_FLAG="--minhash --minhash-threshold $MINHASH_THRESHOLD --minhash-workers $MINHASH_WORKERS"
  fi
  setsid nohup python "$SCRIPTS/block_features.py" \
      --candidate-source S2 \
      --chunk-size "$CHUNK_SIZE" --sn-window "$SN_WINDOW" --top-k "$TOP_K" \
      $MINHASH_FLAG \
      > "$LOG" 2>&1 < /dev/null &
  PID=$!
  echo "  S2 PID=$PID  log=$LOG"
  echo "  monitor: tail -f $LOG"
  if wait $PID; then
    echo "  S2 done. parquet: $ARTIFACTS/block_S2_features.parquet"
  else
    echo "  S2 FAILED — see $LOG"
    exit 2
  fi
fi

# ---- 3. Full S3 ----
if $RUN_S3; then
  echo
  echo "[3] FULL S3 — blocking (~30-60 min with MinHash)"
  LOG="/tmp/block_S3.log"
  MINHASH_FLAG=""
  if $USE_MINHASH; then
    MINHASH_FLAG="--minhash --minhash-threshold $MINHASH_THRESHOLD --minhash-workers $MINHASH_WORKERS"
  fi
  setsid nohup python "$SCRIPTS/block_features.py" \
      --candidate-source S3 \
      --chunk-size "$CHUNK_SIZE" --sn-window "$SN_WINDOW" --top-k "$TOP_K" \
      $MINHASH_FLAG \
      > "$LOG" 2>&1 < /dev/null &
  PID=$!
  echo "  S3 PID=$PID  log=$LOG"
  echo "  monitor: tail -f $LOG"
  if wait $PID; then
    echo "  S3 done. parquet: $ARTIFACTS/block_S3_features.parquet"
  else
    echo "  S3 FAILED — see $LOG"
    exit 2
  fi
fi

# ---- 4. Combine (lazy streaming — RAM-safe at cap=400 output) ----
echo
echo "[4] COMBINE — lazy concat of S2 + S3 (streaming, zstd)"
python "$SCRIPTS/combine_block_features_lazy.py"

# ---- 5. Validate (lazy streaming — RAM-safe at 100 M+ rows) ----
echo
echo "[5] VALIDATE — recall vs ground truth (lazy streaming)"
python "$SCRIPTS/validate_block_recall_lazy.py" \
    --blocks "$ARTIFACTS/block_S2_features.parquet" \
             "$ARTIFACTS/block_S3_features.parquet" \
    --min-recall 0.90 || {
  echo "  RECALL BELOW 0.90 — see status above. Do NOT proceed to Phase E without review."
  exit 3
}

echo
echo "============================================================"
echo " ALL DONE — Phase C v4 (M26 method) complete."
echo " Recall >= 0.90. Ready for Phase E (LightGBM training)."
echo "============================================================"
