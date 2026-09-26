#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# run_blocking_v3.sh — end-to-end Phase C v3 blocking + combine + validate
#
# What this does (v3 — NO trigrams; see docs/STATUS.md "Phase C v3 LOCKED-IN"):
#
#   0. Pre-flight: check artifacts, disk, RAM
#   1. Smoke test on S2: --dry-run --max-chunks 3  (5% slice, no final concat)
#   2. (skip smoke if user passes --no-smoke)
#   3. Full S2 run (detached, monitored via /tmp/block_S2.log)
#   4. Full S3 run (detached, monitored via /tmp/block_S3.log)
#   5. Combine S2 + S3 into block_features.parquet
#   6. Validate recall vs train_ground_truth.tsv
#
# Use on AWS EC2 (t3.medium / 8 GB works for blocking).
#
# Usage:
#   bash scripts/run_blocking_v3.sh                 # run everything
#   bash scripts/run_blocking_v3.sh --no-smoke      # skip smoke test
#   bash scripts/run_blocking_v3.sh --skip-s2        # S3-only
#   bash scripts/run_blocking_v3.sh --skip-s3        # S2-only
#   bash scripts/run_blocking_v3.sh --chunk-size 5000   # lower RAM
# ------------------------------------------------------------------------------
set -euo pipefail

# ---- args ----
SMOKE=true
RUN_S2=true
RUN_S3=true
DRY=""
CHUNK_SIZE="${CHUNK_SIZE:-10000}"
SN_WINDOW="${SN_WINDOW:-50}"
TOP_K="${TOP_K:-50}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-smoke)   SMOKE=false; shift ;;
    --skip-s2)    RUN_S2=false; shift ;;
    --skip-s3)    RUN_S3=false; shift ;;
    --chunk-size) CHUNK_SIZE="$2"; shift 2 ;;
    --sn-window)  SN_WINDOW="$2"; shift 2 ;;
    --top-k)      TOP_K="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
ARTIFACTS="$ROOT/artifacts"
SCRIPTS="$ROOT/scripts"
mkdir -p "$ARTIFACTS"

echo "============================================================"
echo " Phase C v3 — BLOCKING (no trigrams) "
echo " ROOT   : $ROOT"
echo " ARTIF  : $ARTIFACTS"
echo " CHUNK  : $CHUNK_SIZE  TOP_K : $TOP_K  SN_WINDOW : $SN_WINDOW"
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
  echo "  WARN: $GT missing — validate_block_recall.py will fail at the end."
fi
FREE_GB=$(free -g | awk '/Mem:/ {print $7}')
echo "  Free RAM: ${FREE_GB} GiB (need >=6 for v3 blocking)"
if [[ "$FREE_GB" -lt 6 ]]; then
  echo "  WARN: <6 GiB free — raise --chunk-size or lower --bucket-cap/--sn-window"
fi
DISK_FREE=$(df -BG --output=avail "$ARTIFACTS" | tail -1 | tr -dc '0-9')
echo "  Disk free: ${DISK_FREE} GiB"
if [[ "$DISK_FREE" -lt 25 ]]; then
  echo "  WARN: <25 GiB free — v3 writes ~3 GB per direction + final combo"
fi
[[ -d "$ARTIFACTS/_chunks" ]] && rm -rf "$ARTIFACTS/_chunks" && echo "  Cleaned prior _chunks/"

# ---- 1. Smoke test (S2 only, dry-run, max 3 chunks) ----
if $SMOKE; then
  echo
  echo "[1] SMOKE TEST — S2 dry-run, 3 chunks only"
  python "$SCRIPTS/block_features.py" \
      --candidate-source S2 \
      --dry-run --max-chunks 3 \
      --chunk-size "$CHUNK_SIZE" --sn-window "$SN_WINDOW" --top-k "$TOP_K" \
      --suffix SMOKE
  echo "  smoke OK (no parquet written)"
  # clean chunk leftovers
  [[ -d "$ARTIFACTS/_chunks" ]] && rm -rf "$ARTIFACTS/_chunks"
fi

# ---- 2. Full S2 ----
if $RUN_S2; then
  echo
  echo "[2] FULL S2 — blocking (~50-80 min on t3.medium)"
  LOG="/tmp/block_S2.log"
  setsid nohup python "$SCRIPTS/block_features.py" \
      --candidate-source S2 \
      --chunk-size "$CHUNK_SIZE" --sn-window "$SN_WINDOW" --top-k "$TOP_K" \
      > "$LOG" 2>&1 < /dev/null &
  PID=$!
  echo "  S2 PID=$PID  log=$LOG"
  echo "  monitor: tail -f $LOG"
  # wait (foreground so we crash early if S3 follows in same script)
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
  echo "[3] FULL S3 — blocking (~50-80 min on t3.medium)"
  LOG="/tmp/block_S3.log"
  setsid nohup python "$SCRIPTS/block_features.py" \
      --candidate-source S3 \
      --chunk-size "$CHUNK_SIZE" --sn-window "$SN_WINDOW" --top-k "$TOP_K" \
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

# ---- 4. Combine ----
echo
echo "[4] COMBINE — vertical concat S2 + S3"
python "$SCRIPTS/combine_block_features.py"

# ---- 5. Validate ----
echo
echo "[5] VALIDATE — recall vs ground truth"
python "$SCRIPTS/validate_block_recall.py" \
    --blocks "$ARTIFACTS/block_S2_features.parquet" \
             "$ARTIFACTS/block_S3_features.parquet" \
    --min-recall 0.85 || {
  echo "  RECALL FAILED — see status above. Do NOT proceed to Phase E."
  exit 3
}

echo
echo "============================================================"
echo " ALL DONE — Phase C v3 complete."
echo " Recall target met. Ready for Phase E (LightGBM training)."
echo "============================================================"
