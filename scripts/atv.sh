#!/bin/bash
set -euo pipefail

# =========================
# Common config
# =========================
BACKBONE="${BACKBONE:-mmgcn}"
DATASET="${DATASET:-iemocap_coid}"
MODALITIES="${MODALITIES:-atv}"
DEVICE="${DEVICE:-cuda}"

HIDDEN_DIM="${HIDDEN_DIM:-256}"
LR="${LR:-0.0001}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-40}"
EARLY_STOPPING="${EARLY_STOPPING:-20}"
DROP_RATE="${DROP_RATE:-0.3}"
ENCODER_MODULES="${ENCODER_MODULES:-transformer}"
PROJECT_NAME="${PROJECT_NAME:-backbone_coid_new}"

# Use paired seeds for fair comparison
SEEDS=(${SEEDS:-301})

# Optional: only used in refine setting
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-300}"

# Log dir
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/${DATASET}_${BACKBONE}_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "Logs will be saved to: $LOG_DIR"
echo "Seeds: ${SEEDS[*]}"

# =========================
# Helper: run one experiment
# =========================
run_exp () {
  local scenario="$1"   # raw | refine
  local seed="$2"

  local run_name="${DATASET}_${BACKBONE}_${scenario}_seed${seed}"
  local log_file="${LOG_DIR}/${run_name}.log"

  echo "============================================================"
  echo "Running scenario=${scenario} seed=${seed}"
  echo "Run name: ${run_name}"
  echo "Log file: ${log_file}"
  echo "============================================================"

  if [[ "$scenario" == "refine" ]]; then
    python code/train.py \
      --backbone="$BACKBONE" \
      --name="$run_name" \
      --hidden_dim="$HIDDEN_DIM" \
      --learning_rate="$LR" \
      --dataset="$DATASET" \
      --modalities="$MODALITIES" \
      --batch_size="$BATCH_SIZE" \
      --epochs="$EPOCHS" \
      --seed="$seed" \
      --drop_rate="$DROP_RATE" \
      --early_stopping="$EARLY_STOPPING" \
      --encoder_modules="$ENCODER_MODULES" \
      --project_name="$PROJECT_NAME" \
      --device="$DEVICE" \
      --pretrain_epochs="$PRETRAIN_EPOCHS" \
      --use_divide \
      --use_refine \
      2>&1 | tee "$log_file"

  elif [[ "$scenario" == "raw" ]]; then
    python code/train.py \
      --backbone="$BACKBONE" \
      --name="$run_name" \
      --hidden_dim="$HIDDEN_DIM" \
      --learning_rate="$LR" \
      --dataset="$DATASET" \
      --modalities="$MODALITIES" \
      --batch_size="$BATCH_SIZE" \
      --epochs="$EPOCHS" \
      --seed="$seed" \
      --drop_rate="$DROP_RATE" \
      --early_stopping="$EARLY_STOPPING" \
      --encoder_modules="$ENCODER_MODULES" \
      --project_name="$PROJECT_NAME" \
      --device="$DEVICE" \
      2>&1 | tee "$log_file"

  else
    echo "Unknown scenario: $scenario"
    exit 1
  fi
}

# =========================
# Run paired comparison
# =========================
# For each seed:
#   1) raw backbone
#   2) refine + divide
# This makes comparison easier because each pair shares the same seed.
for seed in "${SEEDS[@]}"; do
  run_exp raw "$seed"
  run_exp refine "$seed"
done

echo "All runs finished."
echo "Logs saved under: $LOG_DIR"