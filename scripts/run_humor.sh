#!/bin/bash
set -e

RUNS="${RUNS:-1}"
DATASET="${DATASET:-humor_coid}"
DEVICE="${DEVICE:-cuda}"

for i in $(seq 1 "$RUNS"); do
  python code/train.py \
    --backbone="${BACKBONE:-mmgcn}" \
    --name="dnr_${DATASET}_run${i}" \
    --hidden_dim="${HIDDEN_DIM:-256}" \
    --learning_rate="${LR:-0.0001}" \
    --dataset="$DATASET" \
    --modalities="${MODALITIES:-atv}" \
    --batch_size="${BATCH_SIZE:-32}" \
    --epochs="${EPOCHS:-60}" \
    --seed="${SEED:-50}" \
    --drop_rate="${DROP_RATE:-0.4}" \
    --early_stopping="${EARLY_STOPPING:-20}" \
    --pretrain_epochs="${PRETRAIN_EPOCHS:-200}" \
    --encoder_modules=transformer \
    --project_name="${PROJECT_NAME:-dnr_affect}" \
    --device="$DEVICE" \
    --use_divide \
    --use_refine
done
