#!/bin/bash
set -euo pipefail

SEEDS=(301 302 402 501 502)
DEVICE="${DEVICE:-cuda}"
LOG_DIR="logs/nonconv_validation_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "Logs: $LOG_DIR"
echo "seed,dataset,scenario,best_dev_f1,best_test_f1,best_test_acc" \
  > "logs/summary.csv"

run_exp () {
  local dataset="$1"
  local scenario="$2"   # raw | refine
  local seed="$3"
  local log="${LOG_DIR}/${dataset}_${scenario}_seed${seed}.log"

  echo "=== dataset=${dataset} scenario=${scenario} seed=${seed} ==="

  local EXTRA_ARGS=()
  if [[ "$scenario" == "refine" ]]; then
    EXTRA_ARGS=(--use_divide --use_refine --pretrain_epochs=100)
  fi

  python code/train.py \
    --backbone=simple \
    --name="${dataset}_simple_${scenario}_seed${seed}" \
    --hidden_dim=128 \
    --learning_rate=0.0005 \
    --dataset="$dataset" \
    --modalities=atv \
    --batch_size=128 \
    --epochs=50 \
    --seed="$seed" \
    --drop_rate=0.3 \
    --early_stopping=15 \
    --encoder_modules=transformer \
    --project_name=nonconv_validation \
    --device="$DEVICE" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$log"

  dev_f1=$(grep  "Best dev F1:"  "$log" | tail -1 | awk '{print $NF}')
  test_f1=$(grep "Best test F1:" "$log" | tail -1 | awk '{print $NF}')
  test_acc=$(grep "Best test Acc:" "$log" | tail -1 | awk '{print $NF}')
  echo "${seed},${dataset},${scenario},${dev_f1},${test_f1},${test_acc}" \
    >> "${LOG_DIR}/summary.csv"
}

# Datasets to validate
NON_CONV_DATASETS=( "humor_coid" "sarcasm_coid")

for dataset in "${NON_CONV_DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_exp "$dataset" raw    "$seed"
    run_exp "$dataset" refine "$seed"
  done
done

echo ""
echo "===== FINAL SUMMARY ====="
cat "${LOG_DIR}/summary.csv"

# Compute average per dataset+scenario
python3 - <<'PYEOF'
import csv, sys
from collections import defaultdict

rows = list(csv.DictReader(open("${LOG_DIR}/summary.csv")))
groups = defaultdict(list)
for r in rows:
    key = (r["dataset"], r["scenario"])
    try:
        groups[key].append(float(r["best_test_f1"]))
    except:
        pass

print("\nAverage test F1:")
print(f"{'dataset':<20} {'scenario':<10} {'avg_test_f1':<12} {'n_seeds'}")
for (dataset, scenario), vals in sorted(groups.items()):
    print(f"{dataset:<20} {scenario:<10} {sum(vals)/len(vals):.4f}       {len(vals)}")
PYEOF