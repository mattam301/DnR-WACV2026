#!/bin/bash
set -e

# ──────────────────────────────────────────────────────────────────────
# run_mosi_mosei_table.sh
#
# Runs all (dataset × backbone × modality × method) combinations for
# MOSI and MOSEI, collects Acc / W-F1, and dumps results to JSON + CSV
# ready for the LaTeX table.
# ──────────────────────────────────────────────────────────────────────

RUNS="${RUNS:-1}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-50}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
LR="${LR:-0.0005}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-60}"
DROP_RATE="${DROP_RATE:-0.6}"
EARLY_STOPPING="${EARLY_STOPPING:-20}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-150}"
PROJECT_NAME="${PROJECT_NAME:-dnr_affect}"

DATASETS=("mosi_coid" "mosei_coid")
BACKBONES=("simple" "mmgcn" "dialogue_gcn" "mm_dfn")
MODALITIES_LIST=("atv" "av" "at" "tv")
METHODS=("baseline" "dnr")

RESULT_DIR="results_tables"
mkdir -p "$RESULT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_JSON="${RESULT_DIR}/results_${TIMESTAMP}.json"
RESULT_CSV="${RESULT_DIR}/results_${TIMESTAMP}.csv"
RESULT_LATEX="${RESULT_DIR}/results_${TIMESTAMP}.tex"
LOG_DIR="${RESULT_DIR}/logs_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

# ── Initialise JSON ──────────────────────────────────────────────────
echo "{}" > "$RESULT_JSON"

# ── Initialise CSV ───────────────────────────────────────────────────
echo "dataset,backbone,modalities,method,run,acc,wf1" > "$RESULT_CSV"

# ──────────────────────────────────────────────────────────────────────
# Helper: parse Acc and W-F1 from the training script stdout.
#
# The train.py prints lines like:
#   Best test F1: 0.7812
#   Best test Acc: 0.7934
#
# We capture these from the log file.
# ──────────────────────────────────────────────────────────────────────
parse_results() {
    local logfile="$1"
    # Extract last occurrence (in case of multiple prints)
    local acc=$(grep -oP '(?<=Best test Acc: )[0-9.]+' "$logfile" | tail -1)
    local f1=$(grep -oP '(?<=Best test F1: )[0-9.]+' "$logfile" | tail -1)
    
    if [ -z "$acc" ]; then acc="0.0"; fi
    if [ -z "$f1" ]; then f1="0.0"; fi
    
    echo "$acc $f1"
}

# ──────────────────────────────────────────────────────────────────────
# Helper: compute mean across runs using awk
# ──────────────────────────────────────────────────────────────────────
compute_mean() {
    # Reads values from stdin, one per line, prints mean
    awk '{ s += $1; n++ } END { if (n>0) printf "%.4f", s/n; else print "0.0" }'
}

# ──────────────────────────────────────────────────────────────────────
# Main experiment loop
# ──────────────────────────────────────────────────────────────────────
echo "========================================================================"
echo " Starting experiments: ${#DATASETS[@]} datasets × ${#BACKBONES[@]} backbones"
echo "   × ${#MODALITIES_LIST[@]} modalities × ${#METHODS[@]} methods × ${RUNS} runs"
TOTAL=$(( ${#DATASETS[@]} * ${#BACKBONES[@]} * ${#MODALITIES_LIST[@]} * ${#METHODS[@]} * RUNS ))
echo " Total runs: $TOTAL"
echo "========================================================================"

COUNTER=0

for DATASET in "${DATASETS[@]}"; do
    for BACKBONE in "${BACKBONES[@]}"; do
        for MODALITIES in "${MODALITIES_LIST[@]}"; do
            for METHOD in "${METHODS[@]}"; do

                ACC_VALUES=()
                F1_VALUES=()

                for RUN_ID in $(seq 1 "$RUNS"); do
                    COUNTER=$((COUNTER + 1))
                    
                    EXP_NAME="${METHOD}_${DATASET}_${BACKBONE}_${MODALITIES}_run${RUN_ID}"
                    LOGFILE="${LOG_DIR}/${EXP_NAME}.log"

                    echo ""
                    echo "────────────────────────────────────────────────────────"
                    echo " [$COUNTER/$TOTAL] $EXP_NAME"
                    echo "────────────────────────────────────────────────────────"

                    # Build the command
                    CMD=(
                        python code/train.py
                        --backbone="$BACKBONE"
                        --name="$EXP_NAME"
                        --hidden_dim="$HIDDEN_DIM"
                        --learning_rate="$LR"
                        --dataset="$DATASET"
                        --modalities="$MODALITIES"
                        --batch_size="$BATCH_SIZE"
                        --epochs="$EPOCHS"
                        --seed="$SEED"
                        --drop_rate="$DROP_RATE"
                        --early_stopping="$EARLY_STOPPING"
                        --pretrain_epochs="$PRETRAIN_EPOCHS"
                        --encoder_modules=transformer
                        --project_name="$PROJECT_NAME"
                        --device="$DEVICE"
                    )

                    # Add DnR flags for the dnr method
                    if [ "$METHOD" == "dnr" ]; then
                        CMD+=(--use_divide --use_refine)
                    fi

                    echo "Running: ${CMD[*]}"
                    echo ""

                    # Run and tee to log
                    if "${CMD[@]}" 2>&1 | tee "$LOGFILE"; then
                        # Parse results
                        read -r ACC F1 <<< "$(parse_results "$LOGFILE")"
                        echo ""
                        echo "  ✅ $EXP_NAME → Acc=$ACC, W-F1=$F1"
                    else
                        echo ""
                        echo "  ❌ $EXP_NAME FAILED — setting Acc=0, W-F1=0"
                        ACC="0.0"
                        F1="0.0"
                    fi

                    ACC_VALUES+=("$ACC")
                    F1_VALUES+=("$F1")

                    # Append to CSV
                    echo "${DATASET},${BACKBONE},${MODALITIES},${METHOD},${RUN_ID},${ACC},${F1}" >> "$RESULT_CSV"

                done  # runs

                # ── Compute mean across runs ─────────────────────────
                MEAN_ACC=$(printf '%s\n' "${ACC_VALUES[@]}" | compute_mean)
                MEAN_F1=$(printf '%s\n' "${F1_VALUES[@]}" | compute_mean)

                # Convert to percentage (×100) for the LaTeX table
                MEAN_ACC_PCT=$(echo "$MEAN_ACC" | awk '{ printf "%.2f", $1 * 100 }')
                MEAN_F1_PCT=$(echo "$MEAN_F1" | awk '{ printf "%.2f", $1 * 100 }')

                echo ""
                echo "  📊 MEAN for $METHOD | $DATASET | $BACKBONE | $MODALITIES:"
                echo "     Acc = $MEAN_ACC_PCT%  |  W-F1 = $MEAN_F1_PCT%"

                # Append mean row to CSV
                echo "${DATASET},${BACKBONE},${MODALITIES},${METHOD},mean,${MEAN_ACC_PCT},${MEAN_F1_PCT}" >> "$RESULT_CSV"

                # ── Update JSON (using python for robustness) ────────
                python3 -c "
import json, sys

jpath = '$RESULT_JSON'
with open(jpath, 'r') as f:
    data = json.load(f)

key = '${DATASET}|${BACKBONE}|${MODALITIES}|${METHOD}'
data[key] = {
    'dataset':    '${DATASET}',
    'backbone':   '${BACKBONE}',
    'modalities': '${MODALITIES}',
    'method':     '${METHOD}',
    'runs':       ${RUNS},
    'acc_pct':    ${MEAN_ACC_PCT},
    'wf1_pct':    ${MEAN_F1_PCT},
    'acc_raw':    [$(IFS=,; echo "${ACC_VALUES[*]}")],
    'f1_raw':     [$(IFS=,; echo "${F1_VALUES[*]}")],
}

with open(jpath, 'w') as f:
    json.dump(data, f, indent=2)
"

            done  # methods
        done  # modalities
    done  # backbones
done  # datasets

# ──────────────────────────────────────────────────────────────────────
# Generate LaTeX tables from the collected JSON
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "========================================================================"
echo " Generating LaTeX tables..."
echo "========================================================================"

python3 << 'PYEOF'
import json
import sys

RESULT_JSON = "__RESULT_JSON__"
RESULT_LATEX = "__RESULT_LATEX__"

with open(RESULT_JSON, 'r') as f:
    data = json.load(f)

# Map backbone CLI names to display names
BACKBONE_DISPLAY = {
    "mmgcn":        "MMGCN",
    "dialogue_gcn": "DialogueGCN",
    "mm_dfn":       "MM-DFN",
    "simple":       "SimpleBackbone",
}

DATASET_DISPLAY = {
    "mosi_coid": "MOSI",
    "mosei_coid": "MOSEI",
}

MODALITIES_ORDER = ["atv", "av", "at", "tv"]
BACKBONES_ORDER  = ["mmgcn", "dialogue_gcn", "mm_dfn", "simple"]
DATASETS_ORDER   = ["mosi_coid", "mosei_coid"]

def lookup(dataset, backbone, modalities, method):
    key = f"{dataset}|{backbone}|{modalities}|{method}"
    entry = data.get(key, None)
    if entry is None:
        return None, None
    return entry["acc_pct"], entry["wf1_pct"]

def fmt_val(val):
    if val is None:
        return "-"
    return f"{val:.2f}"

def fmt_delta(base, improved):
    if base is None or improved is None:
        return ""
    delta = improved - base
    sign = "+" if delta >= 0 else ""
    return f" (\\( {sign}{delta:.2f}\\))"

latex_lines = []

for ds in DATASETS_ORDER:
    ds_display = DATASET_DISPLAY.get(ds, ds)

    latex_lines.append(r"\begin{table*}[!htbp]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{Performance comparison on " + ds_display + r" dataset.}")
    latex_lines.append(r"\resizebox{\textwidth}{!}{%")
    latex_lines.append(r"\begin{tabular}{lcccccccc}")
    latex_lines.append(r"\toprule")
    latex_lines.append(r"\multirow{2}{*}{Backbone} & \multicolumn{2}{c}{atv} & \multicolumn{2}{c}{av} & \multicolumn{2}{c}{at} & \multicolumn{2}{c}{tv} \\")
    latex_lines.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}")
    latex_lines.append(r" & Acc & W-F1 & Acc & W-F1 & Acc & W-F1 & Acc & W-F1 \\")
    latex_lines.append(r"\midrule")

    for bi, bb in enumerate(BACKBONES_ORDER):
        bb_display = BACKBONE_DISPLAY.get(bb, bb)

        # Baseline row
        cells_base = [bb_display]
        base_vals = {}
        for mod in MODALITIES_ORDER:
            acc, f1 = lookup(ds, bb, mod, "baseline")
            base_vals[mod] = (acc, f1)
            cells_base.append(fmt_val(acc))
            cells_base.append(fmt_val(f1))
        latex_lines.append(" & ".join(cells_base) + r" \\")

        # DnR row
        cells_dnr = [r"+ \textbf{\mName{}}"]
        for mod in MODALITIES_ORDER:
            acc_b, f1_b = base_vals[mod]
            acc_d, f1_d = lookup(ds, bb, mod, "dnr")

            acc_str = fmt_val(acc_d)
            f1_str  = fmt_val(f1_d)

            # Bold if improved
            if acc_d is not None and acc_b is not None and acc_d >= acc_b:
                acc_str = r"\textbf{" + acc_str + "}"
            if f1_d is not None and f1_b is not None and f1_d >= f1_b:
                f1_str = r"\textbf{" + f1_str + "}" + fmt_delta(f1_b, f1_d)

            cells_dnr.append(acc_str)
            cells_dnr.append(f1_str)
        latex_lines.append(" & ".join(cells_dnr) + r" \\")

        if bi < len(BACKBONES_ORDER) - 1:
            latex_lines.append(r"\midrule")

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}%")
    latex_lines.append(r"}")
    latex_lines.append(r"\end{table*}")
    latex_lines.append("")

latex_output = "\n".join(latex_lines)
print(latex_output)

with open(RESULT_LATEX, 'w') as f:
    f.write(latex_output)
print(f"\nLaTeX saved to: {RESULT_LATEX}")
PYEOF

# Fix the placeholder paths in the heredoc
# (heredoc doesn't expand bash variables inside 'PYEOF', so we sed them)
TEMP_PY=$(mktemp)
cat << PYEOF2 > "$TEMP_PY"
import json

RESULT_JSON = "${RESULT_JSON}"
RESULT_LATEX = "${RESULT_LATEX}"

with open(RESULT_JSON, 'r') as f:
    data = json.load(f)

BACKBONE_DISPLAY = {
    "mmgcn":        "MMGCN",
    "dialogue_gcn": "DialogueGCN",
    "mm_dfn":       "MM-DFN",
    "simple":       "SimpleBackbone",
}

DATASET_DISPLAY = {
    "mosi_coid": "MOSI",
    "mosei_coid": "MOSEI",
}

MODALITIES_ORDER = ["atv", "av", "at", "tv"]
BACKBONES_ORDER  = ["mmgcn", "dialogue_gcn", "mm_dfn", "simple"]
DATASETS_ORDER   = ["mosi_coid", "mosei_coid"]

def lookup(dataset, backbone, modalities, method):
    key = f"{dataset}|{backbone}|{modalities}|{method}"
    entry = data.get(key, None)
    if entry is None:
        return None, None
    return entry["acc_pct"], entry["wf1_pct"]

def fmt_val(val):
    if val is None:
        return "-"
    return f"{val:.2f}"

def fmt_delta(base, improved):
    if base is None or improved is None:
        return ""
    delta = improved - base
    sign = "+" if delta >= 0 else ""
    return f" (\\\( {sign}{delta:.2f}\\\))"

latex_lines = []

for ds in DATASETS_ORDER:
    ds_display = DATASET_DISPLAY.get(ds, ds)
    latex_lines.append(r"\\begin{table*}[!htbp]")
    latex_lines.append(r"\\centering")
    latex_lines.append(r"\\caption{Performance comparison on " + ds_display + r" dataset.}")
    latex_lines.append(r"\\resizebox{\\textwidth}{!}{%")
    latex_lines.append(r"\\begin{tabular}{lcccccccc}")
    latex_lines.append(r"\\toprule")
    latex_lines.append(r"\\multirow{2}{*}{Backbone} & \\multicolumn{2}{c}{atv} & \\multicolumn{2}{c}{av} & \\multicolumn{2}{c}{at} & \\multicolumn{2}{c}{tv} \\\\")
    latex_lines.append(r"\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7} \\cmidrule(lr){8-9}")
    latex_lines.append(r" & Acc & W-F1 & Acc & W-F1 & Acc & W-F1 & Acc & W-F1 \\\\")
    latex_lines.append(r"\\midrule")

    for bi, bb in enumerate(BACKBONES_ORDER):
        bb_display = BACKBONE_DISPLAY.get(bb, bb)
        cells_base = [bb_display]
        base_vals = {}
        for mod in MODALITIES_ORDER:
            acc, f1 = lookup(ds, bb, mod, "baseline")
            base_vals[mod] = (acc, f1)
            cells_base.append(fmt_val(acc))
            cells_base.append(fmt_val(f1))
        latex_lines.append(" & ".join(cells_base) + r" \\\\")

        cells_dnr = ["+ \\\\textbf{\\\\mName{}}"]
        for mod in MODALITIES_ORDER:
            acc_b, f1_b = base_vals[mod]
            acc_d, f1_d = lookup(ds, bb, mod, "dnr")
            acc_str = fmt_val(acc_d)
            f1_str  = fmt_val(f1_d)
            if acc_d is not None and acc_b is not None and acc_d >= acc_b:
                acc_str = "\\\\textbf{" + acc_str + "}"
            if f1_d is not None and f1_b is not None and f1_d >= f1_b:
                f1_str = "\\\\textbf{" + f1_str + "}" + fmt_delta(f1_b, f1_d)
            cells_dnr.append(acc_str)
            cells_dnr.append(f1_str)
        latex_lines.append(" & ".join(cells_dnr) + r" \\\\")

        if bi < len(BACKBONES_ORDER) - 1:
            latex_lines.append(r"\\midrule")

    latex_lines.append(r"\\bottomrule")
    latex_lines.append(r"\\end{tabular}%")
    latex_lines.append(r"}")
    latex_lines.append(r"\\end{table*}")
    latex_lines.append("")

latex_output = "\\n".join(latex_lines)
with open(RESULT_LATEX, 'w') as f:
    f.write(latex_output)
print(f"LaTeX saved to: {RESULT_LATEX}")
PYEOF2

# Actually use the proper Python script with correct variable expansion
python3 << PYEOF3
import json

RESULT_JSON = "${RESULT_JSON}"
RESULT_LATEX = "${RESULT_LATEX}"

with open(RESULT_JSON, 'r') as f:
    data = json.load(f)

BACKBONE_DISPLAY = {
    "mmgcn":        "MMGCN",
    "dialogue_gcn": "DialogueGCN",
    "mm_dfn":       "MM-DFN",
    "simple":       "SimpleBackbone",
}

DATASET_DISPLAY = {
    "mosi_coid": "MOSI",
    "mosei_coid": "MOSEI",
}

MODALITIES_ORDER = ["atv", "av", "at", "tv"]
BACKBONES_ORDER  = ["mmgcn", "dialogue_gcn", "mm_dfn", "simple"]
DATASETS_ORDER   = ["mosi_coid", "mosei_coid"]

def lookup(dataset, backbone, modalities, method):
    key = f"{dataset}|{backbone}|{modalities}|{method}"
    entry = data.get(key, None)
    if entry is None:
        return None, None
    return entry["acc_pct"], entry["wf1_pct"]

def fmt_val(val):
    if val is None:
        return "-"
    return f"{val:.2f}"

def fmt_delta(base_val, improved_val):
    if base_val is None or improved_val is None:
        return ""
    delta = improved_val - base_val
    sign = "+" if delta >= 0 else ""
    return f" (\( {sign}{delta:.2f}\))"

lines = []

for ds in DATASETS_ORDER:
    ds_display = DATASET_DISPLAY.get(ds, ds)
    lines.append(r"\begin{table*}[!htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Performance comparison on " + ds_display + " dataset.}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{lcccccccc}")
    lines.append(r"\toprule")
    lines.append(r"\multirow{2}{*}{Backbone} & \multicolumn{2}{c}{atv} & \multicolumn{2}{c}{av} & \multicolumn{2}{c}{at} & \multicolumn{2}{c}{tv} \\\\")
    lines.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}")
    lines.append(r" & Acc & W-F1 & Acc & W-F1 & Acc & W-F1 & Acc & W-F1 \\\\")
    lines.append(r"\midrule")

    for bi, bb in enumerate(BACKBONES_ORDER):
        bb_display = BACKBONE_DISPLAY.get(bb, bb)

        # Baseline row
        row_base = [bb_display]
        base_vals = {}
        for mod in MODALITIES_ORDER:
            acc, f1 = lookup(ds, bb, mod, "baseline")
            base_vals[mod] = (acc, f1)
            row_base.append(fmt_val(acc))
            row_base.append(fmt_val(f1))
        lines.append(" & ".join(row_base) + r" \\\\")

        # DnR row
        row_dnr = [r"+ \textbf{\mName{}}"]
        for mod in MODALITIES_ORDER:
            acc_b, f1_b = base_vals[mod]
            acc_d, f1_d = lookup(ds, bb, mod, "dnr")
            acc_s = fmt_val(acc_d)
            f1_s  = fmt_val(f1_d)
            if acc_d is not None and acc_b is not None and acc_d >= acc_b:
                acc_s = r"\textbf{" + acc_s + "}"
            if f1_d is not None and f1_b is not None and f1_d >= f1_b:
                f1_s = r"\textbf{" + f1_s + "}" + fmt_delta(f1_b, f1_d)
            row_dnr.append(acc_s)
            row_dnr.append(f1_s)
        lines.append(" & ".join(row_dnr) + r" \\\\")

        if bi < len(BACKBONES_ORDER) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append("}")
    lines.append(r"\end{table*}")
    lines.append("")

output = "\n".join(lines)
print(output)
with open(RESULT_LATEX, 'w') as f:
    f.write(output)
print(f"\nLaTeX table saved to: {RESULT_LATEX}")
PYEOF3

rm -f "$TEMP_PY"

# ──────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "========================================================================"
echo " ✅ ALL EXPERIMENTS COMPLETE"
echo "========================================================================"
echo ""
echo " Results:"
echo "   JSON  → $RESULT_JSON"
echo "   CSV   → $RESULT_CSV"
echo "   LaTeX → $RESULT_LATEX"
echo "   Logs  → $LOG_DIR/"
echo ""
echo " CSV preview:"
echo "────────────────────────────────────────────────────────"
head -20 "$RESULT_CSV"
echo "────────────────────────────────────────────────────────"
echo ""
echo " To re-generate LaTeX from JSON later, run:"
echo "   python3 generate_latex.py --json $RESULT_JSON"