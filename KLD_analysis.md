# DnR KLD Interpretability Analysis

This analysis treats each DnR component as a high-dimensional distribution instead of a scalar score. It is designed for the `U/R/S` outputs from the Divide module:

- `U`: modality-unique information
- `R`: cross-modal redundant information
- `S`: synergistic information

The implementation is in `code/dnr_kld_analysis.py` and is enabled from `code/train.py` with `--run_kld_analysis`.

## Run

Use the same command as training, but add the KLD flag:

```bash
python code/train.py \
  --dataset sarcasm_coid \
  --backbone mmgcn \
  --modalities atv \
  --seed 42 \
  --learning_rate 0.0001 \
  --weight_decay 1e-8 \
  --early_stopping 20 \
  --batch_size 32 \
  --epochs 60 \
  --use_refine \
  --use_divide \
  --run_kld_analysis
```

Optional arguments:

```bash
--analysis_dir analysis_outputs
--analysis_tau 1.0
```

`--analysis_tau` is the softmax temperature used to convert component vectors into distributions. A smaller value makes the distribution sharper; a larger value makes it smoother. Keep it fixed across experiments.

## Outputs

The default output path is:

```text
analysis_outputs/<dataset>/test/
```

Files:

- `pairwise_component_skl.csv`
- `prediction_sensitivity_kl.csv`
- `summary.json`

## 1. Pairwise Component SKL

File:

```text
pairwise_component_skl.csv
```

This computes symmetric KLD between component distributions:

```text
SKL(P, Q) = 0.5 * KL(P || Q) + 0.5 * KL(Q || P)
```

Each component is named as:

```text
t-U, t-R, t-S
a-U, a-R, a-S
v-U, v-R, v-S
```

Useful readings:

- High `SKL(t-U, t-R)` means text-unique and text-redundant representations are well separated.
- Low `SKL(t-R, a-R)` means text/audio redundancy components are distributionally similar.
- Meaningful `SKL(R, S)` means synergy has not collapsed into redundancy.

For sarcasm, useful comparisons include:

```text
t-U vs t-S
t-R vs a-R
t-S vs a-S
a-S vs v-S
R components vs S components
```

The CSV also includes class-conditioned columns such as:

```text
mean_skl_not_sarcasm
mean_skl_sarcasm
```

These help answer whether sarcasm samples have stronger component separation or stronger cross-modal divergence than non-sarcasm samples.

## 2. Prediction-Sensitivity KL

File:

```text
prediction_sensitivity_kl.csv
```

This measures how much the model prediction distribution changes when one DnR component is removed:

```text
KL(P_full(y | x) || P_without_component(y | x))
```

Interpretation:

- Larger KL means the removed component was more important to the prediction distribution.
- Smaller KL means the model prediction was relatively insensitive to that component.

Example reading:

```text
removed_component = t-U
mean_kl_full_to_removed = 0.42
```

This means removing the text-unique component substantially changed the prediction distribution.

For sarcasm, strong evidence would be:

- `t-U` has high KL: lexical or semantic uniqueness drives sarcasm.
- `a-S` or `v-S` has high KL: cross-modal incongruity contributes to sarcasm.
- `R` has low KL for sarcasm: sarcasm is less about simple modality agreement.

## 3. Summary JSON

File:

```text
summary.json
```

This stores compact averages for quick inspection:

```json
{
  "pairwise_component_skl": {
    "t-U__t-R": 0.0
  },
  "prediction_sensitivity_kl": {
    "t-U": 0.0
  }
}
```

Use the CSV files for detailed tables and the JSON file for quick reporting or plotting.

## Suggested Paper Tables

Pairwise disentanglement table:

```text
Metric                         not_sarcasm    sarcasm
SKL(U, R) within modality
SKL(U, S) within modality
SKL(R across modalities)
SKL(S across modalities)
SKL(R, S)
```

Prediction-sensitivity table:

```text
Removed component    Overall KL    not_sarcasm KL    sarcasm KL
t-U
t-R
t-S
a-U
a-R
a-S
v-U
v-R
v-S
```

## Suggested Case Study

For a single correctly classified sarcasm sample, plot the `prediction_sensitivity_kl.csv` values as a bar chart:

```text
t-U, t-R, t-S, a-U, a-R, a-S, v-U, v-R, v-S
```

A strong explanation might read:

```text
The prediction is most sensitive to text-unique and audio-synergistic components,
suggesting that the model relies on lexical cues and cross-modal incongruity
rather than only redundant agreement across modalities.
```

## Notes

- KLD is asymmetric, so pairwise component analysis uses symmetric KLD.
- Prediction sensitivity keeps the asymmetric form because the direction is meaningful: full prediction distribution to component-removed distribution.
- The analysis runs after the best checkpoint is restored, so it explains the same model used for final test metrics.
- This analysis requires `--use_divide --use_refine`; raw-feature baselines do not expose `U/R/S` components.
