# Divide and Refine (DnR)

Official PyTorch implementation of **Divide and Refine: Enhancing Multimodal Representation and Explainability for Emotion Recognition in Conversation**, accepted at **WACV 2026**.

DnR is a plug-and-play framework for multimodal emotion and affect recognition. It improves multimodal representations by explicitly modeling modality-specific uniqueness, cross-modal redundancy, and multimodal synergy across text, audio, and visual features.

## Overview

DnR has two stages:

- **Divide:** decomposes each modality into unique, redundant, and synergistic components.
- **Refine:** strengthens the decomposed representations with redundancy-focused augmentation and contrastive learning.

The framework is model-agnostic and can be attached to several multimodal backbones with minimal changes to the original training pipeline.

## Supported Backbones

The current executable training path supports:

| CLI value | Backbone |
| --- | --- |
| `late_fusion` | Late Fusion |
| `mmgcn` | MMGCN |
| `dialogue_gcn` | DialogueGCN |
| `mm_dfn` | MM-DFN |

The parser still contains a legacy `biddin` option, but the BiDDIN module is not included in the current runnable code path.

## Supported Datasets

| Dataset | CLI values | Task format |
| --- | --- | --- |
| IEMOCAP | `iemocap`, `iemocap_coid` | Emotion recognition in conversation |
| MELD | `meld`, `meld_coid` | Emotion recognition in conversation |
| CMU-MOSI | `mosi`, `mosi_coid` | Binary sentiment classification |
| CMU-MOSEI | `mosei`, `mosei_coid` | Binary sentiment classification |
| UR-FUNNY | `humor`, `humor_coid` | Binary humor classification |
| MUSTARD | `sarcasm`, `sarcasm_coid` | Binary sarcasm classification |

The `*_coid` names are kept for compatibility with DnR experiments. When `--use_divide --use_refine` is enabled, the model trains on refined DnR representations. When those flags are omitted, the same dataset name can be used for ablation runs on the raw input features.

For MOSI, MOSEI, UR-FUNNY, and MUSTARD, each clip is adapted to the existing dialogue-style pipeline as a one-utterance sample by mean-pooling valid text-aligned timesteps.

## Installation

```bash
git clone https://github.com/mattam301/DnR-WACV2026.git
cd DnR-WACV2026
pip install -r requirements.txt
```

## Data

Place the original IEMOCAP and MELD pickles at:

```text
data/iemocap/iemocap.pkl
data/meld/meld.pkl
```

Download the supported MultiBench affect pickles:

```bash
bash scripts/download_affect.sh --datasets mosi mosei humor sarcasm
```

Downloaded affect data is expected at:

```text
data/mosi/mosi_data.pkl
data/mosei/mosei_senti_data.pkl
data/humor/humor.pkl
data/sarcasm/sarcasm.pkl
```

## Training

Run the original IEMOCAP DnR script:

```bash
bash scripts/atv.sh
```

Run DnR on the MultiBench affect datasets:

```bash
bash scripts/run_mosi.sh
bash scripts/run_mosei.sh
bash scripts/run_humor.sh
bash scripts/run_sarcasm.sh
```

Run all affect dataset scripts sequentially:

```bash
bash scripts/run_affect_all.sh
```

The dataset scripts default to `mmgcn`, `atv` modalities, and the DnR flags `--use_divide --use_refine`. Common settings can be overridden from the shell:

```bash
DEVICE=cpu EPOCHS=1 PRETRAIN_EPOCHS=0 BATCH_SIZE=64 bash scripts/run_mosi.sh
```

## Ablations

To run a raw-feature backbone baseline, omit `--use_divide --use_refine`:

```bash
python code/train.py \
  --backbone=mmgcn \
  --dataset=mosi_coid \
  --modalities=atv \
  --batch_size=32 \
  --epochs=6 \
  --device=cuda
```

To switch backbones:

```bash
BACKBONE=dialogue_gcn bash scripts/run_mosi.sh
BACKBONE=mm_dfn bash scripts/run_humor.sh
BACKBONE=late_fusion bash scripts/run_sarcasm.sh
```

## Useful Arguments

| Argument | Description |
| --- | --- |
| `--dataset` | Dataset name, including optional `*_coid` aliases |
| `--backbone` | One of `late_fusion`, `mmgcn`, `dialogue_gcn`, `mm_dfn` |
| `--modalities` | Modality subset: `atv`, `at`, `av`, `tv`, `a`, `t`, or `v` |
| `--use_divide` | Enable the Divide module |
| `--use_refine` | Enable refined DnR representations and augmentation |
| `--divide_dim` | Per-component DnR representation size |
| `--pretrain_epochs` | Number of Divide/SMURF pretraining epochs |
| `--comet` | Enable Comet logging |

## Citation

```bibtex
@article{mai2026divide,
  title={Divide and Refine: Enhancing Multimodal Representation and Explainability for Emotion Recognition in Conversation},
  author={Mai, Anh-Tuan and Nguyen, Cam-Van Thi and Le, Duc-Trong},
  journal={arXiv preprint arXiv:2601.14274},
  year={2026}
}
```
