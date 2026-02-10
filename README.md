## Official PyTorch implementation of Divide and Refine (DnR), a plug-and-play framework for improving multimodal representations in Emotion Recognition in Conversation (MERC) by explicitly modeling uniqueness, redundancy, and synergy across text, audio, and visual modalities.

📄 Accepted at WACV 2026
Divide and Refine: Enhancing Multimodal Representation and Explainability for Emotion Recognition in Conversation

⸻

Overview

DnR is a two-phase framework:
	•	Divide — Decomposes each modality into:
	•	Unique (U)
	•	Redundant (R)
	•	Synergistic (S)
	•	Refine — Strengthens representations via redundancy-focused augmentation and contrastive learning.

The method is lightweight, model-agnostic, and can be integrated into existing MERC backbones.

⸻

Features
	•	Plug-and-play with common models (MMGCN, DialogueGCN, MM-DFN, SDT, etc.)
	•	Consistent gains on IEMOCAP and MELD
	•	Improved robustness to missing/noisy modalities
	•	Better interpretability of multimodal signals

⸻

Installation

git clone https://github.com/mattam301/DnR-WACV2026.git
cd DnR-WACV2026
pip install -r requirements.txt


⸻

End2End training
bash scripts/atv.sh 



⸻

Citation

@article{mai2026divide,
  title={Divide and Refine: Enhancing Multimodal Representation and Explainability for Emotion Recognition in Conversation},
  author={Mai, Anh-Tuan and Nguyen, Cam-Van Thi and Le, Duc-Trong},
  journal={arXiv preprint arXiv:2601.14274},
  year={2026}
}

:::
