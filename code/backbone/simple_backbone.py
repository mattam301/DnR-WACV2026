"""
backbone/simple_backbone.py
────────────────────────────
Utterance-level multimodal classifier for non-conversational datasets.
(MOSI, MOSEI, Humor, Sarcasm)

Handles two tensor formats from the dataloader:
  Utterance-level (batch_size > 1):
    data["tensor"][m]: [1, batch, dim]   seq=1, one utterance per sample
    → squeeze(0) → [batch, dim]

  SMURF refine path:
    data["tensor"][m]: [seq, 1, dim]     whole dialogue as sequence
    → length masking → [N_valid, dim]

Interface (exact match to train.py):
  model.net(data)      → (logits, ratio, rep)         3 values
  model.forward(data)  → (prob,   ratio, rep)         3 values
  model.get_loss(data) → (nll, ratio, take_samp, uni_nll)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, drop_rate: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_rate),
        )

    def forward(self, x):
        return self.net(x)


class ModalityAttentionFusion(nn.Module):
    def __init__(self, hidden_dim: int, n_modalities: int):
        super().__init__()
        self.attention    = nn.Linear(hidden_dim, 1)
        self.n_modalities = n_modalities

    def forward(self, modality_reps: list):
        stacked = torch.stack(modality_reps, dim=1)    # [N, n_mod, H]
        scores  = self.attention(stacked)               # [N, n_mod, 1]
        weights = torch.softmax(scores, dim=1)          # [N, n_mod, 1]
        fused   = (weights * stacked).sum(dim=1)        # [N, H]
        return fused, weights.squeeze(-1)               # [N, H], [N, n_mod]


class SimpleMultimodalModel(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.modalities = list(args.modalities)
        self.dataset    = args.dataset
        self.n_classes  = len(args.dataset_label_dict[args.dataset])
        self.hidden_dim = args.hidden_dim

        emb_dims = args.embedding_dim[args.dataset]

        self.encoders = nn.ModuleDict({
            m: ModalityEncoder(emb_dims[m], args.hidden_dim, args.drop_rate)
            for m in self.modalities
        })

        self.fusion = ModalityAttentionFusion(
            args.hidden_dim, len(self.modalities)
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(args.hidden_dim),
            nn.Dropout(args.drop_rate),
            nn.Linear(args.hidden_dim, args.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(args.drop_rate),
            nn.Linear(args.hidden_dim // 2, self.n_classes),
        )

        self.uni_classifiers = nn.ModuleDict({
            m: nn.Linear(args.hidden_dim, self.n_classes)
            for m in self.modalities
        })

        self.criterion = nn.NLLLoss()

    # ──────────────────────────────────────────────────────────────
    #  Tensor conversion
    # ──────────────────────────────────────────────────────────────

    def _to_utterance_matrix(
        self,
        raw: torch.Tensor,
        lengths=None,
    ) -> torch.Tensor:
        """
        Convert any dataloader tensor format to [N, dim].

        Rules (in order):
          dim==2              → already [N, dim], return as-is
          dim==3, seq==1      → [1, batch, dim] utterance-level, squeeze(0)
          dim==3, feat_dim==1 → [dim, n_utt, 1] old format, squeeze+transpose
          dim==3, seq>1       → [seq, batch, dim] sequence, apply length mask
          else                → flatten to [N, dim]
        """
        if raw.dim() == 2:
            return raw

        if raw.dim() == 3:
            seq, batch, dim = raw.shape

            # Utterance-level batch: each sample is one utterance
            # shape [1, B, dim] → [B, dim]
            if seq == 1:
                return raw.squeeze(0)

            # Old dataloader format: [dim, n_utt, 1]
            # dim is large (feature dim), last axis is dummy 1
            if dim == 1:
                return raw.squeeze(-1).transpose(0, 1)   # [n_utt, feat_dim]

            # Sequence format: [seq, batch, dim]
            # Apply length masking when lengths is reliable
            if (
                lengths is not None
                and lengths.numel() == batch
                and int(lengths.max()) <= seq
            ):
                steps = torch.arange(seq, device=raw.device).unsqueeze(1)
                mask  = steps < lengths.to(raw.device).unsqueeze(0)
                return raw[mask]                          # [N_valid, dim]

            # Fallback: flatten everything
            return raw.reshape(seq * batch, dim)

        # dim > 3 or unexpected: flatten all but last axis
        return raw.reshape(-1, raw.shape[-1])

    def _unpack_feats(self, data):
        """
        Returns
        ───────
        feats  : dict {m: [N, dim]}
        labels : [N]
        """
        lengths = data.get("length", None)
        feats   = {}

        for m in self.modalities:
            raw      = data["tensor"][m]
            feats[m] = self._to_utterance_matrix(raw, lengths)

        # Align utterance counts across modalities
        # (can differ when masked data has been zeroed)
        counts = [feats[m].shape[0] for m in self.modalities]
        if len(set(counts)) > 1:
            min_n  = min(counts)
            feats  = {m: feats[m][:min_n] for m in self.modalities}

        labels = data["label_tensor"]
        return feats, labels

    # ──────────────────────────────────────────────────────────────
    #  Shared forward computation
    # ──────────────────────────────────────────────────────────────

    def _encode_and_fuse(self, data):
        feats, labels = self._unpack_feats(data)

        encoded = {m: self.encoders[m](feats[m]) for m in self.modalities}

        mod_list     = [encoded[m] for m in self.modalities]
        rep, weights = self.fusion(mod_list)

        logits = self.classifier(rep)

        ratio = {
            m: weights[:, i].mean()
            for i, m in enumerate(self.modalities)
        }

        uni_logits = {
            m: self.uni_classifiers[m](encoded[m])
            for m in self.modalities
        }

        return logits, ratio, rep, labels, uni_logits

    # ──────────────────────────────────────────────────────────────
    #  Public interface
    # ──────────────────────────────────────────────────────────────

    def net(self, data):
        """Returns (logits, ratio, rep) — 3 values, matches train.py."""
        logits, ratio, rep, labels, uni_logits = self._encode_and_fuse(data)
        return logits, ratio, rep

    def forward(self, data):
        """Returns (prob, ratio, rep) — 3 values, matches train.py."""
        logits, ratio, rep, labels, uni_logits = self._encode_and_fuse(data)
        prob = F.log_softmax(logits, dim=-1)
        return prob, ratio, rep

    def get_loss(self, data):
        """Returns (nll, ratio, take_samp, uni_nll) — 4 values, matches train.py."""
        logits, ratio, rep, labels, uni_logits = self._encode_and_fuse(data)

        prob = F.log_softmax(logits, dim=-1)
        nll  = self.criterion(prob, labels)

        uni_nll = {
            m: self.criterion(
                F.log_softmax(uni_logits[m], dim=-1),
                labels,
            )
            for m in self.modalities
        }

        return nll, ratio, labels.shape[0], uni_nll