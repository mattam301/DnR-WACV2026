"""
new_smurf_decomp.py  (v3 — label-guided synergy, conflict-free)
────────────────────────────────────────────────────────────────
Key changes from v2:
  1. Synergy loss uses LABEL PREDICTION gap, not raw reconstruction.
     → directly aligned with downstream task
     → cannot be gamed by encoding raw features
     → margin-based, bounded, stable

  2. Cross-modal SYNERGY alignment removed from compute_corr_loss.
     → was in direct conflict with masked-necessity objective
     → only REDUNDANT heads (r) are aligned across modalities now

  3. Separate synergy classifier (GradReverse on single-head paths)
     → each sᵢ alone should be WEAK at predicting y # Actually this is not really intuitive
     → [s1,s2,s3] together should be STRONG at predicting y
     → this is the operational definition of label-relevant synergy

  4. guard loss kept but reweighted — now only pushes sᵢ ⊥ (uⱼ,rⱼ)
     for cross-modal pairs (i≠j), within-modal already covered by L_unco
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
#  Gradient reversal (for single-head weakness enforcement)
# ══════════════════════════════════════════════════════════════════════

class _GradReverse(torch.autograd.Function):
    """Reverses gradient during backward pass (GRL)."""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x, alpha=1.0):
    return _GradReverse.apply(x, alpha)


# ══════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════

def _flatten_valid(x: torch.Tensor, lengths=None) -> torch.Tensor:
    """
    [seq, batch, dim] → [N_valid, dim], dropping padded positions.
    [batch, dim] → returned as-is.
    """
    if x.dim() == 2:
        return x
    if lengths is None:
        return x.reshape(-1, x.shape[-1])
    seq_len = x.shape[0]
    steps = torch.arange(seq_len, device=x.device).unsqueeze(1)
    mask  = steps < lengths.to(x.device).unsqueeze(0)
    return x[mask]


def _flatten_labels(labels: torch.Tensor, lengths=None) -> torch.Tensor:
    """
    Flatten label tensor to match _flatten_valid output.
    labels: [batch] or [N] (already flat from train.py masking logic).
    """
    # In train.py labels are already flattened to [sum(lengths)]
    # so we just return as-is
    return labels


def _feature_corr(a, b, lengths=None, eps=1e-6):
    a = _flatten_valid(a, lengths)
    b = _flatten_valid(b, lengths)
    if a.shape[0] < 2:
        return a.new_zeros(())
    a = a - a.mean(0, keepdim=True)
    b = b - b.mean(0, keepdim=True)
    a = a / a.std(0, unbiased=False, keepdim=True).clamp_min(eps)
    b = b / b.std(0, unbiased=False, keepdim=True).clamp_min(eps)
    return (a * b).mean(0).clamp(-1.0, 1.0)


def _push(a, b, lengths=None):
    """Decorrelation: minimise |corr(a,b)|."""
    return torch.mean(torch.abs(_feature_corr(a, b, lengths)))


def _pull(a, b, lengths=None):
    """Alignment: maximise |corr(a,b)|."""
    return 1.0 - torch.mean(torch.abs(_feature_corr(a, b, lengths)))


# ══════════════════════════════════════════════════════════════════════
#  Modality branch
# ══════════════════════════════════════════════════════════════════════

class ModalityBranch(nn.Module):
    """
    Projects one modality into three disentangled sub-spaces.

    Output: (u, r, s)
      u – unique   : exclusive to this modality
      r – redundant: shared across modalities
      s – synergy  : only meaningful in combination
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm       = nn.LayerNorm(in_dim)
        self.fc_unique  = nn.Linear(in_dim, out_dim)
        self.fc_shared  = nn.Linear(in_dim, out_dim)
        self.fc_synergy = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor):
        x = self.norm(x)
        return (
            self.fc_unique(x),
            self.fc_shared(x),
            self.fc_synergy(x),
        )


# ══════════════════════════════════════════════════════════════════════
#  Label-guided synergy module
# ══════════════════════════════════════════════════════════════════════

class SynergyModule(nn.Module):
    """
    Label-guided synergy loss  (stable, task-aligned, conflict-free).

    Core idea
    ──────────
    True synergy w.r.t. label y means:
      • [s1,s2,s3] together  → GOOD at predicting y
      • s1, s2, s3 alone     → WEAK at predicting y

    We enforce this with:

    (A) Joint classifier:   C_joint([s1‖s2‖s3]) → y
        L_joint = CE(C_joint([s1,s2,s3]), y)   ← minimise

    (B) Single-head weakness via GRL:
        Each sᵢ passes through a Gradient Reversal Layer before
        a single-head classifier.  The classifier tries to predict y,
        GRL reverses gradient → encoder is pushed to make sᵢ
        individually uninformative.
        L_single = mean CE(C_i(GRL(sᵢ)), y)    ← classifier minimises,
                                                    encoder maximises

    (C) Margin gap (bounded):
        gap = mean(CE_single_i) - CE_joint
        L_masked = CE_joint + ReLU(margin - gap)
        → 0 when joint is good AND single heads are individually weak
        → Bounded, cannot diverge

    (D) Guard: sᵢ ⊥ (uⱼ, rⱼ) for cross-modal pairs (i≠j)
        Within-modal orthogonality already handled by L_unco.

    Why label-guided > reconstruction-guided
    ──────────────────────────────────────────
    • Reconstruction can be satisfied by partitioning raw features
      (each sᵢ encodes its own modality's raw signal).
    • Label guidance cannot: the label is a joint property, not
      decomposable by modality.  The only way to satisfy both
      (A) and (B) simultaneously is to encode cross-modal interactions
      that are relevant to the prediction target.

    Parameters
    ──────────
    syn_dim    : dimension of each synergy vector
    n_classes  : number of output classes
    margin     : required CE gap between joint and single-head
    grl_alpha  : gradient reversal strength (tune: 0.1-1.0)
    """

    def __init__(
        self,
        syn_dim:   int,
        n_classes: int,
        margin:    float = 0.3,
        grl_alpha: float = 0.5,
    ):
        super().__init__()
        self.margin    = margin
        self.grl_alpha = grl_alpha

        # Joint classifier — receives all three synergy vectors
        self.joint_clf = nn.Sequential(
            nn.Linear(syn_dim * 3, syn_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(syn_dim, n_classes),
        )

        # Single-head classifiers — one per modality
        # GRL is applied at call time, not here
        self.single_clfs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(syn_dim, syn_dim // 2),
                nn.ReLU(),
                nn.Linear(syn_dim // 2, n_classes),
            )
            for _ in range(3)
        ])

    def forward(
        self,
        s1: torch.Tensor,
        s2: torch.Tensor,
        s3: torch.Tensor,
        labels: torch.Tensor,
        m1: tuple,
        m2: tuple,
        m3: tuple,
        lengths=None,
    ):
        """
        Parameters
        ──────────
        s1,s2,s3 : synergy vectors [seq, batch, syn_dim]
        labels   : class indices [N_valid]  (already flattened in train.py)
        m1,m2,m3 : (u,r,s) tuples for guard loss
        lengths  : sequence lengths or None

        Returns
        ───────
        L_syn   : total synergy loss
        L_joint : joint classification loss component
        L_guard : orthogonality guard component
        """
        # ── flatten synergy vectors ────────────────────────────────
        s1f = _flatten_valid(s1, lengths)
        s2f = _flatten_valid(s2, lengths)
        s3f = _flatten_valid(s3, lengths)

        # ── (A+B+C) label-guided synergy gap ──────────────────────

        # Joint: should predict well
        joint_logits = self.joint_clf(torch.cat([s1f, s2f, s3f], dim=-1))
        L_joint = F.cross_entropy(joint_logits, labels)

        # Single-head with GRL: each should be individually weak
        # GRL reverses gradient → encoder learns to make sᵢ uninformative
        single_losses = []
        for clf, sf in zip(self.single_clfs, [s1f, s2f, s3f]):
            sf_grl  = grad_reverse(sf, self.grl_alpha)
            logits  = clf(sf_grl)
            # We compute CE normally; GRL handles the adversarial push
            single_losses.append(F.cross_entropy(logits, labels))
        L_single_mean = sum(single_losses) / len(single_losses)

        # Margin gap: joint should be better than single by >= margin
        # Use CE values as proxy for "informativeness"
        # High single CE = single heads are weak (good)
        # Low joint CE  = joint is strong (good)
        gap     = torch.clamp(
            L_single_mean - L_joint,
            min=0.0, max=3.0,          # 3.0 ≈ ln(n_classes)+1, natural ceiling
        )
        L_syn_gap = L_joint + F.relu(self.margin - gap)

        # ── (D) Cross-modal guard ──────────────────────────────────
        # Only cross-modal pairs (i≠j) — within-modal covered by L_unco
        u1, r1, _ = m1
        u2, r2, _ = m2
        u3, r3, _ = m3

        cross_guard_pairs = [
            # s1 ⊥ u2, u3, r2, r3
            (s1, u2), (s1, u3), (s1, r2), (s1, r3),
            # s2 ⊥ u1, u3, r1, r3
            (s2, u1), (s2, u3), (s2, r1), (s2, r3),
            # s3 ⊥ u1, u2, r1, r2
            (s3, u1), (s3, u2), (s3, r1), (s3, r2),
        ]
        L_guard = sum(
            _push(a, b, lengths) for a, b in cross_guard_pairs
        ) / len(cross_guard_pairs)

        L_syn = L_syn_gap + L_guard
        return L_syn, L_joint, L_guard


# ══════════════════════════════════════════════════════════════════════
#  Disentanglement loss
# ══════════════════════════════════════════════════════════════════════

def compute_corr_loss(m1, m2, m3, lengths=None):
    """
    Disentanglement loss: within-modality orthogonality + redundant
    head alignment.

    Changes from v2
    ────────────────
    REMOVED: cross-modal synergy alignment (s1↔s2, s2↔s3, s3↔s1).
    Reason:  that alignment conflicts with the masked-necessity /
             label-guided synergy objective.  Aligning synergy heads
             encourages redundancy; the synergy module encourages
             non-redundancy.  Keeping both causes a tug-of-war.

    KEPT:
      L_unco  : u⊥r, u⊥s, r⊥s within each modality (9 terms)
      L_cross : r1↔r2, r2↔r3, r3↔r1 only (3 terms)

    Returns
    ───────
    corr_loss, L_unco, L_cross
    """
    u1, r1, s1 = m1
    u2, r2, s2 = m2
    u3, r3, s3 = m3

    # Within-modality orthogonality (all three head-pairs)
    unco_pairs = [
        (u1, r1), (u2, r2), (u3, r3),   # u ⊥ r
        (u1, s1), (u2, s2), (u3, s3),   # u ⊥ s
        (r1, s1), (r2, s2), (r3, s3),   # r ⊥ s
    ]
    L_unco = sum(
        _push(a, b, lengths) for a, b in unco_pairs
    ) / len(unco_pairs)

    # Cross-modal alignment for REDUNDANT heads only
    cross_pairs = [
        (r1, r2), (r2, r3), (r3, r1),
    ]
    L_cross = sum(
        _pull(a, b, lengths) for a, b in cross_pairs
    ) / len(cross_pairs)

    return L_unco + L_cross, L_unco, L_cross


# ══════════════════════════════════════════════════════════════════════
#  Top-level SMURF model
# ══════════════════════════════════════════════════════════════════════

class ThreeModalityModel(nn.Module):
    """
    SMURF v3 — label-guided synergy, conflict-free disentanglement.

    Changes from v2
    ────────────────
    • SynergyModule now uses label-guided CE gap instead of
      raw-input masked reconstruction.
    • n_classes parameter added (required for synergy classifier).
    • margin/grl_alpha exposed as constructor args.
    • compute_synergy_loss now requires labels argument.
    """

    def __init__(
        self,
        t_dim:     int,
        a_dim:     int,
        v_dim:     int,
        out_dim:   int,
        final_dim: int,
        margin:    float = 0.3,
        grl_alpha: float = 0.5,
    ):
        super().__init__()
        self.branch_t = ModalityBranch(t_dim, out_dim)
        self.branch_a = ModalityBranch(a_dim, out_dim)
        self.branch_v = ModalityBranch(v_dim, out_dim)
        self.fusion   = nn.Linear(out_dim, final_dim)
        self.synergy  = SynergyModule(
            syn_dim   = out_dim,
            n_classes = final_dim,
            margin    = margin,
            grl_alpha = grl_alpha,
        )

    def forward(self, x1, x2, x3):
        m1 = self.branch_t(x1)
        m2 = self.branch_a(x2)
        m3 = self.branch_v(x3)
        fused      = sum(m[i] for m in (m1, m2, m3) for i in range(3))
        final_repr = self.fusion(fused)
        return m1, m2, m3, final_repr

    def compute_synergy_loss(
        self,
        m1, m2, m3,
        labels: torch.Tensor,
        lengths=None,
    ):
        """
        Parameters
        ──────────
        m1, m2, m3 : (u, r, s) tuples from forward()
        labels     : [N_valid] class indices — same flattened labels
                     used for NLL in smurf_pretrain
        lengths    : LongTensor [batch] or None

        Returns
        ───────
        L_syn   : total synergy loss
        L_joint : joint classifier CE  (should decrease)
        L_guard : cross-modal orthogonality  (should decrease)
        """
        s1, s2, s3 = m1[2], m2[2], m3[2]
        return self.synergy(
            s1, s2, s3,
            labels,
            m1, m2, m3,
            lengths,
        )