"""
analysis/explainability_analysis.py

Three targeted explainability analyses:

1. Component Attribution
   For each utterance: which component (U/R/S) contributed most
   to the correct prediction?
   → Shows U/R/S have meaningful semantic roles

2. Modality Dominance by Emotion
   For each emotion class: which modality's unique component
   is most activated?
   → Shows model uses different modalities for different emotions
   → Directly interpretable: "for Angry, audio U dominates"

3. Contrastive Case Explanation
   For flip-positive cases: compare what changed between
   raw and DnR representations
   → Shows the decomposition reveals signal the raw model missed
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys
import copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_utils import apply_smurf_to_data


# ══════════════════════════════════════════════════════════════════════
#  Analysis 1: Component Attribution
# ══════════════════════════════════════════════════════════════════════

def component_attribution(smurf_model, model_dnr, test_set, args, device, output_dir):
    """
    For each utterance, measure how much each component (U/R/S)
    contributes to the correct class probability.

    Method: zero-ablation
      - Run DnR with all components → get baseline prob
      - Zero out one component at a time → measure drop in correct-class prob
      - Attribution = baseline_prob - ablated_prob

    This directly answers: "which component was most important
    for this prediction?" — a form of feature attribution.
    """
    os.makedirs(output_dir, exist_ok=True)

    label_dict      = args.dataset_label_dict[args.dataset]
    label_names_inv = {v: k for k, v in label_dict.items()}

    smurf_model.eval()
    model_dnr.eval()

    # Detect feature format
    emb_dim_t  = args.embedding_dim[args.dataset]["t"]
    divide_dim = args.divide_dim
    use_cross  = (emb_dim_t // divide_dim == 7)

    records = []

    with torch.no_grad():
        for idx in range(len(test_set)):
            data = test_set[idx]
            for k, v in data.items():
                if k == "utterance_texts": continue
                if k == "tensor":
                    for m, feat in data[k].items():
                        data[k][m] = feat.to(device)
                else:
                    data[k] = v.to(device)

            labels  = data["label_tensor"].cpu().numpy()
            lengths = data["length"]
            texts   = data.get("utterance_texts", [])

            # Get SMURF components
            x1 = data["tensor"]['t']
            x2 = data["tensor"]['a']
            x3 = data["tensor"]['v']
            textf   = (x1.permute(1, 2, 0)).transpose(1, 2)
            audiof  = (x2.permute(1, 2, 0)).transpose(1, 2)
            visualf = (x3.permute(1, 2, 0)).transpose(1, 2)
            m1, m2, m3, _ = smurf_model(textf, audiof, visualf)

            u1, r1, s1 = m1
            u2, r2, s2 = m2
            u3, r3, s3 = m3

            def make_features(u1, r1, s1, u2, r2, s2, u3, r3, s3):
                if use_cross:
                    cross_t = torch.cat([u1-u2, u1-u3, u1*u2, u1*u3], dim=-1)
                    cross_a = torch.cat([u2-u1, u2-u3, u2*u1, u2*u3], dim=-1)
                    cross_v = torch.cat([u3-u1, u3-u2, u3*u1, u3*u2], dim=-1)
                    ft = torch.cat([u1, r1, s1, cross_t], dim=-1)
                    fa = torch.cat([u2, r2, s2, cross_a], dim=-1)
                    fv = torch.cat([u3, r3, s3, cross_v], dim=-1)
                else:
                    ft = torch.cat([u1, r1, s1], dim=-1)
                    fa = torch.cat([u2, r2, s2], dim=-1)
                    fv = torch.cat([u3, r3, s3], dim=-1)
                return ft, fa, fv

            zero = torch.zeros_like(u1)

            def run_model(ft, fa, fv):
                d = copy.deepcopy(data)
                d["tensor"]['t'] = ft.transpose(0, 1)
                d["tensor"]['a'] = fa.transpose(0, 1)
                d["tensor"]['v'] = fv.transpose(0, 1)
                prob, _, _ = model_dnr(d)
                return torch.softmax(prob, dim=-1).cpu().numpy()

            # Baseline (all components)
            ft, fa, fv = make_features(u1,r1,s1, u2,r2,s2, u3,r3,s3)
            prob_base = run_model(ft, fa, fv)

            # Ablate U (zero all unique heads)
            ft_nu, fa_nu, fv_nu = make_features(zero,r1,s1, zero,r2,s2, zero,r3,s3)
            prob_no_u = run_model(ft_nu, fa_nu, fv_nu)

            # Ablate R (zero all redundant heads)
            ft_nr, fa_nr, fv_nr = make_features(u1,zero,s1, u2,zero,s2, u3,zero,s3)
            prob_no_r = run_model(ft_nr, fa_nr, fv_nr)

            # Ablate S (zero all synergy heads)
            ft_ns, fa_ns, fv_ns = make_features(u1,r1,zero, u2,r2,zero, u3,r3,zero)
            prob_no_s = run_model(ft_ns, fa_ns, fv_ns)

            for i in range(len(labels)):
                gold = int(labels[i])
                pred = int(np.argmax(prob_base[i]))
                correct = (pred == gold)

                # Attribution = drop in correct-class prob when ablated
                attr_u = float(prob_base[i, gold] - prob_no_u[i, gold])
                attr_r = float(prob_base[i, gold] - prob_no_r[i, gold])
                attr_s = float(prob_base[i, gold] - prob_no_s[i, gold])

                # Which component was most important?
                attrs  = {"U": attr_u, "R": attr_r, "S": attr_s}
                dominant = max(attrs, key=attrs.get)

                text = texts[i] if i < len(texts) else ""

                records.append({
                    "gold":       label_names_inv.get(gold, str(gold)),
                    "pred":       label_names_inv.get(pred, str(pred)),
                    "correct":    correct,
                    "prob_base":  round(float(prob_base[i, gold]), 4),
                    "attr_U":     round(attr_u, 4),
                    "attr_R":     round(attr_r, 4),
                    "attr_S":     round(attr_s, 4),
                    "dominant":   dominant,
                    "text":       text,
                })

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(output_dir, "component_attribution.csv"), index=False)

    # ── Summary: dominant component by emotion class ──────────────────────
    summary_rows = []
    for emotion in df["gold"].unique():
        sub = df[(df["gold"] == emotion) & df["correct"]]
        if len(sub) == 0:
            continue
        total = len(sub)
        for comp in ["U", "R", "S"]:
            pct = (sub["dominant"] == comp).sum() / total * 100
            mean_attr = sub[f"attr_{comp}"].mean()
            summary_rows.append({
                "emotion":   emotion,
                "component": comp,
                "dominant_%": round(pct, 1),
                "mean_attr":  round(mean_attr, 4),
                "n_correct":  total,
            })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(
        os.path.join(output_dir, "attribution_by_emotion.csv"), index=False
    )

    print("\n=== Component Attribution by Emotion (correct predictions only) ===")
    pivot = df_summary.pivot(index="emotion", columns="component", values="dominant_%")
    print(pivot.to_string())

    # ── Plot ──────────────────────────────────────────────────────────────
    _plot_attribution(df_summary, output_dir)

    return df, df_summary


def _plot_attribution(df_summary, output_dir):
    """Stacked bar: dominant component % per emotion class."""
    emotions = df_summary["emotion"].unique()
    u_vals = [df_summary[(df_summary["emotion"]==e) &
               (df_summary["component"]=="U")]["dominant_%"].values[0]
              for e in emotions]
    r_vals = [df_summary[(df_summary["emotion"]==e) &
               (df_summary["component"]=="R")]["dominant_%"].values[0]
              for e in emotions]
    s_vals = [df_summary[(df_summary["emotion"]==e) &
               (df_summary["component"]=="S")]["dominant_%"].values[0]
              for e in emotions]

    x = np.arange(len(emotions))
    width = 0.5
    colors = {"U": "#4575b4", "R": "#d73027", "S": "#1a9850"}

    fig, ax = plt.subplots(figsize=(9, 4))
    b1 = ax.bar(x, u_vals, width, label="Unique (U)",    color=colors["U"])
    b2 = ax.bar(x, r_vals, width, label="Redundant (R)", color=colors["R"],
                bottom=u_vals)
    b3 = ax.bar(x, s_vals, width, label="Synergy (S)",   color=colors["S"],
                bottom=[u+r for u,r in zip(u_vals, r_vals)])

    ax.set_xticks(x)
    ax.set_xticklabels(emotions, fontsize=11)
    ax.set_ylabel("% of correct predictions where component dominates", fontsize=10)
    ax.set_title("Dominant PID Component per Emotion Class (IEMOCAP)", fontsize=12)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 100)
    ax.axhline(33.3, color="gray", linestyle="--", linewidth=0.8,
               label="Uniform baseline")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "attribution_stacked.pdf"),
                bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "attribution_stacked.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved attribution plot.")


# ══════════════════════════════════════════════════════════════════════
#  Analysis 2: Modality Dominance by Emotion
# ══════════════════════════════════════════════════════════════════════

def modality_dominance_by_emotion(
    smurf_model, test_set, args, device, output_dir
):
    """
    For each emotion class, measure the mean L2 norm of each
    modality's UNIQUE component (U_text, U_audio, U_video).

    High norm = that modality's unique signal is strongly activated.

    Interpretation:
      If U_audio is highest for Angry → audio carries most unique
      emotion information for angry utterances.
      This is directly human-interpretable and connects to
      emotion recognition literature.
    """
    os.makedirs(output_dir, exist_ok=True)

    label_dict      = args.dataset_label_dict[args.dataset]
    label_names_inv = {v: k for k, v in label_dict.items()}

    smurf_model.eval()

    # Collect per-utterance norms
    records = []

    with torch.no_grad():
        for idx in range(len(test_set)):
            data = test_set[idx]
            for k, v in data.items():
                if k == "utterance_texts": continue
                if k == "tensor":
                    for m, feat in data[k].items():
                        data[k][m] = feat.to(device)
                else:
                    data[k] = v.to(device)

            labels  = data["label_tensor"].cpu().numpy()
            lengths = data["length"]

            x1 = data["tensor"]['t']
            x2 = data["tensor"]['a']
            x3 = data["tensor"]['v']
            textf   = (x1.permute(1, 2, 0)).transpose(1, 2)
            audiof  = (x2.permute(1, 2, 0)).transpose(1, 2)
            visualf = (x3.permute(1, 2, 0)).transpose(1, 2)
            m1, m2, m3, _ = smurf_model(textf, audiof, visualf)

            def flatten(x):
                seq_len = x.shape[0]
                steps = torch.arange(seq_len, device=x.device).unsqueeze(1)
                mask  = steps < lengths.to(x.device).unsqueeze(0)
                return x[mask].cpu().numpy()

            u_t = flatten(m1[0])
            r_t = flatten(m1[1])
            s_t = flatten(m1[2])
            u_a = flatten(m2[0])
            r_a = flatten(m2[1])
            s_a = flatten(m2[2])
            u_v = flatten(m3[0])
            r_v = flatten(m3[1])
            s_v = flatten(m3[2])

            for i in range(len(labels)):
                gold = int(labels[i])
                records.append({
                    "emotion":  label_names_inv.get(gold, str(gold)),
                    "u_t_norm": float(np.linalg.norm(u_t[i])),
                    "u_a_norm": float(np.linalg.norm(u_a[i])),
                    "u_v_norm": float(np.linalg.norm(u_v[i])),
                    "r_t_norm": float(np.linalg.norm(r_t[i])),
                    "r_a_norm": float(np.linalg.norm(r_a[i])),
                    "r_v_norm": float(np.linalg.norm(r_v[i])),
                    "s_t_norm": float(np.linalg.norm(s_t[i])),
                    "s_a_norm": float(np.linalg.norm(s_a[i])),
                    "s_v_norm": float(np.linalg.norm(s_v[i])),
                })

    df = pd.DataFrame(records)

    # Mean norm per emotion per component
    summary = df.groupby("emotion").mean().round(4)
    summary.to_csv(os.path.join(output_dir, "modality_dominance.csv"))

    print("\n=== Mean Component Norm by Emotion Class ===")
    print(summary[["u_t_norm", "u_a_norm", "u_v_norm"]].to_string())

    # ── Plot: heatmap of unique component norms ───────────────────────────
    _plot_modality_heatmap(summary, output_dir)

    return df, summary


def _plot_modality_heatmap(summary, output_dir):
    """
    Heatmap: rows = emotion classes, columns = U_text/U_audio/U_video
    Cell value = mean L2 norm (normalised per column)
    """
    data_plot = summary[["u_t_norm", "u_a_norm", "u_v_norm"]].copy()
    # Normalise each column to [0,1] for visual comparison
    data_norm = (data_plot - data_plot.min()) / (
        data_plot.max() - data_plot.min() + 1e-8
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(data_norm.values, cmap="YlOrRd", aspect="auto",
                   vmin=0, vmax=1)

    ax.set_xticks(range(3))
    ax.set_xticklabels(["Text (U)", "Audio (U)", "Video (U)"], fontsize=11)
    ax.set_yticks(range(len(data_norm.index)))
    ax.set_yticklabels(data_norm.index, fontsize=11)
    ax.set_title("Unique Component Activation by Emotion\n"
                 "(normalised mean L2 norm)", fontsize=11)

    # Annotate cells with raw values
    for i in range(len(data_norm.index)):
        for j in range(3):
            val = data_plot.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9,
                    color="white" if data_norm.values[i,j] > 0.6 else "black")

    plt.colorbar(im, ax=ax, label="Normalised activation")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "modality_dominance_heatmap.pdf"),
                bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "modality_dominance_heatmap.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved modality dominance heatmap.")


# ══════════════════════════════════════════════════════════════════════
#  Analysis 3: Contrastive Explanation for Flip Cases
# ══════════════════════════════════════════════════════════════════════

def contrastive_flip_explanation(
    smurf_model, model_raw, model_dnr,
    test_set, args, device, output_dir,
    n_examples=5
):
    """
    For the top flip-positive cases (raw wrong, DnR correct),
    explain WHAT changed between raw and DnR:

    For each such case, report:
      - The utterance text
      - Gold label, raw prediction, DnR prediction
      - Which component dominated in the DnR correct prediction
      - The raw model's confidence on gold class vs DnR's confidence
      - A human-readable explanation: "DnR succeeded because the
        [U/R/S] component of [text/audio/video] was decisive,
        which the raw model could not access"

    This is the most directly explainability-relevant output.
    """
    os.makedirs(output_dir, exist_ok=True)

    label_dict      = args.dataset_label_dict[args.dataset]
    label_names_inv = {v: k for k, v in label_dict.items()}

    emb_dim_t  = args.embedding_dim[args.dataset]["t"]
    divide_dim = args.divide_dim
    use_cross  = (emb_dim_t // divide_dim == 7)

    model_raw.eval()
    model_dnr.eval()
    smurf_model.eval()

    flip_cases = []

    with torch.no_grad():
        for idx in range(len(test_set)):
            data = test_set[idx]
            for k, v in data.items():
                if k == "utterance_texts": continue
                if k == "tensor":
                    for m, feat in data[k].items():
                        data[k][m] = feat.to(device)
                else:
                    data[k] = v.to(device)

            labels = data["label_tensor"].cpu().numpy()
            texts  = data.get("utterance_texts", [])

            # Raw prediction
            prob_raw, _, _ = model_raw(data)
            preds_raw = torch.argmax(prob_raw, dim=-1).cpu().numpy()
            conf_raw  = torch.softmax(prob_raw, dim=-1).cpu().numpy()

            # DnR prediction
            data_dnr = copy.deepcopy(data)
            data_dnr, m1, m2, m3 = apply_smurf_to_data(
                data_dnr, smurf_model, args, device
            )
            prob_dnr, _, _ = model_dnr(data_dnr)
            preds_dnr = torch.argmax(prob_dnr, dim=-1).cpu().numpy()
            conf_dnr  = torch.softmax(prob_dnr, dim=-1).cpu().numpy()

            # Component attribution for DnR
            u1, r1, s1 = m1
            u2, r2, s2 = m2
            u3, r3, s3 = m3
            zero = torch.zeros_like(u1)
            lengths = data["length"]

            def make_feat(u1,r1,s1, u2,r2,s2, u3,r3,s3):
                if use_cross:
                    cross_t = torch.cat([u1-u2,u1-u3,u1*u2,u1*u3], dim=-1)
                    cross_a = torch.cat([u2-u1,u2-u3,u2*u1,u2*u3], dim=-1)
                    cross_v = torch.cat([u3-u1,u3-u2,u3*u1,u3*u2], dim=-1)
                    return (torch.cat([u1,r1,s1,cross_t],-1),
                            torch.cat([u2,r2,s2,cross_a],-1),
                            torch.cat([u3,r3,s3,cross_v],-1))
                return (torch.cat([u1,r1,s1],-1),
                        torch.cat([u2,r2,s2],-1),
                        torch.cat([u3,r3,s3],-1))

            def run_dnr(ft, fa, fv):
                d = copy.deepcopy(data)
                d["tensor"]['t'] = ft.transpose(0,1)
                d["tensor"]['a'] = fa.transpose(0,1)
                d["tensor"]['v'] = fv.transpose(0,1)
                p, _, _ = model_dnr(d)
                return torch.softmax(p, dim=-1).cpu().numpy()

            ft,fa,fv = make_feat(u1,r1,s1,u2,r2,s2,u3,r3,s3)
            p_base   = run_dnr(ft,fa,fv)
            p_no_u   = run_dnr(*make_feat(zero,r1,s1,zero,r2,s2,zero,r3,s3))
            p_no_r   = run_dnr(*make_feat(u1,zero,s1,u2,zero,s2,u3,zero,s3))
            p_no_s   = run_dnr(*make_feat(u1,r1,zero,u2,r2,zero,u3,r3,zero))

            for i in range(len(labels)):
                gold     = int(labels[i])
                pred_raw = int(preds_raw[i])
                pred_dnr = int(preds_dnr[i])

                if pred_raw == gold or pred_dnr != gold:
                    continue   # only flip-positive cases

                attr_u = float(p_base[i,gold] - p_no_u[i,gold])
                attr_r = float(p_base[i,gold] - p_no_r[i,gold])
                attr_s = float(p_base[i,gold] - p_no_s[i,gold])

                dom_comp = max({"U":attr_u,"R":attr_r,"S":attr_s},
                               key=lambda k: {"U":attr_u,"R":attr_r,"S":attr_s}[k])
                conf_improvement = float(conf_dnr[i,gold] - conf_raw[i,gold])

                flip_cases.append({
                    "dialogue_idx":    idx,
                    "utterance_pos":   i,
                    "text":            texts[i] if i < len(texts) else "",
                    "gold":            label_names_inv.get(gold, str(gold)),
                    "pred_raw":        label_names_inv.get(pred_raw, str(pred_raw)),
                    "pred_dnr":        label_names_inv.get(pred_dnr, str(pred_dnr)),
                    "conf_raw":        round(float(conf_raw[i,gold]), 4),
                    "conf_dnr":        round(float(conf_dnr[i,gold]), 4),
                    "conf_improvement":round(conf_improvement, 4),
                    "attr_U":          round(attr_u, 4),
                    "attr_R":          round(attr_r, 4),
                    "attr_S":          round(attr_s, 4),
                    "dominant_comp":   dom_comp,
                })

    df = pd.DataFrame(flip_cases).sort_values(
        "conf_improvement", ascending=False
    )
    df.to_csv(os.path.join(output_dir, "flip_positive_explained.csv"), index=False)

    # ── Print top examples with human-readable explanation ────────────────
    comp_descriptions = {
        "U": "unique modality-specific information",
        "R": "shared redundant information across modalities",
        "S": "synergistic cross-modal interaction",
    }

    print(f"\n=== Top {n_examples} Flip-Positive Cases with Explanation ===\n")
    for _, row in df.head(n_examples).iterrows():
        print(f"Text:       \"{row['text']}\"")
        print(f"Gold:        {row['gold']}")
        print(f"Raw pred:    {row['pred_raw']} "
              f"(confidence on gold: {row['conf_raw']:.3f})")
        print(f"DnR pred:    {row['pred_dnr']} "
              f"(confidence on gold: {row['conf_dnr']:.3f})")
        print(f"Improvement: +{row['conf_improvement']:.3f}")
        print(f"Attribution: U={row['attr_U']:.3f} "
              f"R={row['attr_R']:.3f} S={row['attr_S']:.3f}")
        print(f"Explanation: DnR correctly predicted [{row['gold']}] "
              f"by leveraging the [{row['dominant_comp']}] component "
              f"({comp_descriptions[row['dominant_comp']]}), "
              f"which was not explicitly available to the raw backbone.")
        print()

    return df
# Add to explainability_analysis.py

def modality_contribution_by_emotion(
    smurf_model, model_dnr, test_set, args, device, output_dir
):
    """
    For each utterance, measure how much each MODALITY contributes
    to the correct prediction by zeroing out one modality at a time.

    Unlike norm-based analysis, this captures the functional importance
    of each modality — how much the prediction CHANGES when that
    modality is removed.

    Attribution = baseline_prob_correct - ablated_prob_correct
    """
    os.makedirs(output_dir, exist_ok=True)

    label_dict      = args.dataset_label_dict[args.dataset]
    label_names_inv = {v: k for k, v in label_dict.items()}

    emb_dim_t  = args.embedding_dim[args.dataset]["t"]
    divide_dim = args.divide_dim
    use_cross  = (emb_dim_t // divide_dim == 7)

    smurf_model.eval()
    model_dnr.eval()

    records = []

    with torch.no_grad():
        for idx in range(len(test_set)):
            data = test_set[idx]
            for k, v in data.items():
                if k == "utterance_texts": continue
                if k == "tensor":
                    for m, feat in data[k].items():
                        data[k][m] = feat.to(device)
                else:
                    data[k] = v.to(device)

            labels = data["label_tensor"].cpu().numpy()

            x1 = data["tensor"]['t']
            x2 = data["tensor"]['a']
            x3 = data["tensor"]['v']
            textf   = (x1.permute(1, 2, 0)).transpose(1, 2)
            audiof  = (x2.permute(1, 2, 0)).transpose(1, 2)
            visualf = (x3.permute(1, 2, 0)).transpose(1, 2)
            m1, m2, m3, _ = smurf_model(textf, audiof, visualf)

            u1,r1,s1 = m1
            u2,r2,s2 = m2
            u3,r3,s3 = m3
            zero = torch.zeros_like(u1)

            def make_feat(u1,r1,s1, u2,r2,s2, u3,r3,s3):
                if use_cross:
                    ct = torch.cat([u1-u2,u1-u3,u1*u2,u1*u3],-1)
                    ca = torch.cat([u2-u1,u2-u3,u2*u1,u2*u3],-1)
                    cv = torch.cat([u3-u1,u3-u2,u3*u1,u3*u2],-1)
                    return (torch.cat([u1,r1,s1,ct],-1),
                            torch.cat([u2,r2,s2,ca],-1),
                            torch.cat([u3,r3,s3,cv],-1))
                return (torch.cat([u1,r1,s1],-1),
                        torch.cat([u2,r2,s2],-1),
                        torch.cat([u3,r3,s3],-1))

            def run_model(ft, fa, fv):
                d = copy.deepcopy(data)
                d["tensor"]['t'] = ft.transpose(0,1)
                d["tensor"]['a'] = fa.transpose(0,1)
                d["tensor"]['v'] = fv.transpose(0,1)
                prob, _, _ = model_dnr(d)
                return torch.softmax(prob, dim=-1).cpu().numpy()

            # Baseline
            ft,fa,fv = make_feat(u1,r1,s1, u2,r2,s2, u3,r3,s3)
            p_base = run_model(ft, fa, fv)

            # Zero out entire text modality
            ft_nt,fa_nt,fv_nt = make_feat(zero,zero,zero, u2,r2,s2, u3,r3,s3)
            p_no_t = run_model(ft_nt, fa_nt, fv_nt)

            # Zero out entire audio modality
            ft_na,fa_na,fv_na = make_feat(u1,r1,s1, zero,zero,zero, u3,r3,s3)
            p_no_a = run_model(ft_na, fa_na, fv_na)

            # Zero out entire video modality
            ft_nv,fa_nv,fv_nv = make_feat(u1,r1,s1, u2,r2,s2, zero,zero,zero)
            p_no_v = run_model(ft_nv, fa_nv, fv_nv)

            for i in range(len(labels)):
                gold    = int(labels[i])
                correct = (int(np.argmax(p_base[i])) == gold)
                if not correct:
                    continue   # only analyse correct predictions

                attr_t = float(p_base[i,gold] - p_no_t[i,gold])
                attr_a = float(p_base[i,gold] - p_no_a[i,gold])
                attr_v = float(p_base[i,gold] - p_no_v[i,gold])

                dominant = max(
                    {"text": attr_t, "audio": attr_a, "video": attr_v},
                    key=lambda k: {"text":attr_t,"audio":attr_a,"video":attr_v}[k]
                )

                records.append({
                    "emotion":   label_names_inv.get(gold, str(gold)),
                    "attr_text": round(attr_t, 4),
                    "attr_audio":round(attr_a, 4),
                    "attr_video":round(attr_v, 4),
                    "dominant":  dominant,
                })

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(output_dir, "modality_contribution.csv"), index=False)

    # ── Summary: mean attribution per emotion per modality ────────────────
    summary = df.groupby("emotion")[
        ["attr_text","attr_audio","attr_video"]
    ].mean().round(4)
    summary.to_csv(
        os.path.join(output_dir, "modality_contribution_summary.csv")
    )

    # Dominant modality percentage per emotion
    dom_pct = df.groupby(["emotion","dominant"]).size().unstack(fill_value=0)
    dom_pct = dom_pct.div(dom_pct.sum(axis=1), axis=0) * 100
    dom_pct = dom_pct.round(1)
    dom_pct.to_csv(
        os.path.join(output_dir, "modality_dominant_pct.csv")
    )

    print("\n=== Mean Modality Attribution by Emotion ===")
    print(summary.to_string())
    print("\n=== Dominant Modality % by Emotion ===")
    print(dom_pct.to_string())

    # ── Plot ──────────────────────────────────────────────────────────────
    _plot_modality_contribution(summary, output_dir)

    return df, summary, dom_pct


def _plot_modality_contribution(summary, output_dir):
    """Grouped bar chart: mean modality attribution per emotion."""
    emotions = summary.index.tolist()
    x = np.arange(len(emotions))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 4))

    bars_t = ax.bar(x - width, summary["attr_text"],  width,
                    label="Text",  color="#4575b4", alpha=0.85)
    bars_a = ax.bar(x,          summary["attr_audio"], width,
                    label="Audio", color="#d73027", alpha=0.85)
    bars_v = ax.bar(x + width, summary["attr_video"],  width,
                    label="Video", color="#1a9850", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(emotions, fontsize=11)
    ax.set_ylabel("Mean attribution\n(drop in correct-class prob)", fontsize=10)
    ax.set_title("Modality Contribution by Emotion Class (IEMOCAP)", fontsize=12)
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "modality_contribution.pdf"),
                bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "modality_contribution.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved modality contribution plot.")


# ══════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════

def run_explainability_analysis(
    smurf_model, model_raw, model_dnr,
    test_set, args, args_raw, device, output_dir
):
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*60)
    print("EXPLAINABILITY 1: Component Attribution")
    print("="*60)
    df_attr, df_attr_summary = component_attribution(
        smurf_model, model_dnr, test_set, args, device,
        os.path.join(output_dir, "attribution")
    )

    print("\n" + "="*60)
    print("EXPLAINABILITY 2: Modality Dominance by Emotion")
    print("="*60)
    df_mod, df_mod_summary = modality_dominance_by_emotion(
        smurf_model, test_set, args_raw, device,
        os.path.join(output_dir, "modality_dominance")
    )

    print("\n" + "="*60)
    print("EXPLAINABILITY 2b: Modality Contribution by Emotion")
    print("="*60)
    df_mod2, summary2, dom_pct2 = modality_contribution_by_emotion(
        smurf_model, model_dnr, test_set, args, device,
        os.path.join(output_dir, "modality_contribution")
    )

    print("\n" + "="*60)
    print("EXPLAINABILITY 3: Contrastive Flip Explanation")
    print("="*60)
    df_flip = contrastive_flip_explanation(
        smurf_model, model_raw, model_dnr,
        test_set, args, device,
        os.path.join(output_dir, "flip_explanation"),
        n_examples=5
    )

    return df_attr_summary, df_mod_summary, df_flip

