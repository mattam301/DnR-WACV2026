"""
analysis/case_study_miner.py
"""

import torch
import numpy as np
import pandas as pd
from collections import defaultdict
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_utils import apply_smurf_to_data


def kl_divergence_vectors(p, q, eps=1e-8):
    p = np.abs(p) + eps
    q = np.abs(q) + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def mine_cases(
    model_raw, model_dnr, smurf_model,
    test_set, args,          # args here is args_dnr
    device, output_dir,
    n_cases_per_type=10,
):
    os.makedirs(output_dir, exist_ok=True)

    label_dict      = args.dataset_label_dict[args.dataset]
    label_names_inv = {v: k for k, v in label_dict.items()}

    model_raw.eval()
    model_dnr.eval()
    smurf_model.eval()

    cases        = []
    utterance_id = 0

    with torch.no_grad():
        for idx in range(len(test_set)):

            # ── Load raw data ────────────────────────────────────────────
            data_raw = test_set[idx]
            for k, v in data_raw.items():
                if k == "utterance_texts":
                    continue
                if k == "tensor":
                    for m, feat in data_raw[k].items():
                        data_raw[k][m] = feat.to(device)
                else:
                    data_raw[k] = v.to(device)

            labels  = data_raw["label_tensor"].cpu().numpy()
            lengths = data_raw["length"]
            texts   = data_raw.get("utterance_texts", None)

            # ── Raw backbone ─────────────────────────────────────────────
            prob_raw, _, _ = model_raw(data_raw)
            preds_raw      = torch.argmax(prob_raw, dim=-1).cpu().numpy()
            conf_raw       = torch.softmax(prob_raw, dim=-1).cpu().numpy()

            # ── SMURF + DnR backbone ─────────────────────────────────────
            # Deep-copy so raw data is untouched
            data_dnr = copy.deepcopy(data_raw)
            data_dnr, m1, m2, m3 = apply_smurf_to_data(
                data_dnr, smurf_model, args, device
            )

            prob_dnr  = model_dnr(data_dnr)[0]
            preds_dnr = torch.argmax(prob_dnr, dim=-1).cpu().numpy()
            conf_dnr  = torch.softmax(prob_dnr, dim=-1).cpu().numpy()

            # ── Flatten components ───────────────────────────────────────
            def flatten(x, lengths):
                seq_len = x.shape[0]
                steps   = torch.arange(seq_len, device=x.device).unsqueeze(1)
                mask    = steps < lengths.to(x.device).unsqueeze(0)
                return x[mask].cpu().numpy()

            u_t = flatten(m1[0], lengths)
            r_t = flatten(m1[1], lengths)
            s_t = flatten(m1[2], lengths)
            u_a = flatten(m2[0], lengths)
            r_a = flatten(m2[1], lengths)
            s_a = flatten(m2[2], lengths)
            u_v = flatten(m3[0], lengths)
            r_v = flatten(m3[1], lengths)
            s_v = flatten(m3[2], lengths)

            # ── Per-utterance records ────────────────────────────────────
            for i in range(len(labels)):
                gold      = int(labels[i])
                pred_raw  = int(preds_raw[i])
                pred_dnr  = int(preds_dnr[i])

                gold_name     = label_names_inv.get(gold,     str(gold))
                pred_raw_name = label_names_inv.get(pred_raw, str(pred_raw))
                pred_dnr_name = label_names_inv.get(pred_dnr, str(pred_dnr))

                raw_correct = (pred_raw == gold)
                dnr_correct = (pred_dnr == gold)

                if raw_correct and dnr_correct:
                    case_type = "both_correct"
                elif not raw_correct and dnr_correct:
                    case_type = "flip_positive"
                elif raw_correct and not dnr_correct:
                    case_type = "flip_negative"
                else:
                    case_type = "both_wrong"

                kl_u_r_text   = kl_divergence_vectors(u_t[i], r_t[i])
                kl_u_r_audio  = kl_divergence_vectors(u_a[i], r_a[i])
                kl_u_r_visual = kl_divergence_vectors(u_v[i], r_v[i])

                u_norm_t  = float(np.linalg.norm(u_t[i]))
                r_norm_t  = float(np.linalg.norm(r_t[i]))
                s_norm_t  = float(np.linalg.norm(s_t[i]))
                s_norm_all = float(
                    np.linalg.norm(s_t[i]) +
                    np.linalg.norm(s_a[i]) +
                    np.linalg.norm(s_v[i])
                ) / 3.0

                cos_r_ta = float(np.dot(r_t[i], r_a[i]) / (
                    np.linalg.norm(r_t[i]) * np.linalg.norm(r_a[i]) + 1e-8
                ))
                cos_r_tv = float(np.dot(r_t[i], r_v[i]) / (
                    np.linalg.norm(r_t[i]) * np.linalg.norm(r_v[i]) + 1e-8
                ))

                text_content = ""
                if texts is not None and i < len(texts):
                    text_content = texts[i]

                cases.append({
                    "utterance_id":      utterance_id,
                    "dialogue_idx":      idx,
                    "position_in_dial":  i,
                    "gold":              gold_name,
                    "pred_raw":          pred_raw_name,
                    "pred_dnr":          pred_dnr_name,
                    "case_type":         case_type,
                    "conf_raw_correct":  float(conf_raw[i, gold]),
                    "conf_dnr_correct":  float(conf_dnr[i, gold]),
                    "kl_u_r_text":       round(kl_u_r_text,   4),
                    "kl_u_r_audio":      round(kl_u_r_audio,  4),
                    "kl_u_r_visual":     round(kl_u_r_visual, 4),
                    "u_norm_text":       round(u_norm_t,  4),
                    "r_norm_text":       round(r_norm_t,  4),
                    "s_norm_text":       round(s_norm_t,  4),
                    "s_norm_mean":       round(s_norm_all, 4),
                    "cos_r_text_audio":  round(cos_r_ta,  4),
                    "cos_r_text_visual": round(cos_r_tv,  4),
                    "text":              text_content,
                })
                utterance_id += 1

    df = pd.DataFrame(cases)
    df.to_csv(os.path.join(output_dir, "all_cases.csv"), index=False)

    # ── Select most informative cases ────────────────────────────────────

    # 1. Flip positives
    df_flip_pos = df[df["case_type"] == "flip_positive"].copy()
    df_flip_pos["conf_improvement"] = (
        df_flip_pos["conf_dnr_correct"] - df_flip_pos["conf_raw_correct"]
    )
    df_flip_pos.sort_values("conf_improvement", ascending=False).head(
        n_cases_per_type
    ).to_csv(os.path.join(output_dir, "cases_flip_positive.csv"), index=False)

    # 2. Flip negatives
    df_flip_neg = df[df["case_type"] == "flip_negative"].copy()
    df_flip_neg["conf_drop"] = (
        df_flip_neg["conf_raw_correct"] - df_flip_neg["conf_dnr_correct"]
    )
    df_flip_neg.sort_values("conf_drop", ascending=False).head(
        n_cases_per_type
    ).to_csv(os.path.join(output_dir, "cases_flip_negative.csv"), index=False)

    # 3. Redundancy-collapse cases (low KL between U and R)
    df.sort_values("kl_u_r_text").head(n_cases_per_type * 2).to_csv(
        os.path.join(output_dir, "cases_redundancy_collapse.csv"), index=False
    )

    # 4. High-synergy cases
    df.sort_values("s_norm_mean", ascending=False).head(n_cases_per_type).to_csv(
        os.path.join(output_dir, "cases_high_synergy.csv"), index=False
    )

    # 5. Cross-modal alignment extremes
    df.sort_values("cos_r_text_audio", ascending=False).head(n_cases_per_type).to_csv(
        os.path.join(output_dir, "cases_high_alignment.csv"), index=False
    )
    df.sort_values("cos_r_text_audio").head(n_cases_per_type).to_csv(
        os.path.join(output_dir, "cases_low_alignment.csv"), index=False
    )

    # ── Summary ──────────────────────────────────────────────────────────
    summary = {
        "total_utterances":   len(df),
        "both_correct":       int((df["case_type"] == "both_correct").sum()),
        "flip_positive":      int((df["case_type"] == "flip_positive").sum()),
        "flip_negative":      int((df["case_type"] == "flip_negative").sum()),
        "both_wrong":         int((df["case_type"] == "both_wrong").sum()),
        "mean_kl_u_r_text":   float(df["kl_u_r_text"].mean()),
        "mean_kl_u_r_audio":  float(df["kl_u_r_audio"].mean()),
        "mean_cos_r_ta":      float(df["cos_r_text_audio"].mean()),
        "flip_pos_kl_mean":   float(df_flip_pos["kl_u_r_text"].mean())
                              if len(df_flip_pos) > 0 else 0.0,
        "flip_neg_kl_mean":   float(df_flip_neg["kl_u_r_text"].mean())
                              if len(df_flip_neg) > 0 else 0.0,
    }
    pd.DataFrame([summary]).to_csv(
        os.path.join(output_dir, "summary.csv"), index=False
    )

    print("\n=== Case Mining Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    return df, summary