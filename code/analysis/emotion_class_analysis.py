"""
analysis/emotion_class_analysis.py
"""

import torch
import numpy as np
import pandas as pd
from sklearn import metrics
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_utils import apply_smurf_to_data


def evaluate_with_breakdown(model, smurf_model, dataset, args, device):
    """
    Evaluate model and return per-class metrics.

    If smurf_model is not None and args.use_divide and args.use_refine,
    applies SMURF feature construction (with or without cross-modal features,
    detected automatically from args.embedding_dim).
    """
    model.eval()
    criterion = torch.nn.NLLLoss()

    label_dict  = args.dataset_label_dict[args.dataset]
    label_names = list(label_dict.keys())

    golds, preds = [], []

    with torch.no_grad():
        for idx in range(len(dataset)):
            data = dataset[idx]
            for k, v in data.items():
                if k == "utterance_texts":
                    continue
                if k == "tensor":
                    for m, feat in data[k].items():
                        data[k][m] = feat.to(device)
                else:
                    data[k] = v.to(device)

            if smurf_model is not None and args.use_divide and args.use_refine:
                data, _, _, _ = apply_smurf_to_data(
                    data, smurf_model, args, device
                )

            labels = data["label_tensor"]
            golds.append(labels.cpu())
            prob, _, _ = model(data)
            preds.append(torch.argmax(prob, dim=-1).detach().cpu())

    golds = torch.cat(golds).numpy()
    preds = torch.cat(preds).numpy()

    report = metrics.classification_report(
        golds, preds,
        target_names=label_names,
        output_dict=True,
        digits=4,
    )
    per_class_f1 = {
        name: report[name]["f1-score"]
        for name in label_names
        if name in report
    }
    return golds, preds, per_class_f1


def compute_per_class_improvement(
    model_raw, model_dnr, smurf_model,
    test_set, args_raw, args_dnr,
    device, output_dir,
):
    os.makedirs(output_dir, exist_ok=True)

    label_dict  = args_raw.dataset_label_dict[args_raw.dataset]
    label_names = list(label_dict.keys())

    print("Evaluating raw backbone...")
    golds_raw, preds_raw, f1_raw = evaluate_with_breakdown(
        model_raw, None, test_set, args_raw, device
    )

    print("Evaluating DnR backbone...")
    golds_dnr, preds_dnr, f1_dnr = evaluate_with_breakdown(
        model_dnr, smurf_model, test_set, args_dnr, device
    )

    rows = []
    for name in label_names:
        raw       = f1_raw.get(name, 0.0)
        dnr       = f1_dnr.get(name, 0.0)
        n_samples = int((golds_raw == label_dict[name]).sum())
        rows.append({
            "emotion":   name,
            "f1_raw":    round(raw,       4),
            "f1_dnr":    round(dnr,       4),
            "delta":     round(dnr - raw, 4),
            "n_samples": n_samples,
        })

    df = pd.DataFrame(rows).sort_values("delta", ascending=False)
    df.to_csv(os.path.join(output_dir, "per_class_improvement.csv"), index=False)

    print("\n=== Per-Class F1 Improvement ===")
    print(df.to_string(index=False))

    pd.DataFrame(
        metrics.confusion_matrix(golds_raw, preds_raw),
        index=label_names, columns=label_names
    ).to_csv(os.path.join(output_dir, "confusion_matrix_raw.csv"))

    pd.DataFrame(
        metrics.confusion_matrix(golds_dnr, preds_dnr),
        index=label_names, columns=label_names
    ).to_csv(os.path.join(output_dir, "confusion_matrix_dnr.csv"))

    return df