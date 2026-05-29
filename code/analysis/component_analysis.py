"""
analysis/component_analysis.py

Analyses what information each PID component (U, R, S) captures
by measuring:
  1. Per-component emotion class separability (Fisher's criterion)
  2. Cross-modal alignment of R components (should be high)
  3. Cross-modal divergence of U components (should be high)
  4. Predictive utility of each component alone (single-head accuracy)

Produces:
  - component_separability.csv
  - cross_modal_alignment.csv
  - single_head_accuracy.csv
  - component_norm_by_class.csv
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from new_divide_decomp import ThreeModalityModel
from dataloader import Dataloader, load_iemocap, load_meld, base_dataset_name


def extract_components(smurf_model, dataset, args, device):
    """
    Run SMURF over the full dataset and collect all components.
    
    Returns
    -------
    components : dict with keys
        'u_t', 'r_t', 's_t',
        'u_a', 'r_a', 's_a',
        'u_v', 'r_v', 's_v'
        each a numpy array of shape [N, d]
    labels : numpy array [N]
    """
    smurf_model.eval()
    
    all_components = defaultdict(list)
    all_labels = []
    
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
            
            x1 = data["tensor"]['t']
            x2 = data["tensor"]['a']
            x3 = data["tensor"]['v']
            textf   = (x1.permute(1, 2, 0)).transpose(1, 2)
            audiof  = (x2.permute(1, 2, 0)).transpose(1, 2)
            visualf = (x3.permute(1, 2, 0)).transpose(1, 2)
            
            m1, m2, m3, _ = smurf_model(textf, audiof, visualf)
            
            # Flatten valid utterances
            lengths = data["length"]
            
            def flatten(x, lengths):
                """[seq, batch, dim] → [N_valid, dim]"""
                seq_len = x.shape[0]
                steps = torch.arange(seq_len, device=x.device).unsqueeze(1)
                mask  = steps < lengths.to(x.device).unsqueeze(0)
                return x[mask]
            
            u_t = flatten(m1[0], lengths).cpu().numpy()
            r_t = flatten(m1[1], lengths).cpu().numpy()
            s_t = flatten(m1[2], lengths).cpu().numpy()
            u_a = flatten(m2[0], lengths).cpu().numpy()
            r_a = flatten(m2[1], lengths).cpu().numpy()
            s_a = flatten(m2[2], lengths).cpu().numpy()
            u_v = flatten(m3[0], lengths).cpu().numpy()
            r_v = flatten(m3[1], lengths).cpu().numpy()
            s_v = flatten(m3[2], lengths).cpu().numpy()
            
            labels = data["label_tensor"].cpu().numpy()
            
            all_components['u_t'].append(u_t)
            all_components['r_t'].append(r_t)
            all_components['s_t'].append(s_t)
            all_components['u_a'].append(u_a)
            all_components['r_a'].append(r_a)
            all_components['s_a'].append(s_a)
            all_components['u_v'].append(u_v)
            all_components['r_v'].append(r_v)
            all_components['s_v'].append(s_v)
            all_labels.append(labels)
    
    # Concatenate all batches
    components = {k: np.concatenate(v, axis=0) for k, v in all_components.items()}
    labels = np.concatenate(all_labels, axis=0)
    
    return components, labels


def compute_fisher_criterion(X, y):
    """
    Fisher's Linear Discriminant criterion:
    ratio of between-class to within-class scatter.
    Higher = better class separability.
    
    Uses mean of the ratio across PCA-projected dimensions.
    """
    classes = np.unique(y)
    overall_mean = X.mean(axis=0)
    
    S_B = np.zeros((X.shape[1], X.shape[1]))  # between-class scatter
    S_W = np.zeros((X.shape[1], X.shape[1]))  # within-class scatter
    
    for c in classes:
        X_c = X[y == c]
        n_c = len(X_c)
        mean_c = X_c.mean(axis=0)
        diff = (mean_c - overall_mean).reshape(-1, 1)
        S_B += n_c * (diff @ diff.T)
        S_W += (X_c - mean_c).T @ (X_c - mean_c)
    
    # Scalar criterion: tr(S_B) / tr(S_W)
    tr_SW = np.trace(S_W)
    if tr_SW < 1e-10:
        return 0.0
    return np.trace(S_B) / tr_SW


def compute_cosine_alignment(A, B):
    """
    Mean cosine similarity between corresponding rows of A and B.
    High = the two components encode similar content.
    """
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return float((A_norm * B_norm).sum(axis=1).mean())


def compute_single_head_accuracy(X, y, n_splits=5):
    """
    Train a linear classifier on X and evaluate on held-out split.
    Returns mean accuracy across folds.
    
    Uses LDA (fast, interpretable, no hyperparameters).
    """
    from sklearn.model_selection import StratifiedKFold
    
    le  = LabelEncoder()
    y_  = le.fit_transform(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    accs = []
    for train_idx, test_idx in skf.split(X, y_):
        clf = LinearDiscriminantAnalysis()
        try:
            clf.fit(X[train_idx], y_[train_idx])
            pred = clf.predict(X[test_idx])
            accs.append(accuracy_score(y_[test_idx], pred))
        except Exception:
            accs.append(0.0)
    return float(np.mean(accs))


def component_norm_by_class(X, y, label_names):
    """
    Mean L2 norm of component per emotion class.
    Reveals which classes activate each component most strongly.
    """
    rows = []
    for c, name in label_names.items():
        mask = y == c
        if mask.sum() == 0:
            continue
        mean_norm = float(np.linalg.norm(X[mask], axis=1).mean())
        rows.append({"class": name, "mean_norm": mean_norm})
    return pd.DataFrame(rows)


def run_component_analysis(smurf_model, test_set, args, output_dir, device):
    """Main analysis function."""
    os.makedirs(output_dir, exist_ok=True)
    
    label_names = args.dataset_label_dict[args.dataset]
    label_names_inv = {v: k for k, v in label_names.items()}
    
    print("Extracting components...")
    components, labels = extract_components(smurf_model, test_set, args, device)
    
    comp_names = list(components.keys())
    
    # ── 1. Fisher criterion (class separability) ──────────────────────────
    print("Computing Fisher criterion...")
    fisher_rows = []
    for name in comp_names:
        score = compute_fisher_criterion(components[name], labels)
        fisher_rows.append({"component": name, "fisher_criterion": score})
    
    df_fisher = pd.DataFrame(fisher_rows).sort_values("fisher_criterion", ascending=False)
    df_fisher.to_csv(os.path.join(output_dir, "component_separability.csv"), index=False)
    print("\n=== Component Separability (Fisher Criterion) ===")
    print(df_fisher.to_string(index=False))
    
    # ── 2. Cross-modal alignment of R and U ──────────────────────────────
    print("\nComputing cross-modal alignment...")
    alignment_rows = []
    
    # R should be aligned (high cosine similarity)
    alignment_rows.append({
        "pair": "r_t vs r_a",
        "cosine_sim": compute_cosine_alignment(components['r_t'], components['r_a']),
        "type": "redundant"
    })
    alignment_rows.append({
        "pair": "r_t vs r_v",
        "cosine_sim": compute_cosine_alignment(components['r_t'], components['r_v']),
        "type": "redundant"
    })
    alignment_rows.append({
        "pair": "r_a vs r_v",
        "cosine_sim": compute_cosine_alignment(components['r_a'], components['r_v']),
        "type": "redundant"
    })
    
    # U should be diverse (low cosine similarity)
    alignment_rows.append({
        "pair": "u_t vs u_a",
        "cosine_sim": compute_cosine_alignment(components['u_t'], components['u_a']),
        "type": "unique"
    })
    alignment_rows.append({
        "pair": "u_t vs u_v",
        "cosine_sim": compute_cosine_alignment(components['u_t'], components['u_v']),
        "type": "unique"
    })
    alignment_rows.append({
        "pair": "u_a vs u_v",
        "cosine_sim": compute_cosine_alignment(components['u_a'], components['u_v']),
        "type": "unique"
    })
    
    # S should be aligned (synergy is a joint property)
    alignment_rows.append({
        "pair": "s_t vs s_a",
        "cosine_sim": compute_cosine_alignment(components['s_t'], components['s_a']),
        "type": "synergy"
    })
    alignment_rows.append({
        "pair": "s_t vs s_v",
        "cosine_sim": compute_cosine_alignment(components['s_t'], components['s_v']),
        "type": "synergy"
    })
    alignment_rows.append({
        "pair": "s_a vs s_v",
        "cosine_sim": compute_cosine_alignment(components['s_a'], components['s_v']),
        "type": "synergy"
    })
    
    df_align = pd.DataFrame(alignment_rows)
    df_align.to_csv(os.path.join(output_dir, "cross_modal_alignment.csv"), index=False)
    print("\n=== Cross-Modal Alignment ===")
    print(df_align.to_string(index=False))
    
    # ── 3. Single-head predictive accuracy ───────────────────────────────
    print("\nComputing single-head accuracy...")
    acc_rows = []
    for name in comp_names:
        acc = compute_single_head_accuracy(components[name], labels)
        modality = name.split("_")[1]  # t, a, or v
        comp_type = name.split("_")[0]  # u, r, or s
        acc_rows.append({
            "component": name,
            "modality": modality,
            "type": comp_type,
            "lda_accuracy": acc,
        })
    
    df_acc = pd.DataFrame(acc_rows).sort_values("lda_accuracy", ascending=False)
    df_acc.to_csv(os.path.join(output_dir, "single_head_accuracy.csv"), index=False)
    print("\n=== Single-Head LDA Accuracy ===")
    print(df_acc.to_string(index=False))
    
    # ── 4. Component norm by emotion class ───────────────────────────────
    print("\nComputing component norms by emotion class...")
    norm_dfs = []
    for name in comp_names:
        df = component_norm_by_class(components[name], labels, label_names_inv)
        df["component"] = name
        norm_dfs.append(df)
    
    df_norms = pd.concat(norm_dfs, ignore_index=True)
    df_norms.to_csv(os.path.join(output_dir, "component_norm_by_class.csv"), index=False)
    
    print(f"\nAll outputs saved to: {output_dir}")
    return {
        "fisher":    df_fisher,
        "alignment": df_align,
        "accuracy":  df_acc,
        "norms":     df_norms,
    }