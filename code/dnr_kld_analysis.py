import copy
import csv
import json
import os

import torch
import torch.nn.functional as F


# Human-readable names for the three decomposition heads produced by SMURF.
COMPONENTS = [("U", 0), ("R", 1), ("S", 2)]
MODALITY_NAMES = {"t": "Text", "a": "Audio", "v": "Visual"}


def _ensure_dir(path):
    """Create the analysis output directory once before writing reports."""
    os.makedirs(path, exist_ok=True)


def _seq_batch_features(data):
    """Convert dataloader tensors from [batch, seq, dim] to [seq, batch, dim]."""
    textf = data["tensor"]["t"].permute(1, 0, 2)
    audiof = data["tensor"]["a"].permute(1, 0, 2)
    visualf = data["tensor"]["v"].permute(1, 0, 2)
    return textf, audiof, visualf


def _flatten_valid(x, lengths):
    """Flatten [seq, batch, dim] features and remove padded utterances."""
    seq_len = x.shape[0]
    steps = torch.arange(seq_len, device=x.device).unsqueeze(1)
    mask = steps < lengths.to(x.device).unsqueeze(0)
    return x[mask]


def _component_distribution(x, tau):
    """Treat a high-dimensional component vector as a softened distribution."""
    return F.log_softmax(x / tau, dim=-1), F.softmax(x / tau, dim=-1)


def _kl_from_log_prob(log_p, p, log_q):
    """Compute KL(P || Q) for already-normalized log probabilities."""
    return (p * (log_p - log_q)).sum(dim=-1)


def _symmetric_component_kl(x, y, tau):
    """Use symmetric KLD so component distances are easier to compare."""
    log_x, prob_x = _component_distribution(x, tau)
    log_y, prob_y = _component_distribution(y, tau)
    kl_xy = _kl_from_log_prob(log_x, prob_x, log_y)
    kl_yx = _kl_from_log_prob(log_y, prob_y, log_x)
    return 0.5 * (kl_xy + kl_yx)


def _prediction_kl(full_log_prob, removed_log_prob):
    """Measure how much the prediction distribution changes after masking."""
    full_prob = full_log_prob.exp()
    return _kl_from_log_prob(full_log_prob, full_prob, removed_log_prob)


def _mean_or_zero(values):
    """Return a Python float while handling empty class-specific slices."""
    if values.numel() == 0:
        return 0.0
    return values.mean().item()


def _move_batch_to_device(data, device):
    """Mirror the training/evaluation device transfer for one dataloader batch."""
    for k, v in data.items():
        if k == "utterance_texts":
            continue
        if k == "tensor":
            for m, feat in data[k].items():
                data[k][m] = feat.to(device)
        else:
            data[k] = v.to(device)
    return data


def _smurf_components(data, smurf_model):
    """Run SMURF once and return modality-keyed U/R/S component tuples."""
    textf, audiof, visualf = _seq_batch_features(data)
    m_text, m_audio, m_visual, _ = smurf_model(textf, audiof, visualf)
    return {"t": m_text, "a": m_audio, "v": m_visual}


def _refined_data_from_components(data, components, remove=None):
    """
    Build a normal model input from U/R/S components.

    remove can be a tuple like ("t", "S") to zero one component. This supports
    prediction-sensitivity KLD without changing the trained model.
    """
    refined = copy.copy(data)
    refined["tensor"] = {}

    remove_modality, remove_component = remove if remove is not None else (None, None)
    for modality, parts in components.items():
        new_parts = []
        for name, index in COMPONENTS:
            part = parts[index]
            if modality == remove_modality and name == remove_component:
                part = torch.zeros_like(part)
            new_parts.append(part)
        refined["tensor"][modality] = torch.cat(new_parts, dim=-1).transpose(0, 1)

    return refined


def _label_name(label_dict, label_id):
    """Map a numeric class id back to the dataset label string."""
    inverse = {v: k for k, v in label_dict.items()}
    return inverse.get(int(label_id), str(int(label_id)))


def _add_pairwise_kld(rows, components, lengths, labels, label_dict, tau):
    """Append class-conditioned symmetric KLD rows for all component pairs."""
    flat_components = {}
    for modality, parts in components.items():
        for component_name, component_index in COMPONENTS:
            key = f"{modality}-{component_name}"
            flat_components[key] = _flatten_valid(parts[component_index], lengths)

    class_ids = sorted(label_dict.values())
    keys = list(flat_components.keys())
    for i, left_key in enumerate(keys):
        for right_key in keys[i + 1:]:
            values = _symmetric_component_kl(
                flat_components[left_key],
                flat_components[right_key],
                tau,
            )
            row = {
                "left": left_key,
                "right": right_key,
                "pair": f"{left_key}__{right_key}",
                "mean_skl": values.mean().item(),
                "count": int(values.numel()),
            }
            for class_id in class_ids:
                class_mask = labels == class_id
                row[f"mean_skl_{_label_name(label_dict, class_id)}"] = _mean_or_zero(
                    values[class_mask]
                )
            rows.append(row)


def _add_prediction_sensitivity(rows, model, data, components, label_dict):
    """Append KLD rows showing prediction sensitivity to each U/R/S component."""
    full_data = _refined_data_from_components(data, components)
    full_log_prob, _, _ = model(full_data)
    labels = data["label_tensor"]

    class_ids = sorted(label_dict.values())
    for modality in components.keys():
        for component_name, _ in COMPONENTS:
            removed_data = _refined_data_from_components(
                data, components, remove=(modality, component_name)
            )
            removed_log_prob, _, _ = model(removed_data)
            values = _prediction_kl(full_log_prob, removed_log_prob)

            row = {
                "removed_component": f"{modality}-{component_name}",
                "mean_kl_full_to_removed": values.mean().item(),
                "count": int(values.numel()),
            }
            for class_id in class_ids:
                class_mask = labels == class_id
                row[f"mean_kl_{_label_name(label_dict, class_id)}"] = _mean_or_zero(
                    values[class_mask]
                )
            rows.append(row)


def _write_csv(path, rows):
    """Write rows with a union of all keys so class-specific columns are retained."""
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summarize_rows(rows, key_field, value_field):
    """Create compact JSON summaries, weighted by utterance count when present."""
    totals = {}
    counts = {}
    for row in rows:
        key = row[key_field]
        weight = row.get("count", 1)
        totals[key] = totals.get(key, 0.0) + row[value_field] * weight
        counts[key] = counts.get(key, 0) + weight
    return {key: totals[key] / counts[key] for key in totals}


def run_dnr_kld_analysis(model, smurf_model, dataset, args, split_name="test"):
    """
    Run KLD-based DnR interpretability analysis.

    Outputs:
    - pairwise_component_skl.csv: distributional separation among U/R/S components.
    - prediction_sensitivity_kl.csv: prediction change after removing one component.
    - summary.json: compact averages for quick reporting.
    """
    if smurf_model is None:
        raise ValueError("KLD analysis requires --use_divide and --use_refine.")

    output_dir = os.path.join(args.analysis_dir, args.dataset, split_name)
    _ensure_dir(output_dir)

    device = args.device
    tau = args.analysis_tau
    label_dict = args.dataset_label_dict[args.dataset]

    pairwise_rows = []
    sensitivity_rows = []

    model.eval()
    smurf_model.eval()
    with torch.no_grad():
        for idx in range(len(dataset)):
            data = _move_batch_to_device(dataset[idx], device)
            components = _smurf_components(data, smurf_model)

            batch_pairwise_rows = []
            _add_pairwise_kld(
                batch_pairwise_rows,
                components,
                data["length"],
                data["label_tensor"],
                label_dict,
                tau,
            )
            for row in batch_pairwise_rows:
                row["batch_index"] = idx
            pairwise_rows.extend(batch_pairwise_rows)

            batch_sensitivity_rows = []
            _add_prediction_sensitivity(
                batch_sensitivity_rows,
                model,
                data,
                components,
                label_dict,
            )
            for row in batch_sensitivity_rows:
                row["batch_index"] = idx
            sensitivity_rows.extend(batch_sensitivity_rows)

    pairwise_path = os.path.join(output_dir, "pairwise_component_skl.csv")
    sensitivity_path = os.path.join(output_dir, "prediction_sensitivity_kl.csv")
    summary_path = os.path.join(output_dir, "summary.json")

    _write_csv(pairwise_path, pairwise_rows)
    _write_csv(sensitivity_path, sensitivity_rows)

    summary = {
        "dataset": args.dataset,
        "split": split_name,
        "temperature": tau,
        "pairwise_component_skl": _summarize_rows(
            pairwise_rows, "pair", "mean_skl"
        ),
        "prediction_sensitivity_kl": _summarize_rows(
            sensitivity_rows,
            "removed_component",
            "mean_kl_full_to_removed",
        ),
        "files": {
            "pairwise_component_skl": pairwise_path,
            "prediction_sensitivity_kl": sensitivity_path,
        },
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary
