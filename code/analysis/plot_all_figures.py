"""
analysis/plot_all_figures.py

Generates three publication-ready figures:
  1. attribution_stacked.pdf   — Component attribution (U/R/S) by emotion
  2. modality_contribution.pdf — Modality attribution (T/A/V) by emotion
  3. perclass_improvement.pdf  — Per-class F1 delta bar chart

Usage:
  python code/analysis/plot_all_figures.py \
    --analysis_dir analysis_outputs/iemocap_mmgcn \
    --output_dir figures
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Consistent styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

# Preferred emotion order (arousal-based grouping)
EMOTION_ORDER = ["hap", "neu", "fru", "exc", "ang", "sad"]
EMOTION_DISPLAY = {
    "hap": "Happy", "neu": "Neutral", "fru": "Frustrated",
    "exc": "Excited", "ang": "Angry", "sad": "Sad",
}

COMPONENT_COLORS = {
    "U": "#4575b4",  # blue
    "R": "#d73027",  # red
    "S": "#1a9850",  # green
}

MODALITY_COLORS = {
    "Text":  "#4575b4",   # blue
    "Audio": "#d73027",   # red
    "Video": "#1a9850",   # green
}


def reorder_emotions(df, index_col="emotion"):
    """Reorder DataFrame rows to match EMOTION_ORDER."""
    if index_col in df.columns:
        df = df.set_index(index_col)
    ordered = [e for e in EMOTION_ORDER if e in df.index]
    return df.loc[ordered]


# ══════════════════════════════════════════════════════════════════════
#  Figure 1: Component Attribution Stacked Bar
# ══════════════════════════════════════════════════════════════════════

def plot_attribution_stacked(analysis_dir, output_dir):
    """
    Stacked horizontal bar: for each emotion, what % of correct
    predictions were dominated by U, R, or S.
    """
    csv_path = os.path.join(
        analysis_dir, "explainability", "attribution",
        "attribution_by_emotion.csv"
    )
    df = pd.read_csv(csv_path)

    # Pivot: rows=emotion, columns=component, values=dominant_%
    pivot = df.pivot(index="emotion", columns="component", values="dominant_%")
    pivot = reorder_emotions(pivot, "emotion")

    # Ensure all three columns exist
    for c in ["U", "R", "S"]:
        if c not in pivot.columns:
            pivot[c] = 0.0

    emotions = [EMOTION_DISPLAY.get(e, e) for e in pivot.index]
    u_vals = pivot["U"].values
    r_vals = pivot["R"].values
    s_vals = pivot["S"].values

    fig, ax = plt.subplots(figsize=(8, 4.5))

    y = np.arange(len(emotions))
    height = 0.55

    bars_u = ax.barh(y, u_vals, height,
                     label="Unique (U)", color=COMPONENT_COLORS["U"],
                     edgecolor="white", linewidth=0.5)
    bars_r = ax.barh(y, r_vals, height, left=u_vals,
                     label="Redundant (R)", color=COMPONENT_COLORS["R"],
                     edgecolor="white", linewidth=0.5)
    bars_s = ax.barh(y, s_vals, height, left=u_vals + r_vals,
                     label="Synergy (S)", color=COMPONENT_COLORS["S"],
                     edgecolor="white", linewidth=0.5)

    # Annotate percentages inside bars (only if > 10%)
    for bars, vals, lefts in [
        (bars_u, u_vals, np.zeros(len(y))),
        (bars_r, r_vals, u_vals),
        (bars_s, s_vals, u_vals + r_vals),
    ]:
        for i, (bar, val, left) in enumerate(zip(bars, vals, lefts)):
            if val > 10:
                cx = left + val / 2
                text_color = "white" if val > 20 else "black"
                ax.text(cx, y[i], f"{val:.0f}%",
                        ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color=text_color)

    ax.set_yticks(y)
    ax.set_yticklabels(emotions)
    ax.set_xlabel("Percentage of correct predictions (%)")
    ax.set_title("Dominant PID Component per Emotion Class")
    ax.set_xlim(0, 100)
    ax.legend(loc="lower right", framealpha=0.9)

    # Reference line at 33.3%
    ax.axvline(33.3, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.text(34, len(y) - 0.3, "uniform", color="gray",
            fontsize=8, fontstyle="italic")

    ax.invert_yaxis()
    plt.tight_layout()

    out = os.path.join(output_dir, "attribution_stacked")
    plt.savefig(f"{out}.pdf", bbox_inches="tight")
    plt.savefig(f"{out}.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {out}.pdf")


# ══════════════════════════════════════════════════════════════════════
#  Figure 2: Modality Contribution Grouped Bar
# ══════════════════════════════════════════════════════════════════════

def plot_modality_contribution(analysis_dir, output_dir):
    """
    Grouped bar chart: mean modality attribution per emotion.
    Positive = modality helps; negative = modality introduces noise.
    """
    csv_path = os.path.join(
        analysis_dir, "explainability", "modality_contribution",
        "modality_contribution_summary.csv"
    )
    df = pd.read_csv(csv_path, index_col=0)
    df.columns = ["Text", "Audio", "Video"]
    df = reorder_emotions(df, df.index.name or "emotion")

    emotions = [EMOTION_DISPLAY.get(e, e) for e in df.index]
    x = np.arange(len(emotions))
    width = 0.22
    offsets = [-width, 0, width]

    fig, ax = plt.subplots(figsize=(9, 4.5))

    for i, (mod, color) in enumerate(MODALITY_COLORS.items()):
        vals = df[mod].values
        bars = ax.bar(x + offsets[i], vals, width,
                      label=mod, color=color, alpha=0.85,
                      edgecolor="white", linewidth=0.5)

        # Annotate values on top of bars
        for bar, val in zip(bars, vals):
            bx = bar.get_x() + bar.get_width() / 2
            by = bar.get_height()
            # Position annotation above positive bars, below negative
            if val >= 0:
                ax.text(bx, by + 0.01, f"{val:.2f}",
                        ha="center", va="bottom", fontsize=7.5,
                        color=color, fontweight="bold")
            else:
                ax.text(bx, by - 0.01, f"{val:.2f}",
                        ha="center", va="top", fontsize=7.5,
                        color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(emotions)
    ax.set_ylabel("Mean attribution\n(drop in correct-class prob)")
    ax.set_title("Modality Contribution by Emotion Class")
    ax.legend(loc="upper right", framealpha=0.9)

    # Zero line
    ax.axhline(0, color="black", linewidth=0.6)

    # Light grid
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()

    out = os.path.join(output_dir, "modality_contribution")
    plt.savefig(f"{out}.pdf", bbox_inches="tight")
    plt.savefig(f"{out}.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {out}.pdf")


# ══════════════════════════════════════════════════════════════════════
#  Figure 3: Per-Class F1 Improvement
# ══════════════════════════════════════════════════════════════════════

def plot_perclass_improvement(analysis_dir, output_dir):
    """
    Horizontal bar chart: per-class F1 delta (DnR - Raw).
    Sorted by improvement. Green = gain, Red = regression.
    """
    csv_path = os.path.join(
        analysis_dir, "per_class", "per_class_improvement.csv"
    )
    df = pd.read_csv(csv_path)

    # Sort by delta ascending (so largest gain is at top visually)
    df = df.sort_values("delta", ascending=True)

    labels = [
        f"{EMOTION_DISPLAY.get(e, e)}\n(n={n})"
        for e, n in zip(df["emotion"], df["n_samples"])
    ]

    colors = [
        "#d73027" if d < -0.001 else "#1a9850" if d > 0.001 else "#999999"
        for d in df["delta"]
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    bars = ax.barh(labels, df["delta"] * 100,
                   color=colors, edgecolor="white",
                   height=0.55, linewidth=0.5)

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("ΔW-F1 (%)")
    ax.set_title("Per-Class W-F1 Improvement: DnR vs Raw Backbone")

    # Annotate each bar with its value
    for bar, val in zip(bars, df["delta"] * 100):
        x = bar.get_width()
        ha = "left" if x >= 0 else "right"
        offset = 0.3 if x >= 0 else -0.3
        ax.text(x + offset,
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.1f}%",
                va="center", ha=ha, fontsize=10, fontweight="bold")

    # Legend
    ax.legend(
        handles=[
            mpatches.Patch(color="#1a9850", label="Improvement"),
            mpatches.Patch(color="#d73027", label="Regression"),
        ],
        loc="lower right", framealpha=0.9,
    )

    # Light grid
    ax.xaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()

    out = os.path.join(output_dir, "perclass_improvement")
    plt.savefig(f"{out}.pdf", bbox_inches="tight")
    plt.savefig(f"{out}.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {out}.pdf")


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-ready analysis figures."
    )
    parser.add_argument(
        "--analysis_dir", type=str,
        default="analysis_outputs/iemocap_mmgcn",
        help="Root directory of analysis outputs.",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="figures",
        help="Directory to save generated figures.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Generating figures from: {args.analysis_dir}")
    print(f"Saving to: {args.output_dir}\n")

    print("Figure 1: Component Attribution")
    plot_attribution_stacked(args.analysis_dir, args.output_dir)

    print("Figure 2: Modality Contribution")
    plot_modality_contribution(args.analysis_dir, args.output_dir)

    print("Figure 3: Per-Class Improvement")
    plot_perclass_improvement(args.analysis_dir, args.output_dir)

    print(f"\nAll figures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()