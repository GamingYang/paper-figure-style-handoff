"""Generate paper-style figures for the attention mechanism ablation suite."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"C:\Users\ROG\Desktop\paper")
CODEX_DIR = ROOT / "codex"
OUT_DIR = CODEX_DIR / "adaptive-attention-mechanism-ablation-v1"
FIG_DIR = OUT_DIR / "figures"
ADAPTIVE_DIR = CODEX_DIR / "adaptive-attention-v1"


METHOD_ORDER = [
    "adaptive-attention-v1",
    "original-attention",
    "no-degradation-state",
    "no-curve-tokens",
    "fixed-temperature",
    "shuffled-degradation-state",
    "voltage-only-curve-token",
    "cycle-only-state",
]

METHOD_LABELS = {
    "adaptive-attention-v1": "Proposed",
    "original-attention": "Original",
    "no-degradation-state": "w/o state",
    "no-curve-tokens": "w/o curve",
    "fixed-temperature": "fixed tau",
    "shuffled-degradation-state": "shuffled state",
    "voltage-only-curve-token": "voltage only",
    "cycle-only-state": "cycle only",
}

METHOD_COLORS = {
    "adaptive-attention-v1": "#2D9CDB",
    "original-attention": "#A8DADC",
    "no-degradation-state": "#8E7CC3",
    "no-curve-tokens": "#F4A261",
    "fixed-temperature": "#E76F51",
    "shuffled-degradation-state": "#8D99AE",
    "voltage-only-curve-token": "#95D5B2",
    "cycle-only-state": "#B7E4C7",
    "no-physical-base": "#C1121F",
    "no-mmd": "#6C757D",
    "no-attention": "#9D4EDD",
}


def set_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "axes.linewidth": 0.9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.unicode_minus": False,
        }
    )


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(np.mean((y_true.reshape(-1) - y_pred.reshape(-1)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true.reshape(-1) - y_pred.reshape(-1))))


def load_seed42_per_cell() -> pd.DataFrame:
    pred_path = ADAPTIVE_DIR / "adaptive_attention_v1_seed42_target_test_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    full_pred = pd.read_csv(pred_path)
    rows: List[Dict] = []
    for b, g in full_pred.groupby("battery_index"):
        true = g["true_soh"].to_numpy()
        rows.append(
            {
                "battery_index": int(b),
                "method": "original-attention",
                "label": METHOD_LABELS["original-attention"],
                "rmse": rmse(true, g["original_pred"].to_numpy()),
                "mae": mae(true, g["original_pred"].to_numpy()),
            }
        )
        rows.append(
            {
                "battery_index": int(b),
                "method": "adaptive-attention-v1",
                "label": METHOD_LABELS["adaptive-attention-v1"],
                "rmse": rmse(true, g["adaptive_pred"].to_numpy()),
                "mae": mae(true, g["adaptive_pred"].to_numpy()),
            }
        )
    for method in METHOD_ORDER[2:]:
        path = OUT_DIR / f"{method}-seed42" / f"{method}-seed42_target_test_predictions.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for b, g in df.groupby("battery_index"):
            true = g["true_soh"].to_numpy()
            rows.append(
                {
                    "battery_index": int(b),
                    "method": method,
                    "label": METHOD_LABELS[method],
                    "rmse": rmse(true, g["pred_soh"].to_numpy()),
                    "mae": mae(true, g["pred_soh"].to_numpy()),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "attention_mechanism_seed42_per_cell_long.csv", index=False, encoding="utf-8-sig")
    return out


def load_prediction_curves() -> pd.DataFrame:
    return pd.read_csv(ADAPTIVE_DIR / "adaptive_attention_v1_seed42_target_test_predictions.csv")


def close(vals: np.ndarray) -> np.ndarray:
    return np.concatenate([vals, vals[:1]])


def plot_radar_small_multiples(per_cell: pd.DataFrame) -> Path:
    # Reference image #1: 553x265 px, AR=2.087.
    methods = [m for m in METHOD_ORDER if m in set(per_cell["method"])]
    angles = np.linspace(0, 2 * np.pi, len(methods), endpoint=False)
    cells = sorted(per_cell["battery_index"].unique())
    rmax = float(np.ceil(per_cell["rmse"].max() * 100 * 10) / 10)
    rmax = max(rmax, 1.2)

    fig = plt.figure(figsize=(11.2, 5.36))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 0.88], wspace=0.48, hspace=0.42)
    axes = []
    for i in range(8):
        axes.append(fig.add_subplot(gs[i // 4, i % 4], projection="polar"))

    for ax, cell in zip(axes, cells):
        g = per_cell[per_cell["battery_index"] == cell].set_index("method")
        vals = np.array([float(g.loc[m, "rmse"] * 100) if m in g.index else 0.0 for m in methods])
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_ylim(0, rmax)
        ax.bar(
            angles,
            vals,
            width=(2 * np.pi / len(methods)) * 0.74,
            color=[METHOD_COLORS[m] for m in methods],
            edgecolor="white",
            linewidth=0.6,
            alpha=0.82,
        )
        ax.set_title(f"Cell {cell}", fontsize=8.2, pad=7)
        ax.set_xticks(angles)
        ax.set_xticklabels([METHOD_LABELS[m] for m in methods], fontsize=4.9)
        ax.set_yticks([rmax / 3, 2 * rmax / 3, rmax])
        ax.set_yticklabels([f"{rmax/3:.1f}", f"{2*rmax/3:.1f}", f"{rmax:.1f}"], fontsize=5.4, color="#8A8A8A")
        ax.grid(True, color="#D7D7D7", linewidth=0.55, linestyle="--")
        ax.spines["polar"].set_color("#D0D0D0")
        for label, method in zip(ax.get_xticklabels(), methods):
            if method == "adaptive-attention-v1":
                label.set_color(METHOD_COLORS[method])
                label.set_fontweight("bold")

    leg_ax = fig.add_subplot(gs[:, 4])
    leg_ax.axis("off")
    leg_ax.text(0.0, 0.96, "Models", fontsize=8.4, fontweight="bold", va="top")
    y = 0.88
    for method in methods:
        leg_ax.add_patch(plt.Rectangle((0.0, y - 0.018), 0.08, 0.032, color=METHOD_COLORS[method], transform=leg_ax.transAxes))
        leg_ax.text(0.11, y, METHOD_LABELS[method], transform=leg_ax.transAxes, fontsize=7.2, va="center")
        y -= 0.07
    leg_ax.text(0.0, 0.19, "Metric", fontsize=8.4, fontweight="bold", transform=leg_ax.transAxes)
    leg_ax.text(0.0, 0.13, "RMSE (%)", fontsize=7.2, transform=leg_ax.transAxes)
    leg_ax.text(0.0, 0.035, "Batch10 seed42\nlower is better", fontsize=6.5, color="#666666", transform=leg_ax.transAxes)

    fig.suptitle("Batch10 Mechanism Ablation Per-Cell Comparison", fontsize=10.4, fontweight="bold", y=0.995)
    path = FIG_DIR / "fig01_batch10_mechanism_ablation_radar_small_multiples.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def plot_per_cell_rmse_mae(per_cell: pd.DataFrame) -> Path:
    # Reference image #2: 400x207 px, AR=1.932.
    proposed = per_cell[per_cell["method"] == "adaptive-attention-v1"].copy().sort_values("battery_index")
    proposed["cell_id"] = [f"10-{i}" for i in range(1, len(proposed) + 1)]
    x = np.arange(len(proposed))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.72, 4.0))
    rmse_vals = proposed["rmse"].to_numpy() * 100
    mae_vals = proposed["mae"].to_numpy() * 100
    bars1 = ax.bar(
        x - width / 2,
        rmse_vals,
        width,
        label="RMSE",
        color="#4B168C",
        edgecolor="white",
        linewidth=0.7,
        hatch="////",
    )
    bars2 = ax.bar(
        x + width / 2,
        mae_vals,
        width,
        label="MAE",
        color="#B79BE8",
        edgecolor="white",
        linewidth=0.7,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(proposed["cell_id"])
    ax.set_xlabel("Cell ID", fontweight="bold")
    ax.set_ylabel("Error (% SOH)", fontweight="bold")
    ax.set_title("Adaptive Attention v1 Per-Cell Error", fontweight="bold", pad=6)
    ax.yaxis.grid(True, color="#E6E1F2", linestyle="--", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#7A748A")
    ax.spines["bottom"].set_color("#7A748A")
    ax.legend(frameon=False, loc="upper right")
    for bars, color in [(bars1, "#4B168C"), (bars2, "#8D73C7")]:
        for b in bars:
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.06,
                f"{b.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=6.3,
                color=color,
                fontweight="bold",
            )
    ax.set_ylim(0, max(rmse_vals.max(), mae_vals.max()) * 1.22)
    path = FIG_DIR / "fig02_adaptive_v1_seed42_per_cell_rmse_mae.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def plot_prediction_curves() -> Path:
    # Reference image #3: 554x252 px, AR=2.198.
    pred_df = load_prediction_curves()
    cells = sorted(pred_df["battery_index"].unique())
    fig, axes = plt.subplots(2, 4, figsize=(11.0, 5.0), sharey=True)
    axes = axes.ravel()
    for ax, cell in zip(axes, cells):
        g = pred_df[pred_df["battery_index"] == cell].sort_values("cycle")
        ax.plot(g["cycle"], g["true_soh"], color="#58C6D1", linewidth=1.2, label="True SOH")
        ax.plot(g["cycle"], g["adaptive_pred"], color="#5B2ED6", linewidth=1.05, label="Predicted SOH")
        ax.set_title(f"Cell {cell}", fontsize=8)
        ax.tick_params(labelsize=6.5, length=2.8)
        ax.spines["top"].set_color("#777777")
        ax.spines["right"].set_color("#777777")
        ax.spines["left"].set_color("#777777")
        ax.spines["bottom"].set_color("#777777")
        ax.grid(False)
    for ax in axes[4:]:
        ax.set_xlabel("Cycle", fontsize=7, fontweight="bold")
    for ax in axes[::4]:
        ax.set_ylabel("SOH", fontsize=7, fontweight="bold")
    handles, labels = axes[-2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=True, fontsize=7, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Batch10 Final Model Per-Cell SOH Prediction Curves", fontsize=10, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    path = FIG_DIR / "fig03_batch10_final_model_prediction_curves.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def plot_three_seed_summary() -> Path:
    path = OUT_DIR / "attention_ablation_all_evidence_aggregate.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    agg = pd.read_csv(path)
    keep = [
        "adaptive-attention-v1",
        "original-attention",
        "no-degradation-state",
        "no-curve-tokens",
        "fixed-temperature",
        "shuffled-degradation-state",
        "voltage-only-curve-token",
        "cycle-only-state",
    ]
    df = agg[agg["method"].isin(keep)].copy()
    df["order"] = df["method"].map({m: i for i, m in enumerate(keep)})
    df = df.sort_values("order")
    labels = [METHOD_LABELS.get(m, m) for m in df["method"]]
    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    vals = df["mean_rmse"].to_numpy() * 100
    err = df["std_rmse"].fillna(0).to_numpy() * 100
    colors = [METHOD_COLORS.get(m, "#999999") for m in df["method"]]
    bars = ax.bar(x, vals, yerr=err, capsize=3, color=colors, edgecolor="white", linewidth=0.7)
    bars[0].set_hatch("////")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_ylabel("Mean RMSE (% SOH)", fontweight="bold")
    ax.set_title("Three-Seed Attention Mechanism Ablation Summary", fontweight="bold")
    ax.yaxis.grid(True, color="#E8E8E8", linestyle="--", linewidth=0.65)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    y_top = float(np.max(vals + err))
    label_offset = y_top * 0.04
    for i, b in enumerate(bars):
        ax.text(
            b.get_x() + b.get_width() / 2,
            vals[i] + err[i] + label_offset,
            f"{vals[i]:.3f}",
            ha="center",
            va="bottom",
            fontsize=6.7,
            color="#333333",
        )
    ax.set_ylim(0, y_top + label_offset * 2.7)
    out = FIG_DIR / "fig04_three_seed_attention_mechanism_summary.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


def load_attention_stage_summary() -> pd.DataFrame:
    rows = []
    full_path = ADAPTIVE_DIR / "adaptive_attention_v1_seed42_attention_stage_summary.csv"
    if full_path.exists():
        df = pd.read_csv(full_path)
        for _, r in df.iterrows():
            method = "adaptive-attention-v1" if "Adaptive" in r["method"] else "original-attention"
            rows.append({**r.to_dict(), "method": method})
    for method in METHOD_ORDER[2:]:
        path = OUT_DIR / f"{method}-seed42" / f"{method}-seed42_attention_stage_summary.csv"
        if path.exists():
            df = pd.read_csv(path)
            for _, r in df.iterrows():
                rows.append({"method": method, **r.to_dict()})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "attention_mechanism_seed42_stage_interpretability_long.csv", index=False, encoding="utf-8-sig")
    return out


def plot_attention_stage_interpretability() -> Path:
    df = load_attention_stage_summary()
    keep = ["original-attention", "adaptive-attention-v1", "no-degradation-state", "no-curve-tokens", "fixed-temperature"]
    df = df[df["method"].isin(keep)].copy()
    stages = ["Early", "Middle", "Late"]
    metrics = [
        ("mean_entropy", "Attention entropy"),
        ("mean_max_weight", "Max attention weight"),
        ("mean_temperature", "Temperature"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.2))
    for ax, (metric, title) in zip(axes, metrics):
        for method in keep:
            g = df[df["method"] == method].set_index("stage")
            if g.empty or metric not in g.columns:
                continue
            vals = [float(g.loc[s, metric]) if s in g.index and pd.notna(g.loc[s, metric]) else np.nan for s in stages]
            ax.plot(
                stages,
                vals,
                marker="o",
                linewidth=1.5 if method == "adaptive-attention-v1" else 1.0,
                color=METHOD_COLORS.get(method, "#888888"),
                label=METHOD_LABELS.get(method, method),
            )
        ax.set_title(title, fontweight="bold")
        ax.grid(True, color="#E8E8E8", linestyle="--", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Value", fontweight="bold")
    axes[-1].legend(frameon=True, edgecolor="#DDDDDD", fontsize=6.8, loc="best")
    fig.suptitle("Seed42 Stage-Wise Attention Mechanism Profiles", fontsize=10.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "fig05_seed42_attention_stage_interpretability.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


def write_figure_index(paths: List[Path]) -> None:
    lines = ["# Attention Mechanism Figures", ""]
    for p in paths:
        lines.append(f"- `{p}`")
    (OUT_DIR / "ATTENTION_MECHANISM_FIGURE_INDEX_20260722.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    set_paper_style()
    per_cell = load_seed42_per_cell()
    paths = [
        plot_radar_small_multiples(per_cell),
        plot_per_cell_rmse_mae(per_cell),
        plot_prediction_curves(),
    ]
    if (OUT_DIR / "attention_ablation_all_evidence_aggregate.csv").exists():
        paths.append(plot_three_seed_summary())
    paths.append(plot_attention_stage_interpretability())
    write_figure_index(paths)
    print("Saved figures:")
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
