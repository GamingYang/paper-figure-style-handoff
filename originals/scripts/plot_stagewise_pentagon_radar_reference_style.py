"""Reference-style pentagon radar plots for Section 4.5 stage-aware error profiles."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"C:\Users\ROG\Desktop\paper")
CODEX_DIR = ROOT / "codex"
OUT_DIR = CODEX_DIR / "section4-5-new-attention-ablation-figures-v1"
FIG_DIR = OUT_DIR / "figures"
SRC_DIR = OUT_DIR / "source_data"
CTRL_DIR = CODEX_DIR / "adaptive-attention-final-controls-v1"
MECH_DIR = CODEX_DIR / "adaptive-attention-mechanism-ablation-v1"


AXES = ["Early", "Middle", "Late", "Overall", "Macro"]
PANEL_SPECS = [
    (
        "(a) MMD control",
        ["Proposed", "static MMD", "w/o adaptive MMD"],
    ),
    (
        "(b) Attention mechanism",
        ["Proposed", "w/o degradation state", "w/o curve tokens", "fixed temperature"],
    ),
    (
        "(c) Structural prior",
        ["Proposed", "Original attention", "w/o physical base"],
    ),
]
COLORS = {
    "Proposed": "#12A7C9",
    "static MMD": "#84D8B6",
    "w/o adaptive MMD": "#B95CE5",
    "w/o degradation state": "#39B56B",
    "w/o curve tokens": "#F26A5B",
    "fixed temperature": "#7C6FE3",
    "Original attention": "#F0B74F",
    "w/o physical base": "#E84C4F",
}
HATCHES = {
    "Proposed": "////",
    "static MMD": "\\\\\\\\",
    "w/o adaptive MMD": "////",
    "w/o degradation state": "\\\\\\\\",
    "w/o curve tokens": "////",
    "fixed temperature": "\\\\\\\\",
    "Original attention": "////",
    "w/o physical base": "\\\\\\\\",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
        }
    )


def pct(value: float) -> float:
    return float(value) * 100.0


def close(vals: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(vals), dtype=float)
    return np.append(arr, arr[0])


def build_profile_table() -> pd.DataFrame:
    ctrl = pd.read_csv(CTRL_DIR / "adaptive_attention_final_controls_aggregate.csv")
    mech = pd.read_csv(MECH_DIR / "attention_ablation_all_evidence_aggregate.csv")

    rows = []

    def add(label: str, source: str, early: float, middle: float, late: float, overall: float, macro: float) -> None:
        values = {
            "Early": pct(early),
            "Middle": pct(middle),
            "Late": pct(late),
            "Overall": pct(overall),
            "Macro": pct(macro),
        }
        for axis, value in values.items():
            rows.append({"label": label, "source": source, "axis": axis, "rmse_pct": value})

    ctrl_map = {row["label"]: row for _, row in ctrl.iterrows()}
    for label in ["Proposed", "static MMD", "w/o adaptive MMD", "w/o physical base"]:
        row = ctrl_map[label]
        add(
            label,
            "final-control",
            row["mean_early_rmse"],
            row["mean_mid_rmse"],
            row["mean_late_rmse"],
            row["mean_rmse"],
            row["mean_macro_rmse"],
        )

    mech_map = {row["method"]: row for _, row in mech.iterrows()}
    for method, label in [
        ("original-attention", "Original attention"),
        ("no-degradation-state", "w/o degradation state"),
        ("no-curve-tokens", "w/o curve tokens"),
        ("fixed-temperature", "fixed temperature"),
    ]:
        row = mech_map[method]
        add(
            label,
            "mechanism-control",
            row["mean_early_rmse"],
            row["mean_mid_rmse"],
            row["mean_late_rmse"],
            row["mean_rmse"],
            row["mean_macro_rmse"],
        )

    df = pd.DataFrame(rows)
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SRC_DIR / "fig4_5b_stage_profile_pentagon_source.csv", index=False, encoding="utf-8-sig")
    return df


def values_for(df: pd.DataFrame, label: str) -> np.ndarray:
    pivot = df.pivot(index="label", columns="axis", values="rmse_pct")
    return pivot.loc[label, AXES].values.astype(float)


def normalized(values: np.ndarray, vmax: float) -> np.ndarray:
    rmin, rmax = 0.18, 0.95
    return rmin + (rmax - rmin) * values / vmax


def draw_pentagon_radar(
    ax: plt.Axes,
    df: pd.DataFrame,
    title: str,
    labels: list[str],
    legend_anchor: tuple[float, float] = (-0.22, 1.18),
    legend_fontsize: float = 7.0,
    title_y: float = -0.14,
) -> None:
    angles = np.linspace(0, 2 * np.pi, len(AXES), endpoint=False)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.08)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)
    ax.grid(False)

    panel_values = np.vstack([values_for(df, label) for label in labels])
    vmax = float(np.ceil(panel_values.max() * 10.0) / 10.0)
    vmax = max(vmax, 2.4)

    ring_values = np.linspace(0.0, vmax, 6)[1:]
    for raw in ring_values:
        r = normalized(np.array([raw]), vmax)[0]
        ax.plot(close(angles), close(np.full(len(AXES), r)), color="#B9C0BD", lw=0.45, alpha=0.75)
        ax.text(
            np.deg2rad(6),
            r,
            f"{raw:.1f}",
            ha="left",
            va="center",
            fontsize=6.1,
            color="black",
        )

    for angle in angles:
        ax.plot([angle, angle], [0.18, 0.95], color="#AAB1AE", lw=0.42, alpha=0.72)

    axis_label_r = 1.03
    for angle, axis in zip(angles, AXES):
        ax.text(angle, axis_label_r, axis, ha="center", va="center", fontsize=8.3, color="black")

    for label in labels:
        raw_values = values_for(df, label)
        r_values = normalized(raw_values, vmax)
        color = COLORS[label]
        lw = 1.8 if label == "Proposed" else 1.35
        z = 5 if label == "Proposed" else 4
        ax.fill(
            close(angles),
            close(r_values),
            facecolor=color,
            edgecolor=color,
            linewidth=0.55,
            alpha=0.18,
            hatch=HATCHES[label],
            zorder=2,
        )
        ax.plot(close(angles), close(r_values), color=color, lw=lw, marker="o", ms=2.5, zorder=z)

    # Black numeric labels are placed after all series so values remain readable.
    offsets = {
        "Early": [(-17, 9), (18, 9), (-27, -7), (28, -7)],
        "Middle": [(13, 1), (28, 6), (17, -13), (34, -8)],
        "Late": [(-13, 1), (-28, 6), (-17, -13), (-34, -8)],
        "Overall": [(-13, -11), (14, -11), (-29, 6), (30, 6)],
        "Macro": [(13, -11), (-14, -11), (29, 6), (-30, 6)],
    }
    for j, label in enumerate(labels):
        raw_values = values_for(df, label)
        r_values = normalized(raw_values, vmax)
        for angle, axis, raw, r in zip(angles, AXES, raw_values, r_values):
            dx, dy = offsets[axis][j % len(offsets[axis])]
            ax.annotate(
                f"{raw:.2f}",
                xy=(angle, r),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=5.7,
                color="black",
                bbox=dict(boxstyle="round,pad=0.09", facecolor="white", edgecolor="none", alpha=0.90),
                zorder=10,
            )

    ax.text(0.5, title_y, title, transform=ax.transAxes, ha="center", va="top", fontsize=8.2, fontweight="bold")
    handles = [
        mpl.patches.Patch(
            facecolor=COLORS[label],
            edgecolor=COLORS[label],
            hatch=HATCHES[label],
            alpha=0.23,
            label=label,
        )
        for label in labels
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=legend_anchor, fontsize=legend_fontsize, handlelength=1.6)


def save_all(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    base = FIG_DIR / stem
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_style()
    df = build_profile_table()
    fig, axes = plt.subplots(3, 1, figsize=(5.2, 10.1), subplot_kw=dict(projection="polar"))
    for ax, (title, labels) in zip(axes, PANEL_SPECS):
        draw_pentagon_radar(ax, df, title, labels)
    fig.subplots_adjust(left=0.18, right=0.92, top=0.985, bottom=0.035, hspace=0.42)
    save_all(fig, "fig4_5b_stagewise_rmse_radar")

    backup_fig, backup_axes = plt.subplots(3, 1, figsize=(5.2, 10.1), subplot_kw=dict(projection="polar"))
    for ax, (title, labels) in zip(backup_axes, PANEL_SPECS):
        draw_pentagon_radar(ax, df, title, labels)
    backup_fig.subplots_adjust(left=0.18, right=0.92, top=0.985, bottom=0.035, hspace=0.42)
    save_all(backup_fig, "fig4_5b_stagewise_rmse_radar_pentagon_reference_style")

    horizontal_fig, horizontal_axes = plt.subplots(1, 3, figsize=(12.6, 3.65), subplot_kw=dict(projection="polar"))
    for ax, (title, labels) in zip(horizontal_axes, PANEL_SPECS):
        draw_pentagon_radar(
            ax,
            df,
            title,
            labels,
            legend_anchor=(-0.08, 1.19),
            legend_fontsize=6.5,
            title_y=-0.18,
        )
    horizontal_fig.subplots_adjust(left=0.035, right=0.985, top=0.86, bottom=0.18, wspace=0.30)
    save_all(horizontal_fig, "fig4_5b_stagewise_rmse_radar_horizontal")
    print(FIG_DIR)


if __name__ == "__main__":
    main()
