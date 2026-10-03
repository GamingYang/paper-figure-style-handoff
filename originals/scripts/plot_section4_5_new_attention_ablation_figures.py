"""Generate Section 4.5 ablation figures for the adaptive-attention-v1 manuscript."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(r"C:\Users\ROG\Desktop\paper")
CODEX_DIR = ROOT / "codex"
OUT_DIR = CODEX_DIR / "section4-5-new-attention-ablation-figures-v1"
FIG_DIR = OUT_DIR / "figures"
SRC_DIR = OUT_DIR / "source_data"

ADAPT_DIR = CODEX_DIR / "adaptive-attention-v1"
MECH_DIR = CODEX_DIR / "adaptive-attention-mechanism-ablation-v1"
CTRL_DIR = CODEX_DIR / "adaptive-attention-final-controls-v1"
SUPP_DIR = CODEX_DIR / "supplementary-experiments"
ORIG_DIR = CODEX_DIR / "final-retained-v1"


PALETTE = {
    "full": "#0F4D92",
    "mmd": "#42949E",
    "state": "#7B6DAE",
    "curve": "#D08C3C",
    "focus": "#B64342",
    "structure": "#4D4D4D",
    "baseline": "#8C8C8C",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
            "mathtext.fontset": "stix",
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
        }
    )


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SRC_DIR.mkdir(parents=True, exist_ok=True)


def save_all(fig: plt.Figure, stem: str) -> None:
    base = FIG_DIR / stem
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required result file: {path}")
    return pd.read_csv(path)


def control_summary() -> pd.DataFrame:
    return read_required_csv(CTRL_DIR / "adaptive_attention_final_controls_summary.csv")


def control_aggregate() -> pd.DataFrame:
    return read_required_csv(CTRL_DIR / "adaptive_attention_final_controls_aggregate.csv")


def mechanism_aggregate() -> pd.DataFrame:
    return read_required_csv(MECH_DIR / "attention_ablation_all_evidence_aggregate.csv")


def mechanism_per_seed() -> pd.DataFrame:
    return read_required_csv(MECH_DIR / "attention_ablation_all_evidence_per_seed.csv")


def pct(v: float) -> float:
    return float(v) * 100.0


def build_effect_table() -> pd.DataFrame:
    ctrl = control_aggregate()
    mech = mechanism_aggregate()

    rows: List[Dict] = []

    def add(label: str, category: str, mean_rmse: float, std_rmse: float, mean_mae: float, std_mae: float, source: str):
        rows.append(
            {
                "label": label,
                "category": category,
                "mean_rmse_pct": pct(mean_rmse),
                "std_rmse_pct": pct(std_rmse) if pd.notna(std_rmse) else 0.0,
                "mean_mae_pct": pct(mean_mae),
                "std_mae_pct": pct(std_mae) if pd.notna(std_mae) else 0.0,
                "source": source,
            }
        )

    ctrl_map = {r["label"]: r for _, r in ctrl.iterrows()}
    mech_map = {r["method"]: r for _, r in mech.iterrows()}

    for label, category in [
        ("Proposed", "full"),
        ("static MMD", "mmd"),
        ("w/o adaptive MMD", "mmd"),
        ("w/o physical base", "structure"),
    ]:
        r = ctrl_map[label]
        add(label, category, r["mean_rmse"], r["std_rmse"], r["mean_mae"], r["std_mae"], "new adaptive-attention control")

    mechanism_rows = [
        ("original-attention", "Original attention", "baseline"),
        ("no-degradation-state", "w/o degradation state", "state"),
        ("cycle-only-state", "cycle-only state", "state"),
        ("shuffled-degradation-state", "shuffled state", "state"),
        ("no-curve-tokens", "w/o curve tokens", "curve"),
        ("voltage-only-curve-token", "voltage-only token", "curve"),
        ("fixed-temperature", "fixed temperature", "focus"),
    ]
    for key, label, category in mechanism_rows:
        r = mech_map[key]
        add(label, category, r["mean_rmse"], r["std_rmse"], r["mean_mae"], r["std_mae"], "attention mechanism ablation")

    df = pd.DataFrame(rows)
    proposed_rmse = float(df.loc[df["label"] == "Proposed", "mean_rmse_pct"].iloc[0])
    proposed_mae = float(df.loc[df["label"] == "Proposed", "mean_mae_pct"].iloc[0])
    order = [
        "Proposed",
        "static MMD",
        "w/o adaptive MMD",
        "Original attention",
        "w/o degradation state",
        "cycle-only state",
        "shuffled state",
        "w/o curve tokens",
        "voltage-only token",
        "fixed temperature",
        "w/o physical base",
    ]
    df["delta_rmse_pct"] = df["mean_rmse_pct"] - proposed_rmse
    df["delta_mae_pct"] = df["mean_mae_pct"] - proposed_mae
    df["order"] = df["label"].map({name: i for i, name in enumerate(order)})
    df = df.sort_values("order").reset_index(drop=True)
    df.to_csv(SRC_DIR / "fig4_5a_effect_map_source.csv", index=False, encoding="utf-8-sig")
    return df


def category_color(category: str) -> str:
    return PALETTE.get(category, PALETTE["baseline"])


def plot_effect_map(df: pd.DataFrame) -> None:
    labels = df["label"].tolist()
    y = np.arange(len(labels))[::-1]
    colors = [category_color(c) for c in df["category"]]

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(7.2, 4.55),
        sharey=True,
        gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.08},
    )

    for ax, metric, err, xlabel in [
        (ax1, "delta_rmse_pct", "std_rmse_pct", r"$\Delta$RMSE vs proposed (% SOH)"),
        (ax2, "delta_mae_pct", "std_mae_pct", r"$\Delta$MAE vs proposed (% SOH)"),
    ]:
        ax.axvline(0, color="#272727", lw=0.8, ls="-")
        for yi, row, color in zip(y, df.to_dict(orient="records"), colors):
            marker = "D" if row["label"] == "Proposed" else "o"
            ax.errorbar(
                row[metric],
                yi,
                xerr=row[err],
                fmt=marker,
                color=color,
                ecolor=color,
                elinewidth=1.0,
                capsize=2.0,
                markersize=4.0 if marker == "o" else 4.6,
                alpha=0.95,
                zorder=3,
            )
        ax.grid(axis="x", color="#E7E7E7", lw=0.6, ls="--")
        ax.set_axisbelow(True)
        ax.set_xlabel(xlabel)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        xmin = float((df[metric] - df[err]).min())
        xmax = float((df[metric] + df[err]).max())
        margin = max((xmax - xmin) * 0.08, 0.04)
        ax.set_xlim(xmin - margin, xmax + margin)

    ax1.set_yticks(y)
    ax1.set_yticklabels(labels)
    ax2.tick_params(axis="y", labelleft=False)

    ax1.set_title("Prediction error")
    ax2.set_title("Absolute deviation")

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["mmd"], markersize=4, label="MMD control"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["state"], markersize=4, label="State control"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["curve"], markersize=4, label="Curve control"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["focus"], markersize=4, label="Focus control"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["structure"], markersize=4, label="Structure control"),
    ]
    ax2.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.99, 0.98), ncol=1, handletextpad=0.4)
    save_all(fig, "fig4_5a_ablation_static_mmd_effect_map")


def build_stage_table() -> pd.DataFrame:
    ctrl = control_aggregate()
    mech = mechanism_aggregate()
    rows = []

    def add(label: str, category: str, early: float, mid: float, late: float):
        rows.extend(
            [
                {"label": label, "category": category, "stage": "Early", "rmse_pct": pct(early)},
                {"label": label, "category": category, "stage": "Middle", "rmse_pct": pct(mid)},
                {"label": label, "category": category, "stage": "Late", "rmse_pct": pct(late)},
            ]
        )

    ctrl_map = {r["label"]: r for _, r in ctrl.iterrows()}
    mech_map = {r["method"]: r for _, r in mech.iterrows()}
    for label, category in [
        ("Proposed", "full"),
        ("static MMD", "mmd"),
        ("w/o adaptive MMD", "mmd"),
        ("w/o physical base", "structure"),
    ]:
        r = ctrl_map[label]
        add(label, category, r["mean_early_rmse"], r["mean_mid_rmse"], r["mean_late_rmse"])

    for key, label, category in [
        ("no-degradation-state", "w/o degradation state", "state"),
        ("cycle-only-state", "cycle-only state", "state"),
        ("shuffled-degradation-state", "shuffled state", "state"),
        ("no-curve-tokens", "w/o curve tokens", "curve"),
        ("voltage-only-curve-token", "voltage-only token", "curve"),
        ("fixed-temperature", "fixed temperature", "focus"),
    ]:
        r = mech_map[key]
        add(label, category, r["mean_early_rmse"], r["mean_mid_rmse"], r["mean_late_rmse"])

    df = pd.DataFrame(rows)
    df.to_csv(SRC_DIR / "fig4_5b_stage_radar_source.csv", index=False, encoding="utf-8-sig")
    return df


def close(vals: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(vals), dtype=float)
    return np.append(arr, arr[0])


def plot_stage_radar(df: pd.DataFrame) -> None:
    stages = ["Early", "Middle", "Late"]
    stage_angles = np.linspace(0, 2 * np.pi, len(stages), endpoint=False)
    angles = np.append(stage_angles, stage_angles[0])
    color_map = {
        "Proposed": PALETTE["full"],
        "static MMD": PALETTE["mmd"],
        "w/o adaptive MMD": "#7CBCC3",
        "w/o degradation state": PALETTE["state"],
        "cycle-only state": "#9E91C8",
        "shuffled state": "#B9A7D7",
        "w/o curve tokens": PALETTE["curve"],
        "voltage-only token": "#E5AD66",
        "fixed temperature": PALETTE["focus"],
        "w/o physical base": PALETTE["structure"],
    }
    groups = [
        ("a) MMD adaptation", ["Proposed", "static MMD", "w/o adaptive MMD"]),
        ("b) State conditioning", ["Proposed", "w/o degradation state", "cycle-only state", "shuffled state"]),
        ("c) Curve and focus", ["Proposed", "w/o curve tokens", "voltage-only token", "fixed temperature"]),
        ("d) Physical prior", ["Proposed", "w/o physical base"]),
    ]
    pair_groups = [
        ("a) static MMD", "static MMD", "mmd"),
        ("b) w/o adaptive MMD", "w/o adaptive MMD", "mmd"),
        ("c) w/o degradation state", "w/o degradation state", "state"),
        ("d) w/o curve tokens", "w/o curve tokens", "curve"),
        ("e) fixed temperature", "fixed temperature", "focus"),
        ("f) w/o physical base", "w/o physical base", "structure"),
    ]

    all_labels = [label for _, labels in groups for label in labels] + [label for _, label, _ in pair_groups]
    pivot = df.pivot(index="label", columns="stage", values="rmse_pct").loc[sorted(set(all_labels)), stages]
    rmax = max(3.6, math.ceil(float(pivot.values.max()) * 10.0) / 10.0)
    rings = [0.6, 1.2, 1.8, 2.4, 3.0, rmax]

    def draw_panel(
        ax: plt.Axes,
        title: str,
        labels: List[str],
        export_single: bool = False,
        local_scale: bool = False,
        legend_columns: int = 2,
    ) -> None:
        panel_values = pivot.loc[labels].values.astype(float)
        panel_rmax = rmax
        if local_scale:
            panel_rmax = max(2.5, math.ceil(float(panel_values.max() + 0.18) * 10.0) / 10.0)
        panel_rings = np.linspace(panel_rmax / 5.0, panel_rmax, 5)

        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_ylim(0, panel_rmax)
        ax.grid(False)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines["polar"].set_visible(False)

        for ring in panel_rings:
            ax.plot(angles, close([ring] * len(stages)), color="#D8D8D8", lw=0.55, ls="--", zorder=1)
            ax.text(np.deg2rad(17), ring, f"{ring:.1f}", color="#727272", fontsize=5.1, va="center")
        for angle in stage_angles:
            ax.plot([angle, angle], [0, panel_rmax], color="#C5C5C5", lw=0.55, zorder=1)

        label_offsets = {
            "Early": [(-8, 3), (8, 3), (-16, -4), (16, -4)],
            "Middle": [(-7, -2), (-13, 5), (-11, -9), (-19, -3)],
            "Late": [(7, -2), (13, 5), (11, -9), (19, -3)],
        }

        for j, label in enumerate(labels):
            vals = pivot.loc[label].values.astype(float)
            lw = 2.1 if label == "Proposed" else 1.35
            alpha = 0.17 if label == "Proposed" else 0.10
            z = 5 if label == "Proposed" else 4
            ax.plot(angles, close(vals), color=color_map[label], lw=lw, label=label, zorder=z)
            ax.fill(angles, close(vals), color=color_map[label], alpha=alpha, zorder=2)
            ax.scatter(stage_angles, vals, s=12 if label == "Proposed" else 9, color=color_map[label], zorder=z + 1)
            for angle, stage, value in zip(stage_angles, stages, vals):
                dx, dy = label_offsets[stage][j % len(label_offsets[stage])]
                ax.annotate(
                    f"{value:.2f}",
                    xy=(angle, value),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    fontsize=4.9,
                    color=color_map[label],
                    bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.78),
                    zorder=8,
                )

        for angle, stage in zip(stage_angles, stages):
            ax.text(angle, panel_rmax * 1.10, stage, ha="center", va="center", fontsize=6.8)
        ax.text(0.5, -0.11, title, transform=ax.transAxes, ha="center", va="top", fontsize=7.1, fontweight="bold")
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.20 if export_single else -0.18),
            ncol=legend_columns,
            fontsize=5.4,
            handlelength=1.4,
            columnspacing=0.8,
            handletextpad=0.35,
        )

    fig, axs = plt.subplots(2, 2, figsize=(6.9, 7.0), subplot_kw=dict(projection="polar"))
    for ax, (title, labels) in zip(axs.ravel(), groups):
        draw_panel(ax, title, labels)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.97, bottom=0.08, wspace=0.24, hspace=0.52)
    save_all(fig, "fig4_5b_stagewise_rmse_radar_grouped")

    pair_fig, pair_axs = plt.subplots(2, 3, figsize=(7.4, 5.25), subplot_kw=dict(projection="polar"))
    for ax, (title, control_label, _) in zip(pair_axs.ravel(), pair_groups):
        draw_panel(ax, title, ["Proposed", control_label], local_scale=True, legend_columns=1)
    pair_fig.subplots_adjust(left=0.04, right=0.985, top=0.965, bottom=0.11, wspace=0.26, hspace=0.56)
    save_all(pair_fig, "fig4_5b_stagewise_rmse_radar")

    single_dir = FIG_DIR / "stagewise_radar_panels"
    single_dir.mkdir(parents=True, exist_ok=True)
    old_fig_dir = FIG_DIR
    try:
        globals()["FIG_DIR"] = single_dir
        for idx, (title, labels) in enumerate(groups, start=1):
            single_fig = plt.figure(figsize=(3.55, 3.55))
            single_ax = single_fig.add_subplot(111, projection="polar")
            draw_panel(single_ax, title, labels, export_single=True)
            single_fig.subplots_adjust(left=0.08, right=0.94, top=0.94, bottom=0.22)
            save_all(single_fig, f"fig4_5b_stagewise_rmse_radar_panel_{idx}")
    finally:
        globals()["FIG_DIR"] = old_fig_dir

    pair_dir = FIG_DIR / "stagewise_radar_pairwise_panels"
    pair_dir.mkdir(parents=True, exist_ok=True)
    old_fig_dir = FIG_DIR
    try:
        globals()["FIG_DIR"] = pair_dir
        for idx, (title, control_label, _) in enumerate(pair_groups, start=1):
            single_fig = plt.figure(figsize=(3.35, 3.35))
            single_ax = single_fig.add_subplot(111, projection="polar")
            draw_panel(single_ax, title, ["Proposed", control_label], export_single=True, local_scale=True, legend_columns=1)
            single_fig.subplots_adjust(left=0.08, right=0.94, top=0.94, bottom=0.25)
            save_all(single_fig, f"fig4_5b_stagewise_rmse_radar_pair_{idx}")
    finally:
        globals()["FIG_DIR"] = old_fig_dir


def result_xlsx(label: str, seed: int) -> Path:
    mapping = {
        "Proposed": ADAPT_DIR / f"adaptive-attention-v1-seed{seed}" / f"adaptive-attention-v1-seed{seed}_results.xlsx",
        "Original attention": (
            ORIG_DIR / "final-retained-v1_results.xlsx"
            if seed == 42
            else SUPP_DIR / f"final-retained-seed{seed}" / f"final-retained-seed{seed}_results.xlsx"
        ),
        "w/o degradation state": MECH_DIR / f"no-degradation-state-seed{seed}" / f"no-degradation-state-seed{seed}_results.xlsx",
        "w/o curve tokens": MECH_DIR / f"no-curve-tokens-seed{seed}" / f"no-curve-tokens-seed{seed}_results.xlsx",
        "fixed temperature": MECH_DIR / f"fixed-temperature-seed{seed}" / f"fixed-temperature-seed{seed}_results.xlsx",
        "w/o adaptive MMD": CTRL_DIR / f"adaptive-attention-no-mmd-seed{seed}" / f"adaptive-attention-no-mmd-seed{seed}_results.xlsx",
        "w/o physical base": CTRL_DIR / f"adaptive-attention-no-physical-base-seed{seed}" / f"adaptive-attention-no-physical-base-seed{seed}_results.xlsx",
    }
    if label not in mapping:
        raise KeyError(label)
    return mapping[label]


def read_seed_summary(path: Path) -> pd.Series:
    xl = pd.ExcelFile(path)
    if "summary" in xl.sheet_names:
        return pd.read_excel(path, sheet_name="summary").iloc[0]
    if "metrics" in xl.sheet_names:
        metrics = pd.read_excel(path, sheet_name="metrics")
        test = metrics[metrics["split"].astype(str).str.lower().eq("target_test")]
        if test.empty:
            raise ValueError(f"No target_test row in metrics sheet: {path}")
        row = test.iloc[0]
        return pd.Series(
            {
                "target_test_rmse": row["pooled_rmse"],
                "target_test_macro_rmse": row["macro_rmse"],
                "target_test_mae": row["mae"],
            }
        )
    raise ValueError(f"Neither summary nor metrics sheet exists: {path}")


def build_violin_tables() -> Tuple[pd.DataFrame, pd.DataFrame]:
    labels = [
        "Proposed",
        "Original attention",
        "w/o degradation state",
        "w/o curve tokens",
        "fixed temperature",
        "w/o adaptive MMD",
        "w/o physical base",
    ]
    seeds = [42, 7, 123]
    cell_rows = []
    seed_rows = []
    for label in labels:
        for seed in seeds:
            path = result_xlsx(label, seed)
            if not path.exists():
                raise FileNotFoundError(f"Missing result workbook for {label} seed {seed}: {path}")
            cells = pd.read_excel(path, sheet_name="target_test_cells")
            summary = read_seed_summary(path)
            for _, row in cells.iterrows():
                cell_rows.append(
                    {
                        "label": label,
                        "seed": seed,
                        "battery_index": int(row["battery_index"]),
                        "cell_rmse_pct": pct(row["rmse"]),
                        "cell_mae_pct": pct(row["mae"]),
                    }
                )
            seed_rows.append(
                {
                    "label": label,
                    "seed": seed,
                    "pooled_rmse_pct": pct(summary["target_test_rmse"]),
                    "macro_rmse_pct": pct(summary["target_test_macro_rmse"]),
                    "mae_pct": pct(summary["target_test_mae"]),
                }
            )
    cell_df = pd.DataFrame(cell_rows)
    seed_df = pd.DataFrame(seed_rows)
    cell_df.to_csv(SRC_DIR / "fig4_5c_multiseed_cell_violin_source.csv", index=False, encoding="utf-8-sig")
    seed_df.to_csv(SRC_DIR / "fig4_5c_multiseed_seed_points_source.csv", index=False, encoding="utf-8-sig")
    return cell_df, seed_df


def plot_violin(cell_df: pd.DataFrame, seed_df: pd.DataFrame) -> None:
    labels = [
        "Proposed",
        "Original\nattention",
        "w/o state",
        "w/o curve",
        "fixed\ntemperature",
        "w/o MMD",
        "w/o physical\nbase",
    ]
    long_labels = [
        "Proposed",
        "Original attention",
        "w/o degradation state",
        "w/o curve tokens",
        "fixed temperature",
        "w/o adaptive MMD",
        "w/o physical base",
    ]
    colors = [
        PALETTE["full"],
        PALETTE["baseline"],
        PALETTE["state"],
        PALETTE["curve"],
        PALETTE["focus"],
        PALETTE["mmd"],
        PALETTE["structure"],
    ]
    data = [cell_df.loc[cell_df["label"] == lab, "cell_rmse_pct"].values for lab in long_labels]
    fig, ax = plt.subplots(figsize=(7.2, 4.15))
    positions = np.arange(1, len(labels) + 1)
    parts = ax.violinplot(data, positions=positions, widths=0.72, showmeans=False, showmedians=False, showextrema=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.22)
        body.set_linewidth(1.0)

    for pos, vals, color in zip(positions, data, colors):
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax.plot([pos - 0.22, pos + 0.22], [med, med], color=color, lw=1.6, solid_capstyle="round", zorder=4)
        ax.plot([pos, pos], [q1, q3], color=color, lw=3.0, alpha=0.9, solid_capstyle="round", zorder=4)
        offsets = np.linspace(-0.16, 0.16, len(vals))
        ax.scatter(pos + offsets, vals, s=8, color=color, alpha=0.34, edgecolor="none", zorder=3)

    seed_offsets = {42: -0.16, 7: 0.0, 123: 0.16}
    for pos, lab in zip(positions, long_labels):
        sub = seed_df[seed_df["label"] == lab]
        for _, row in sub.iterrows():
            ax.scatter(
                pos + seed_offsets[int(row["seed"])],
                row["pooled_rmse_pct"],
                s=18,
                marker="D",
                color="#272727",
                edgecolor="white",
                linewidth=0.35,
                zorder=5,
            )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Cell-wise RMSE across seeds (% SOH)")
    ax.grid(axis="y", color="#E7E7E7", lw=0.6, ls="--")
    ax.set_axisbelow(True)
    ymax = max(max(vals) for vals in data) * 1.13
    ax.set_ylim(0, ymax)
    legend_handles = [
        Line2D([0], [0], marker="D", color="none", markerfacecolor="#272727", markeredgecolor="white", markersize=4.5, label="Seed-level pooled RMSE"),
        Line2D([0], [0], color="#272727", lw=1.6, label="Median cell RMSE"),
    ]
    ax.legend(handles=legend_handles, loc="upper left")
    save_all(fig, "fig4_5c_multiseed_stability_violin")


def write_notes(effect_df: pd.DataFrame, stage_df: pd.DataFrame, seed_df: pd.DataFrame) -> None:
    proposed = effect_df[effect_df["label"] == "Proposed"].iloc[0]
    lines = [
        "# Section 4.5 New Attention Ablation Figure Notes",
        "",
        "Final model: physical cubic degradation base + residual temporal network + degradation-state-guided curve attention v1 + life-cycle adaptive MMD.",
        "",
        "## Figure contract",
        "",
        "- Fig. 4.5a: effect-size map for MMD controls, attention-mechanism controls, and structural controls.",
        "- Fig. 4.5b: radar view of stage-wise RMSE across early, middle, and late degradation stages.",
        "- Fig. 4.5c: violin distribution of Batch10 cell-wise RMSE across seeds, with seed-level pooled RMSE overlaid.",
        "",
        "## Key values",
        "",
        f"- Proposed three-seed mean RMSE: {proposed['mean_rmse_pct']:.3f}% SOH.",
        f"- Proposed three-seed mean MAE: {proposed['mean_mae_pct']:.3f}% SOH.",
        "",
        "## Writing guardrail",
        "",
        "The figures support a mechanism-oriented ablation narrative. They should not be used to claim that every expanded curve token monotonically improves every metric. Voltage-only curve tokens are retained in the source data as a diagnostic control.",
    ]
    (OUT_DIR / "SECTION_4_5_NEW_ATTENTION_FIGURE_NOTES_20260729.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_style()
    ensure_dirs()
    effect_df = build_effect_table()
    plot_effect_map(effect_df)
    stage_df = build_stage_table()
    plot_stage_radar(stage_df)
    cell_df, seed_df = build_violin_tables()
    plot_violin(cell_df, seed_df)
    write_notes(effect_df, stage_df, seed_df)
    import plot_stagewise_pentagon_radar_reference_style as pentagon_radar

    pentagon_radar.main()
    print(FIG_DIR)


if __name__ == "__main__":
    main()
