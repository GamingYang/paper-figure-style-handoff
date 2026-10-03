"""Portable figure templates. Read bundled CSV/NPZ only; never train a model."""

from __future__ import annotations

import argparse
import importlib.util
import string
from pathlib import Path

from figure_style import (PACK, STYLE, configure, save_figure, black_axes,
                          panel_xlabel, bar_numbers_above_errors, place_near_point)
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

PALETTE = STYLE["palette"]
RADAR_AXES = ["Early", "Middle", "Late", "Overall", "Macro"]
RADAR_GROUPS = [
    ("MMD control", ["Proposed", "static MMD", "w/o adaptive MMD"]),
    ("Attention mechanism", ["Proposed", "w/o degradation state", "w/o curve tokens", "fixed temperature"]),
    ("Structural prior", ["Proposed", "Original attention", "w/o physical base"]),
]


def read_csv(data, name):
    return pd.read_csv(data / name)


def radar(data, out, save, layout="horizontal"):
    df = read_csv(data, "fig4_5b_stage_profile_pentagon_source.csv")
    if df.duplicated(["label", "axis"]).any():
        raise ValueError("Radar input has duplicate label/axis rows")
    table = df.pivot(index="label", columns="axis", values="rmse_pct")
    chosen = sorted({m for _, methods in RADAR_GROUPS for m in methods})
    values = table.loc[chosen, RADAR_AXES].to_numpy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Radar RMSE must be finite and nonnegative")
    vmax = np.ceil(values.max() * 2) / 2
    horizontal = layout == "horizontal"
    fig, axes = plt.subplots(1 if horizontal else 3, 3 if horizontal else 1,
                             figsize=(12.6, 4.5) if horizontal else (5.2, 13),
                             subplot_kw={"projection": "polar"})
    fig.subplots_adjust(left=.08 if not horizontal else .035, right=.94 if not horizontal else .98,
                        top=.87 if horizontal else .94, bottom=.13 if horizontal else .04,
                        wspace=.33, hspace=.53)
    angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    close = lambda v: np.r_[v, v[0]]
    pending = []
    for index, (ax, (title, methods)) in enumerate(zip(np.ravel(axes), RADAR_GROUPS)):
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_ylim(0, vmax * 1.16)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.spines["polar"].set_visible(False)
        static_text = []
        for level in np.linspace(0, vmax, 6)[1:]:
            ax.plot(close(angles), np.full(6, level), color="#C9CFCD", lw=.55, zorder=0)
            static_text.append(ax.text(np.deg2rad(15), level, f"{level:.1f}", color="black", fontsize=6.2))
        for angle, axis in zip(angles, RADAR_AXES):
            ax.plot([angle, angle], [0, vmax], color="#C9CFCD", lw=.5, zorder=0)
            static_text.append(ax.text(angle, vmax * 1.13, axis, ha="center", va="center", fontsize=8))
        for j, method in enumerate(methods):
            vals = table.loc[method, RADAR_AXES].to_numpy(float)
            ax.fill(close(angles), close(vals), color=PALETTE[method], alpha=STYLE["radar_fill_alpha"], zorder=1)
            ax.plot(close(angles), close(vals), color=PALETTE[method],
                    lw=1.7 if j == 0 else 1.3, marker=["o", "s", "^", "D"][j], ms=3,
                    label=method, zorder=3 + (j == 0))
        ax.legend(loc="upper left", bbox_to_anchor=(-.02, 1.26), fontsize=7, handlelength=2)
        ax.text(.5, -.11, f"{string.ascii_lowercase[index]}) {title}", transform=ax.transAxes,
                ha="center", va="top", fontsize=8.5)
        pending.append((ax, methods, static_text))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, methods, static_text in pending:
        occupied = [t.get_window_extent(renderer).expanded(1.1, 1.1) for t in static_text]
        occupied.append(ax.get_legend().get_window_extent(renderer))
        for angle, axis in zip(angles, RADAR_AXES):
            # Place from low to high, then find a nearby free position in display units.
            for method in sorted(methods, key=lambda m: table.loc[m, axis]):
                value = float(table.loc[method, axis])
                place_near_point(ax, (angle, value), f"{value:.2f}", occupied,
                                 direction=(np.sin(angle), np.cos(angle)))
    save(fig, out, f"radar_{layout}")


def mechanism(data, out, save):
    cache = np.load(data / "attention_mechanism_cache.npz", allow_pickle=False)
    fig = plt.figure(figsize=(9.0, 5.7))
    gs = fig.add_gridspec(2, 3, left=.075, right=.875, top=.96, bottom=.105,
                         wspace=.55, hspace=.42)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    stages = list(cache["stages"])
    cmap = LinearSegmentedColormap.from_list("attention_strength", STYLE["attention_colors"])
    norm = Normalize(0, float(cache["attention_vmax"]))
    x = np.linspace(0, 1, cache["representative_curve"].shape[1])
    for i, stage in enumerate(stages):
        voltage = cache["representative_curve"][i, :, 0]
        weights = cache["representative_attention"][i]
        y = .52 * (voltage - voltage.min()) / max(float(np.ptp(voltage)), 1e-12) + [1.35, .70, .05][i]
        points = np.column_stack([x, y])
        segments = np.stack([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2.6, capstyle="round")
        lc.set_array((weights[:-1] + weights[1:]) / 2)
        axes[0].add_collection(lc)
        axes[0].text(1.035, y[-1], stage, color="black", va="center", fontsize=7)
        axes[1].plot(cache["tokens"], cache["profiles"][i], color=STYLE["stages"][stage],
                     lw=1.5, ls=["-", "--", "-."][i], label=stage)
    axes[0].set(xlim=(-.025, 1.28), ylim=(-.02, 1.99), ylabel="Voltage (offset)")
    axes[0].set_xticks([0, .5, 1])
    panel_xlabel(axes[0], "a", "Token position")
    axes[1].set(ylabel="Mean attention weight", xlim=(cache["tokens"].min(), cache["tokens"].max()))
    axes[1].legend(loc="upper right", fontsize=6.8)
    panel_xlabel(axes[1], "b", "Token index")
    bins = cache["focus_bins"]
    right = axes[2].twinx()
    right.spines["right"].set_visible(True)
    l1 = axes[2].plot(bins[:, 0], bins[:, 1], color="#2563A8", lw=1.4, marker="o", ms=2.5, label="Temperature")
    l2 = right.plot(bins[:, 0], bins[:, 2], color="#D65F5F", lw=1.3, ls="--", marker="s", ms=2.5, label="Entropy")
    axes[2].set_ylabel("Temperature")
    right.set_ylabel("Entropy")
    axes[2].legend(l1 + l2, [a.get_label() for a in l1 + l2], loc="upper right", fontsize=6.5)
    panel_xlabel(axes[2], "c", "Normalized cycle")
    for ax in axes[:3]:
        ax.grid(color=STYLE["grid_color"], lw=.5, ls="--")
        black_axes(ax)
    black_axes(right)
    center = cache["late_center"]
    for ax, name, pair, letter, xlabel, ylabel in zip(
        axes[3:], ["vs", "vc", "sc"], [(0, 1), (0, 2), (1, 2)], ["d", "e", "f"],
        ["Voltage, V", "Voltage, V", r"$\Delta V$"], [r"$\Delta V$", r"$\Delta^2 V$", r"$\Delta^2 V$"]):
        xx, yy, zz = [cache[f"{name}_{key}"] for key in "xyz"]
        cf = ax.contourf(xx, yy, zz, levels=np.linspace(0, 1, 16), cmap=STYLE["surface_cmap"])
        ax.contour(xx, yy, zz, levels=np.linspace(.15, .9, 6), colors="white", linewidths=.35, alpha=.45)
        ax.axvline(center[pair[0]], color="white", lw=.6, ls="--")
        ax.axhline(center[pair[1]], color="white", lw=.6, ls="--")
        ax.scatter(center[pair[0]], center[pair[1]], s=23, color="#F28E2B", edgecolor="white", lw=.6)
        panel_xlabel(ax, letter, xlabel)
        ax.set_ylabel(ylabel)
        black_axes(ax)
    box = axes[-1].get_position()
    cax = fig.add_axes([.92, box.y0, .013, box.height])
    cb = fig.colorbar(cf, cax=cax, ticks=np.linspace(0, 1, 6))
    cb.set_label("Relative compatibility", fontsize=7)
    save(fig, out, "attention_mechanism")


def effect_map(data, out, save):
    df = read_csv(data, "fig4_5a_effect_map_source.csv").sort_values("order")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.5), sharey=True)
    fig.subplots_adjust(left=.23, right=.98, bottom=.15, top=.97, wspace=.22)
    yy = np.arange(len(df))[::-1]
    for index, (ax, metric, err, label) in enumerate(zip(axes,
        ["delta_rmse_pct", "delta_mae_pct"], ["std_rmse_pct", "std_mae_pct"],
        [r"$\Delta$RMSE (% SOH)", r"$\Delta$MAE (% SOH)"])):
        ax.axvline(0, color="black", lw=.65)
        for y, row in zip(yy, df.to_dict("records")):
            ax.errorbar(row[metric], y, xerr=row[err], fmt="o", ms=4,
                        color=PALETTE[row["label"]], capsize=2, lw=1)
        ax.grid(axis="x", color=STYLE["grid_color"], lw=.5, ls="--")
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        panel_xlabel(ax, string.ascii_lowercase[index], label)
    axes[0].set_yticks(yy, df["label"])
    save(fig, out, "ablation_effect_map")


def violin(data, out, save):
    cell = read_csv(data, "fig4_5c_multiseed_cell_violin_source.csv")
    seed = read_csv(data, "fig4_5c_multiseed_seed_points_source.csv")
    methods = cell["label"].drop_duplicates().tolist()
    if cell.duplicated(["label", "seed", "battery_index"]).any():
        raise ValueError("Duplicate method/seed/cell rows")
    if seed.duplicated(["label", "seed"]).any():
        raise ValueError("Duplicate method/seed rows")
    fig, ax = plt.subplots(figsize=(8.0, 4.3))
    fig.subplots_adjust(left=.085, right=.98, bottom=.19, top=.97)
    for i, method in enumerate(methods, 1):
        vals = cell.loc[cell.label == method, "cell_rmse_pct"].to_numpy()
        color = PALETTE[method]
        if len(vals) >= 5 and np.ptp(vals) > 0:
            parts = ax.violinplot([vals], positions=[i], widths=.70, showextrema=False)
            parts["bodies"][0].set(facecolor=color, edgecolor=color, alpha=.22)
        ax.scatter(i + np.linspace(-.14, .14, len(vals)), vals, s=9, color=color, alpha=.45, edgecolor="none")
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax.plot([i, i], [q1, q3], color=color, lw=3)
        ax.plot([i - .17, i + .17], [med, med], color=color, lw=1.5)
        seed_vals = seed.loc[seed.label == method].sort_values("seed")["pooled_rmse_pct"].to_numpy()
        ax.scatter(i + np.linspace(-.16, .16, len(seed_vals)), seed_vals,
                   marker="D", s=20, color="black", edgecolor="white", lw=.4, zorder=5)
    short = [m.replace(" attention", "\nattention").replace("w/o degradation state", "w/o state")
             .replace("w/o curve tokens", "w/o curve").replace("fixed temperature", "fixed\ntemperature")
             .replace("w/o adaptive MMD", "w/o MMD").replace("w/o physical base", "w/o physical\nbase") for m in methods]
    ax.set_xticks(range(1, len(methods) + 1), short)
    ax.set_ylabel("Cell-wise RMSE (% SOH)")
    ax.set_ylim(0, cell.cell_rmse_pct.max() * 1.15)
    ax.grid(axis="y", color=STYLE["grid_color"], lw=.5, ls="--")
    ax.set_axisbelow(True)
    ax.legend(handles=[Line2D([], [], marker="D", color="none", markerfacecolor="black", ms=4, label="Seed pooled RMSE"),
                       Line2D([], [], color="black", lw=1.3, label="Cell median")], loc="upper left")
    save(fig, out, "cell_seed_violin")


def historical_module():
    path = PACK / "originals/scripts/visualize_attention_mechanism_ablation_v1.py"
    spec = importlib.util.spec_from_file_location("archived_ablation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bar_summary(data, out, save):
    df = read_csv(data, "attention_ablation_all_evidence_aggregate.csv").set_index("method")
    old = historical_module()
    df = df.loc[old.METHOD_ORDER]
    vals, err = df.mean_rmse.to_numpy() * 100, df.std_rmse.to_numpy() * 100
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    fig.subplots_adjust(left=.09, right=.98, top=.96, bottom=.24)
    bars = ax.bar(np.arange(len(df)), vals, yerr=err, capsize=3,
                  color=[old.METHOD_COLORS[m] for m in df.index], edgecolor="white", lw=.7)
    bars[0].set_hatch("////")
    ax.set_xticks(np.arange(len(df)), [old.METHOD_LABELS[m] for m in df.index], rotation=22, ha="right")
    ax.set_ylabel("Mean RMSE (% SOH)")
    ax.grid(axis="y", color=STYLE["grid_color"], lw=.5, ls="--")
    ax.set_axisbelow(True)
    bar_numbers_above_errors(ax, bars, vals, err)
    save(fig, out, "bar_error_labels")


def per_cell_error(data, out, save):
    df = read_csv(data, "attention_mechanism_seed42_per_cell_long.csv")
    df = df.loc[df.method == "adaptive-attention-v1"].sort_values("battery_index")
    fig, ax = plt.subplots(figsize=(7.72, 4.0))
    fig.subplots_adjust(left=.09, right=.98, bottom=.14, top=.97)
    x = np.arange(len(df))
    for offset, metric, color, hatch in [(-.18, "rmse", "#673AA5", "////"), (.18, "mae", "#BEA4DF", None)]:
        vals = df[metric].to_numpy() * 100
        bars = ax.bar(x + offset, vals, .36, color=color, edgecolor="white", lw=.6, hatch=hatch, label=metric.upper())
        for bar, val in zip(bars, vals):
            ax.annotate(f"{val:.2f}", (bar.get_x() + bar.get_width()/2, val), xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7, color="black")
    ax.set_xticks(x, [f"10-{i}" for i in range(1, len(df) + 1)])
    ax.set(xlabel="Cell ID", ylabel="Error (% SOH)", ylim=(0, df.rmse.max() * 100 * 1.22))
    ax.grid(axis="y", color=STYLE["grid_color"], lw=.5, ls="--")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")
    save(fig, out, "per_cell_error")


def prediction(data, out, save):
    df = read_csv(data, "adaptive_attention_v1_seed42_target_test_predictions.csv")
    cells = sorted(df.battery_index.unique())
    nrows = int(np.ceil(len(cells) / 4))
    fig, axes = plt.subplots(nrows, 4, figsize=(11, 2.5 * nrows), sharey=True, squeeze=False)
    fig.subplots_adjust(left=.065, right=.99, bottom=.16, top=.97, hspace=.47, wspace=.25)
    for i, (ax, cell) in enumerate(zip(axes.ravel(), cells)):
        sub = df.loc[df.battery_index == cell].sort_values("cycle")
        ax.plot(sub.cycle, sub.true_soh, color="#22A5AE", lw=1.3, label="True SOH")
        ax.plot(sub.cycle, sub.adaptive_pred, color="#7541BA", lw=1.0, ls="--", label="Predicted SOH")
        panel_xlabel(ax, string.ascii_lowercase[i], f"Cell {cell}: cycle")
        if i % 4 == 0:
            ax.set_ylabel("SOH")
    for ax in axes.ravel()[len(cells):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .005), ncol=2)
    save(fig, out, "prediction_curves")


def polar_cells(data, out, save):
    old = historical_module()
    df = read_csv(data, "attention_mechanism_seed42_per_cell_long.csv")
    methods = old.METHOD_ORDER
    cells = sorted(df.battery_index.unique())
    nrows = int(np.ceil(len(cells) / 4))
    fig, axes = plt.subplots(nrows, 4, figsize=(11.2, 3.0 * nrows), subplot_kw={"projection": "polar"}, squeeze=False)
    fig.subplots_adjust(left=.045, right=.965, top=.95, bottom=.20, wspace=.5, hspace=.66)
    angles = np.linspace(0, 2 * np.pi, len(methods), endpoint=False)
    limit = np.ceil(df.rmse.max() * 100 * 2) / 2
    for i, (ax, cell) in enumerate(zip(axes.flat, cells)):
        sub = df.loc[df.battery_index == cell].set_index("method")
        vals = sub.loc[methods, "rmse"].to_numpy() * 100
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.bar(angles, vals, width=2*np.pi/len(methods)*.70, color=[old.METHOD_COLORS[m] for m in methods], edgecolor="white", lw=.4)
        ax.set_ylim(0, limit)
        ax.set_xticks(angles, [str(j + 1) for j in range(len(methods))], fontsize=6)
        ax.set_yticks(np.linspace(0, limit, 4)[1:])
        ax.tick_params(axis="y", labelsize=5.5)
        ax.grid(color="#D9D9D9", lw=.45, ls="--")
        ax.spines["polar"].set_color("#CFCFCF")
        ax.text(.5, -.29, f"{string.ascii_lowercase[i]}) Cell {cell}", transform=ax.transAxes, ha="center")
    handles = [Line2D([], [], marker="s", color="none", markerfacecolor=old.METHOD_COLORS[m], ms=5,
                      label=f"{i + 1}: {old.METHOD_LABELS[m]}") for i, m in enumerate(methods)]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(.5, .005), fontsize=7)
    save(fig, out, "per_cell_polar_bars")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PACK / "data")
    parser.add_argument("--out-dir", type=Path, default=PACK / "outputs")
    parser.add_argument("--figure", choices=["all", "radar", "mechanism", "effect", "violin", "bar", "per-cell", "prediction", "polar"], default="all")
    parser.add_argument("--layout", choices=["both", "horizontal", "vertical"], default="both")
    parser.add_argument("--formats", nargs="+", choices=["png", "pdf", "svg", "tiff"], default=STYLE["formats"])
    parser.add_argument("--dpi", type=int, default=STYLE["dpi"])
    args = parser.parse_args()
    print("Font:", configure(), flush=True)
    save = lambda fig, out, stem: save_figure(fig, out, stem, dpi=args.dpi, formats=args.formats)
    if args.figure in ["all", "radar"]:
        for layout in (["horizontal", "vertical"] if args.layout == "both" else [args.layout]):
            radar(args.data_dir, args.out_dir, save, layout)
    for name, fn in [("mechanism", mechanism), ("effect", effect_map), ("violin", violin),
                     ("bar", bar_summary), ("per-cell", per_cell_error), ("prediction", prediction), ("polar", polar_cells)]:
        if args.figure in ["all", name]:
            fn(args.data_dir, args.out_dir, save)


if __name__ == "__main__":
    main()
