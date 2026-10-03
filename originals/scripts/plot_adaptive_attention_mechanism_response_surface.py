"""Draw a mechanism-oriented response-surface figure for adaptive attention.

The figure uses the frozen adaptive-attention-v1 seed42 checkpoint and Batch10
blind-test samples. It visualizes how local voltage-curve tokens and degradation
state jointly affect learned attention focus.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Dict, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from torch.utils.data import DataLoader


ROOT = Path(r"C:\Users\ROG\Desktop\paper")
CODEX_DIR = ROOT / "codex"
ADAPTIVE_DIR = CODEX_DIR / "adaptive-attention-v1"
OUT_DIR = CODEX_DIR / "adaptive-attention-mechanism-ablation-v1"
FIG_DIR = OUT_DIR / "figures"
SOURCE_DIR = OUT_DIR / "attention_mechanism_response_surface_source"

ADAPTIVE_SCRIPT = ADAPTIVE_DIR / "run_adaptive_attention_v1.py"
ADAPTIVE_CKPT = ADAPTIVE_DIR / "adaptive-attention-v1-seed42" / "adaptive-attention-v1-seed42_best.pt"

GLOBAL_MAX_CYCLE = 3000.0
EVAL_BATCH_SIZE = 1024

STAGES = ("Early", "Middle", "Late")
STAGE_COLORS = {
    "Early": "#9AD7D5",
    "Middle": "#5EA6D7",
    "Late": "#4C3BCF",
}
ACCENT = "#F28E2B"
NEUTRAL = "#4A5568"
ATTN_CMAP = LinearSegmentedColormap.from_list("attention_strength", ["#8DA0B3", "#F28E2B"])
SHOW_TOP_ATTENTION_POINTS = False


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.sans-serif": ["Times New Roman", "Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "mathtext.fontset": "stix",
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def stage_masks(p_norm: np.ndarray) -> Dict[str, np.ndarray]:
    p = p_norm.reshape(-1)
    return {
        "Early": p < 1.0 / 3.0,
        "Middle": (p >= 1.0 / 3.0) & (p < 2.0 / 3.0),
        "Late": p >= 2.0 / 3.0,
    }


def entropy(attn: np.ndarray) -> np.ndarray:
    return -np.sum(attn * np.log(np.clip(attn, 1e-12, None)), axis=1)


def load_model_and_data(device: torch.device):
    adaptive_mod = load_module(ADAPTIVE_SCRIPT, "adaptive_attention_v1_mod_for_mechanism")

    source_x, source_y, source_p, _, _ = adaptive_mod.load_split("source", "nominal")
    target_x, target_y, target_p, target_bidx, _ = adaptive_mod.load_split("target_test", "nominal")

    coeffs = np.polyfit(source_p.flatten() / GLOBAL_MAX_CYCLE, source_y.flatten(), 3)
    coeffs_t = torch.tensor(coeffs, dtype=torch.float32, device=device)

    model = adaptive_mod.ResidualCRNN(coeffs_t, use_physical_base=True).to(device)
    ckpt = torch.load(ADAPTIVE_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dataset = adaptive_mod.HybridDataset(target_x, target_y, target_p, target_bidx)
    return adaptive_mod, model, dataset, target_x, target_y, target_p, target_bidx


def collect_attention_details(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    loader = DataLoader(dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)
    details = {
        "p_norm": [],
        "base_soh": [],
        "true_soh": [],
        "battery_index": [],
        "curve": [],
        "attn": [],
        "temperature": [],
        "context": [],
        "state": [],
    }

    with torch.no_grad():
        for x, y, p, b in loader:
            x = x.float().to(device)
            p = p.float().to(device)
            y = y.float().to(device)

            a, bcoef, c, d = model.base_coeffs
            base_soh = a * (p**3) + bcoef * (p**2) + c * p + d
            conv = model.cnn(x.unsqueeze(1)).permute(0, 2, 1)
            lstm_out, _ = model.lstm(conv)
            state = torch.cat([p, base_soh, 1.0 - base_soh], dim=1)
            curve = model.attention.curve_tokens(x, lstm_out.size(1))
            context, attn, temperature = model.attention(lstm_out, x, state)

            details["p_norm"].append(p.cpu().numpy().reshape(-1))
            details["base_soh"].append(base_soh.cpu().numpy().reshape(-1))
            details["true_soh"].append(y.cpu().numpy().reshape(-1))
            details["battery_index"].append(b.numpy().reshape(-1))
            details["curve"].append(curve.cpu().numpy())
            details["attn"].append(attn.cpu().numpy().squeeze(-1))
            details["temperature"].append(temperature.cpu().numpy().reshape(-1))
            details["context"].append(context.cpu().numpy())
            details["state"].append(state.cpu().numpy())

    return {
        "p_norm": np.concatenate(details["p_norm"]),
        "base_soh": np.concatenate(details["base_soh"]),
        "true_soh": np.concatenate(details["true_soh"]),
        "battery_index": np.concatenate(details["battery_index"]).astype(int),
        "curve": np.concatenate(details["curve"], axis=0),
        "attn": np.concatenate(details["attn"], axis=0),
        "temperature": np.concatenate(details["temperature"]),
        "context": np.concatenate(details["context"], axis=0),
        "state": np.concatenate(details["state"], axis=0),
    }


def representative_indices(p_norm: np.ndarray) -> Dict[str, int]:
    masks = stage_masks(p_norm)
    reps = {}
    for stage, mask in masks.items():
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue
        target = np.median(p_norm[idx])
        reps[stage] = int(idx[np.argmin(np.abs(p_norm[idx] - target))])
    return reps


def weighted_stage_centers(details: Dict[str, np.ndarray]) -> Dict[str, Dict[str, np.ndarray]]:
    centers: Dict[str, Dict[str, np.ndarray]] = {}
    p_norm = details["p_norm"]
    curve = details["curve"]
    attn = details["attn"]
    masks = stage_masks(p_norm)
    for stage, mask in masks.items():
        if not np.any(mask):
            continue
        stage_curve = curve[mask].reshape(-1, 3)
        stage_attn = attn[mask].reshape(-1)
        stage_attn = stage_attn / np.clip(stage_attn.sum(), 1e-12, None)
        curve_center = np.sum(stage_curve * stage_attn[:, None], axis=0)
        context_center = details["context"][mask].mean(axis=0)
        state_center = details["state"][mask].mean(axis=0)
        centers[stage] = {
            "curve": curve_center,
            "context": context_center,
            "state": state_center,
        }
    return centers


def score_surface(
    attention_module: torch.nn.Module,
    context_center: np.ndarray,
    state_center: np.ndarray,
    pair: Tuple[int, int],
    fixed_curve: np.ndarray,
    ranges: Dict[int, Tuple[float, float]],
    device: torch.device,
    n_grid: int = 120,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_idx, y_idx = pair
    xs = np.linspace(ranges[x_idx][0], ranges[x_idx][1], n_grid)
    ys = np.linspace(ranges[y_idx][0], ranges[y_idx][1], n_grid)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.repeat(fixed_curve.reshape(1, 3), xx.size, axis=0)
    grid[:, x_idx] = xx.reshape(-1)
    grid[:, y_idx] = yy.reshape(-1)

    with torch.no_grad():
        curve_t = torch.tensor(grid, dtype=torch.float32, device=device)
        hidden_t = torch.tensor(context_center.reshape(1, -1), dtype=torch.float32, device=device)
        state_t = torch.tensor(state_center.reshape(1, -1), dtype=torch.float32, device=device)
        query = attention_module.query_proj(state_t)
        hidden = attention_module.hidden_proj(hidden_t)
        temp = attention_module.tau_min + torch.nn.functional.softplus(attention_module.tau_proj(state_t))
        score_feat = torch.tanh(hidden + attention_module.curve_proj(curve_t) + query)
        scores = torch.sum(query * score_feat, dim=-1) / temp.reshape(1)
        zz = scores.detach().cpu().numpy().reshape(n_grid, n_grid)

    zz = (zz - zz.min()) / np.clip(zz.max() - zz.min(), 1e-12, None)
    return xx, yy, zz


def save_source_tables(details: Dict[str, np.ndarray], centers: Dict[str, Dict[str, np.ndarray]]) -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    tokens = np.arange(1, details["attn"].shape[1] + 1)
    profile_rows = []
    masks = stage_masks(details["p_norm"])
    for stage, mask in masks.items():
        if not np.any(mask):
            continue
        mean_profile = details["attn"][mask].mean(axis=0)
        for token, val in zip(tokens, mean_profile):
            profile_rows.append({"stage": stage, "token": int(token), "mean_attention": float(val)})
    pd.DataFrame(profile_rows).to_csv(SOURCE_DIR / "stage_attention_profiles.csv", index=False)

    bin_rows = []
    p = details["p_norm"]
    ent = entropy(details["attn"])
    bins = np.linspace(p.min(), p.max(), 13)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & (p < hi)
        if not np.any(mask):
            continue
        bin_rows.append(
            {
                "p_norm_mid": float((lo + hi) / 2),
                "n": int(mask.sum()),
                "mean_temperature": float(details["temperature"][mask].mean()),
                "mean_entropy": float(ent[mask].mean()),
            }
        )
    pd.DataFrame(bin_rows).to_csv(SOURCE_DIR / "temperature_entropy_by_cycle_bin.csv", index=False)

    center_rows = []
    for stage, vals in centers.items():
        center_rows.append(
            {
                "stage": stage,
                "state_p_norm": float(vals["state"][0]),
                "state_base_soh": float(vals["state"][1]),
                "state_degradation_margin": float(vals["state"][2]),
                "curve_voltage": float(vals["curve"][0]),
                "curve_slope": float(vals["curve"][1]),
                "curve_curvature": float(vals["curve"][2]),
            }
        )
    pd.DataFrame(center_rows).to_csv(SOURCE_DIR / "attention_weighted_stage_centers.csv", index=False)


def panel_label(ax, label: str) -> None:
    ax.text(
        0.015,
        0.985,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.2, "alpha": 0.9},
    )


def draw_representative_curves(ax, details: Dict[str, np.ndarray]) -> None:
    reps = representative_indices(details["p_norm"])
    token_pos = np.linspace(0, 1, details["curve"].shape[1])
    offsets = {"Early": 1.35, "Middle": 0.70, "Late": 0.05}
    attn_norm = Normalize(vmin=0.0, vmax=float(np.quantile(details["attn"], 0.995)))
    for stage in STAGES:
        idx = reps.get(stage)
        if idx is None:
            continue
        voltage = details["curve"][idx, :, 0]
        attn = details["attn"][idx]
        v = (voltage - voltage.min()) / np.clip(voltage.max() - voltage.min(), 1e-12, None)
        y = 0.52 * v + offsets[stage]
        points = np.column_stack([token_pos, y]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        segment_attn = (attn[:-1] + attn[1:]) / 2.0
        lc = LineCollection(
            segments,
            cmap=ATTN_CMAP,
            norm=attn_norm,
            linewidth=3.0,
            capstyle="round",
            zorder=3,
        )
        lc.set_array(segment_attn)
        ax.add_collection(lc)
        ax.plot(token_pos, y, color=STAGE_COLORS[stage], lw=1.0, alpha=0.90, zorder=2)
        ax.text(1.025, y[-1], stage, color=STAGE_COLORS[stage], va="center", fontsize=7)
    ax.set_title("Curve tokens")
    ax.set_xlabel("a) Token position")
    ax.set_ylabel("Voltage (offset)")
    ax.set_xlim(-0.02, 1.12)
    ax.set_ylim(-0.02, 1.95)
    ax.grid(True, color="#EBEBEB", linestyle="--", linewidth=0.55)


def draw_stage_attention_profiles(ax, details: Dict[str, np.ndarray]) -> None:
    tokens = np.arange(1, details["attn"].shape[1] + 1)
    masks = stage_masks(details["p_norm"])
    for stage in STAGES:
        mask = masks[stage]
        if not np.any(mask):
            continue
        profile = details["attn"][mask].mean(axis=0)
        ax.plot(tokens, profile, color=STAGE_COLORS[stage], lw=1.6, label=stage)
        ax.fill_between(tokens, 0, profile, color=STAGE_COLORS[stage], alpha=0.10)
    ax.set_title("Attention profile")
    ax.set_xlabel("b) Token index")
    ax.set_ylabel("Mean attention weight")
    ax.set_xlim(tokens.min(), tokens.max())
    ax.grid(True, color="#EBEBEB", linestyle="--", linewidth=0.55)
    ax.legend(loc="upper left", fontsize=6.5, frameon=False)


def draw_temperature_entropy(ax, details: Dict[str, np.ndarray]) -> None:
    p = details["p_norm"]
    ent = entropy(details["attn"])
    bins = np.linspace(p.min(), p.max(), 13)
    mids, temp_vals, ent_vals = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & (p < hi)
        if not np.any(mask):
            continue
        mids.append((lo + hi) / 2)
        temp_vals.append(details["temperature"][mask].mean())
        ent_vals.append(ent[mask].mean())
    mids = np.asarray(mids)
    temp_vals = np.asarray(temp_vals)
    ent_vals = np.asarray(ent_vals)

    l1 = ax.plot(mids, temp_vals, color="#2563A8", marker="o", ms=2.6, lw=1.4, label="Temperature")
    ax.set_xlabel("c) Normalized cycle")
    ax.set_ylabel("Temperature", color="#111111")
    ax.tick_params(axis="y", labelcolor="#111111")
    ax.grid(True, color="#EBEBEB", linestyle="--", linewidth=0.55)
    ax2 = ax.twinx()
    l2 = ax2.plot(mids, ent_vals, color="#D65F5F", linestyle="--", marker="s", ms=2.5, lw=1.2, label="Entropy")
    ax2.set_ylabel("Entropy", color="#111111")
    ax2.tick_params(axis="y", labelcolor="#111111")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_linewidth(0.8)
    ax.set_title("Focus intensity")
    lines = l1 + l2
    ax.legend(lines, [line.get_label() for line in lines], loc="upper right", fontsize=6.2, frameon=False)


def draw_surface_panel(
    ax,
    xx: np.ndarray,
    yy: np.ndarray,
    zz: np.ndarray,
    center: np.ndarray,
    pair: Tuple[int, int],
    title: str,
    xlabel: str,
    ylabel: str,
    top_points: np.ndarray,
):
    levels = np.linspace(0, 1, 16)
    cf = ax.contourf(xx, yy, zz, levels=levels, cmap="YlGnBu_r", extend="both")
    ax.contour(xx, yy, zz, levels=np.linspace(0.15, 0.9, 6), colors="white", linewidths=0.35, alpha=0.45)
    if SHOW_TOP_ATTENTION_POINTS and top_points.size:
        ax.scatter(
            top_points[:, pair[0]],
            top_points[:, pair[1]],
            s=3.2,
            color="#111827",
            alpha=0.11,
            linewidth=0,
            rasterized=True,
        )
    ax.axvline(center[pair[0]], color="white", lw=0.65, ls="--", alpha=0.85)
    ax.axhline(center[pair[1]], color="white", lw=0.65, ls="--", alpha=0.85)
    ax.scatter(center[pair[0]], center[pair[1]], s=23, color=ACCENT, edgecolor="white", linewidth=0.45, zorder=4)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for spine in ax.spines.values():
        spine.set_color("#6B7280")
    return cf


def draw_figure(details: Dict[str, np.ndarray], model: torch.nn.Module, device: torch.device) -> Path:
    centers = weighted_stage_centers(details)
    save_source_tables(details, centers)

    curve_all = details["curve"].reshape(-1, 3)
    late_mask = stage_masks(details["p_norm"])["Late"]
    late_curve = details["curve"][late_mask].reshape(-1, 3)
    late_attn = details["attn"][late_mask].reshape(-1)
    threshold = np.quantile(late_attn, 0.92)
    top_points = late_curve[late_attn >= threshold]
    pd.DataFrame(
        top_points,
        columns=["voltage", "slope", "curvature"],
    ).to_csv(SOURCE_DIR / "late_stage_top_attention_curve_tokens.csv", index=False)
    ranges = {
        0: tuple(np.percentile(curve_all[:, 0], [3, 97])),
        1: tuple(np.percentile(curve_all[:, 1], [2, 98])),
        2: tuple(np.percentile(curve_all[:, 2], [2, 98])),
    }
    late = centers["Late"]
    fixed_curve = late["curve"].copy()

    surfaces = [
        ((0, 1), r"$V$-$\Delta V$ response", "d) Voltage, V", r"$\Delta V$"),
        ((0, 2), r"$V$-$\Delta^2 V$ response", "e) Voltage, V", r"$\Delta^2 V$"),
        ((1, 2), r"$\Delta V$-$\Delta^2 V$ response", r"f) $\Delta V$", r"$\Delta^2 V$"),
    ]

    surface_data = []
    for pair, title, xlabel, ylabel in surfaces:
        xx, yy, zz = score_surface(
            model.attention,
            late["context"],
            late["state"],
            pair,
            fixed_curve,
            ranges,
            device,
        )
        surface_data.append((pair, title, xlabel, ylabel, xx, yy, zz))

    fig = plt.figure(figsize=(7.4, 5.05), constrained_layout=False)
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.0, 1.05], hspace=0.43, wspace=0.38)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    draw_representative_curves(ax_a, details)
    draw_stage_attention_profiles(ax_b, details)
    draw_temperature_entropy(ax_c, details)

    surface_axes = [fig.add_subplot(gs[1, i]) for i in range(3)]
    cf = None
    for i, (ax, data) in enumerate(zip(surface_axes, surface_data)):
        pair, title, xlabel, ylabel, xx, yy, zz = data
        cf = draw_surface_panel(ax, xx, yy, zz, late["curve"], pair, title, xlabel, ylabel, top_points)

    if cf is not None:
        cax = fig.add_axes([0.935, 0.125, 0.012, 0.330])
        cbar = fig.colorbar(cf, cax=cax)
        cbar.set_label("Compatibility")
        cbar.ax.tick_params(labelsize=6)
    fig.subplots_adjust(left=0.075, right=0.895, top=0.955, bottom=0.085)

    out_base = FIG_DIR / "fig06_adaptive_attention_mechanism_response_surface"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(out_base.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_base.with_suffix(".png")


def main() -> None:
    set_style()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, model, dataset, *_ = load_model_and_data(device)
    details = collect_attention_details(model, dataset, device)
    out = draw_figure(details, model, device)
    print(out)


if __name__ == "__main__":
    main()
