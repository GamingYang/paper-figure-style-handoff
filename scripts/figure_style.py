"""Shared typography, point-offset annotations, export, and layout diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text, Annotation
from matplotlib.transforms import Bbox
import numpy as np

PACK = Path(__file__).resolve().parents[1]
STYLE = json.loads((PACK / "style.json").read_text(encoding="utf-8"))


def configure():
    font = STYLE["font"]
    font_path = font_manager.findfont(font_manager.FontProperties(family=font), fallback_to_default=False)
    plt.rcParams.update({
        "font.family": font, "font.size": STYLE["font_size"],
        "mathtext.fontset": "custom", "mathtext.rm": font,
        "mathtext.it": f"{font}:italic", "mathtext.bf": f"{font}:bold",
        "mathtext.cal": font, "mathtext.sf": font, "mathtext.tt": font,
        "mathtext.fallback": None, "axes.labelcolor": "black",
        "text.color": "black", "xtick.color": "black", "ytick.color": "black",
        "axes.edgecolor": "black", "axes.linewidth": STYLE["spine_width"],
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "figure.dpi": 120, "savefig.facecolor": "white",
    })
    return font_path


def panel_xlabel(ax, letter, label):
    ax.set_xlabel(f"{letter}) {label}", labelpad=6)


def black_axes(ax):
    ax.tick_params(axis="both", colors="black")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    for spine in ax.spines.values():
        spine.set_color("black")


def bar_numbers_above_errors(ax, bars, values, errors, decimals=3):
    for bar, value, error in zip(bars, values, errors):
        ax.annotate(f"{value:.{decimals}f}",
                    (bar.get_x() + bar.get_width() / 2, value + error),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom", color="black", fontsize=7)
    top = float(np.max(np.asarray(values) + np.asarray(errors)))
    ax.set_ylim(0, top * 1.18)


def annotation_bbox(annotation, renderer):
    # Annotation.get_window_extent also includes leader lines; use the text box only.
    patch = annotation.get_bbox_patch()
    return patch.get_window_extent(renderer) if patch is not None else Text.get_window_extent(annotation, renderer)


def place_near_point(ax, xy, text, occupied, direction=(0., 1.)):
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    vx, vy = direction
    tangent = (-vy, vx)
    candidates = []
    for distance in [8, 13, 19, 26, 33]:
        for side in [0, 1, -1, 2, -2]:
            candidates.append((vx * distance + tangent[0] * side * 11,
                               vy * distance + tangent[1] * side * 11))
    artist = ax.annotate(text, xy=xy, xytext=candidates[0], textcoords="offset points",
                         ha="center", va="center", fontsize=6.8, color="black",
                         bbox={"facecolor": "white", "edgecolor": "none", "pad": .25, "alpha": .94},
                         arrowprops={"arrowstyle": "-", "color": "#737373", "lw": .35,
                                     "shrinkA": 1.5, "shrinkB": 3}, zorder=20,
                         annotation_clip=False)
    for offset in candidates:
        artist.set_position(offset)
        artist.update_positions(renderer)
        artist.update_bbox_position_size(renderer)
        box = annotation_bbox(artist, renderer).expanded(1.10, 1.15)
        if not any(box.overlaps(existing) for existing in occupied):
            occupied.append(box)
            return artist
    raise RuntimeError("Cannot place all radar values nearby. Split the panel or enlarge it.")


def layout_diagnostics(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    excluded = set()
    for ax in fig.axes:
        for axis in [ax.xaxis, ax.yaxis]:
            low, high = sorted(axis.get_view_interval())
            tolerance = max(abs(high - low) * 1e-8, 1e-12)
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if (not ax.get_visible() or not axis.get_visible()
                        or not low - tolerance <= tick.get_loc() <= high + tolerance):
                    excluded.update([id(tick.label1), id(tick.label2)])
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    # Omit text on hidden axes (e.g. unused cells in a small-multiple grid).
    texts = [t for t in texts if id(t) not in excluded and (t.axes is None or t.axes.get_visible())]
    entries = []
    for t in texts:
        box = annotation_bbox(t, renderer) if isinstance(t, Annotation) else t.get_window_extent(renderer)
        if np.isfinite(box.extents).all() and box.width > 0 and box.height > 0:
            entries.append((t.get_text(), box))
    overlaps = []
    for i, (left, lb) in enumerate(entries):
        for right, rb in entries[i + 1:]:
            hit = Bbox.intersection(lb, rb)
            if hit is not None and hit.width > 1.5 and hit.height > 1.5:
                overlaps.append([left, right])
    return {"text_pair_overlap_candidates": overlaps,
            "text_count": len(entries),
            "font_families": sorted({str(t.get_fontfamily()) for t in texts}),
            "note": "Geometric screening only; inspect the exported PNG and PDF for line/legend/colorbar interference."}


def save_figure(fig, out_dir, stem, dpi=None, formats=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = layout_diagnostics(fig)
    for suffix in formats or STYLE["formats"]:
        kwargs = {"dpi": dpi or STYLE["dpi"], "bbox_inches": "tight", "pad_inches": .08}
        if suffix == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(out_dir / f"{stem}.{suffix}", **kwargs)
    (out_dir / f"{stem}.layout.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    plt.close(fig)
    print(f"Saved {stem}: {len(report['text_pair_overlap_candidates'])} text overlap candidates", flush=True)
    return report
