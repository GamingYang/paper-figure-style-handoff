"""Validate source integrity, real sample structure, and rendered artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

PACK = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_gallery(files, output):
    width, height, columns = 640, 430, 3
    rows = int(np.ceil(len(files) / columns))
    canvas = Image.new("RGB", (width * columns, height * rows), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("times.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    for i, path in enumerate(files):
        tile = Image.open(path).convert("RGB")
        tile.thumbnail((width - 24, height - 54), Image.Resampling.LANCZOS)
        x, y = (i % columns) * width, (i // columns) * height
        canvas.paste(tile, (x + (width - tile.width) // 2, y + 10))
        draw.text((x + 16, y + height - 30), path.stem, fill="black", font=font)
    canvas.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    qa = PACK / "qa"
    qa.mkdir(exist_ok=True)
    errors = []
    provenance = json.loads((PACK / "provenance.json").read_text(encoding="utf-8"))
    for entry in provenance:
        path = PACK / entry["bundled"]
        if not path.is_file() or digest(path) != entry["sha256"]:
            errors.append(f"Historical source copy mismatch: {entry['bundled']}")
    data = PACK / "data"
    seed = pd.read_csv(data / "fig4_5c_multiseed_seed_points_source.csv")
    cell = pd.read_csv(data / "fig4_5c_multiseed_cell_violin_source.csv")
    counts = {name: {"seeds": sorted(int(v) for v in sub.seed.unique()),
                     "n_cell_seed": int((cell.label == name).sum()),
                     "n_cells": int(cell.loc[cell.label == name, "battery_index"].nunique())}
              for name, sub in seed.groupby("label")}
    if seed.duplicated(["label", "seed"]).any() or cell.duplicated(["label", "seed", "battery_index"]).any():
        errors.append("Duplicate source observations")
    effect = pd.read_csv(data / "fig4_5a_effect_map_source.csv").set_index("label")
    for metric in ["rmse", "mae"]:
        expected = effect[f"mean_{metric}_pct"] - effect.loc["Proposed", f"mean_{metric}_pct"]
        if not np.allclose(expected, effect[f"delta_{metric}_pct"]):
            errors.append(f"Delta arithmetic mismatch: {metric}")
    radar = pd.read_csv(data / "fig4_5b_stage_profile_pentagon_source.csv")
    table = radar.pivot(index="label", columns="axis", values="rmse_pct")
    for method in table.index:
        if not np.isclose(table.loc[method, "Overall"], effect.loc[method, "mean_rmse_pct"]):
            errors.append(f"Radar overall inconsistent: {method}")
    cache = np.load(data / "attention_mechanism_cache.npz", allow_pickle=False)
    if not np.allclose(cache["profiles"].sum(axis=1), 1, atol=1e-5):
        errors.append("Stage attention weights do not sum to 1")
    if not np.allclose(cache["representative_attention"].sum(axis=1), 1, atol=1e-5):
        errors.append("Representative attention weights do not sum to 1")
    stage_csv = pd.read_csv(data / "mechanism_tables/stage_attention_profiles.csv")
    for i, stage in enumerate(cache["stages"]):
        series = stage_csv.loc[stage_csv.stage == stage].sort_values("token").mean_attention.to_numpy()
        if not np.allclose(series, cache["profiles"][i], atol=1e-5):
            errors.append(f"Cache differs from historical profiles: {stage}")
    output = PACK / "outputs"
    expected_stems = ["radar_horizontal", "radar_vertical", "attention_mechanism", "ablation_effect_map",
                      "cell_seed_violin", "bar_error_labels", "per_cell_error", "prediction_curves", "per_cell_polar_bars"]
    layouts = {}
    images = []
    for stem in expected_stems:
        for suffix in ["png", "pdf", "svg", "layout.json"]:
            if not (output / f"{stem}.{suffix}").is_file():
                errors.append(f"Missing render: {stem}.{suffix}")
        path = output / f"{stem}.layout.json"
        if path.is_file():
            report = json.loads(path.read_text(encoding="utf-8"))
            layouts[stem] = report
            if report["text_pair_overlap_candidates"]:
                errors.append(f"Unresolved text overlap candidates: {stem}")
            if report["font_families"] != ["['Times New Roman']"]:
                errors.append(f"Unexpected figure fonts: {stem}")
        image_path = output / f"{stem}.png"
        if image_path.is_file():
            images.append(image_path)
            with Image.open(image_path) as img:
                if min(img.size) < 1000:
                    errors.append(f"Unexpected low-resolution output: {stem}")
    make_gallery(images, qa / "gallery.png")
    manifest_path = PACK / "MANIFEST_SHA256.json"
    if manifest_path.exists() and not args.write_manifest:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in manifest:
            path = PACK / item["path"]
            if not path.is_file() or digest(path) != item["sha256"]:
                errors.append(f"Manifest mismatch: {item['path']}")
    report = {"historical_files_verified": len(provenance), "seed_cell_structure": counts,
              "proposed_mean_rmse_pct": float(effect.loc["Proposed", "mean_rmse_pct"]),
              "figure_count": len(images), "layouts": layouts, "errors": errors,
              "status": "PASS" if not errors else "NEEDS_REVIEW",
              "scope": "Data/typography/file checks. Manual visual review remains required."}
    (qa / "validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.write_manifest:
        excluded = {"MANIFEST_SHA256.json", "qa/validation.json", "qa/gallery.png"}
        items = [{"path": p.relative_to(PACK).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)}
                 for p in sorted(PACK.rglob("*")) if p.is_file()
                 and "__pycache__" not in p.parts and p.relative_to(PACK).as_posix() not in excluded]
        manifest_path.write_text(json.dumps(items, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "figures": len(images), "errors": errors}, indent=2), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
