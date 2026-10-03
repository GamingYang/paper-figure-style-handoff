"""Optional one-time inference export. Ordinary figure rendering does not need torch."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

# Anaconda MKL and PyTorch can otherwise initialize separate OpenMP runtimes.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import numpy as np

PACK = Path(__file__).resolve().parents[1]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    args = parser.parse_args()
    import torch

    root = args.project_root.resolve()
    script = root / "codex/adaptive-attention-v1/run_adaptive_attention_v1.py"
    checkpoint = script.parent / "adaptive-attention-v1-seed42/adaptive-attention-v1-seed42_best.pt"
    model_mod = load_module(script, "frozen_attention_export")
    model_mod.DATA_DIR = root / "PYthon训练代码/processed_clean"
    old = load_module(PACK / "originals/scripts/plot_adaptive_attention_mechanism_response_surface.py", "historical_mechanism")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, sy, sp, _, _ = model_mod.load_split("source", "nominal")
    x, y, p, b, _ = model_mod.load_split("target_test", "nominal")
    coeffs = np.polyfit(sp.flatten() / 3000.0, sy.flatten(), 3)
    model = model_mod.ResidualCRNN(torch.tensor(coeffs, dtype=torch.float32, device=device), use_physical_base=True).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    dataset = model_mod.HybridDataset(x, y, p, b)
    print(f"Exporting frozen seed42: {len(dataset)} samples on {device}", flush=True)
    details = old.collect_attention_details(model, dataset, device)
    reps = old.representative_indices(details["p_norm"])
    centers = old.weighted_stage_centers(details)
    masks = old.stage_masks(details["p_norm"])
    tokens = np.arange(1, details["attn"].shape[1] + 1)
    rep_ids = np.array([reps[s] for s in old.STAGES])
    arrays = {
        "stages": np.array(old.STAGES), "representative_indices": rep_ids,
        "representative_curve": details["curve"][rep_ids],
        "representative_attention": details["attn"][rep_ids],
        "representative_battery": details["battery_index"][rep_ids],
        "representative_p_norm": details["p_norm"][rep_ids],
        "tokens": tokens,
        "profiles": np.stack([details["attn"][masks[s]].mean(axis=0) for s in old.STAGES]),
        "attention_vmax": np.array(np.quantile(details["attn"], .995)),
    }
    pn = details["p_norm"]
    entropy = old.entropy(details["attn"])
    bins = np.linspace(pn.min(), pn.max(), 13)
    rows = []
    # Keep the historical half-open bins for exact provenance, including the last bin.
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (pn >= lo) & (pn < hi)
        if mask.any():
            rows.append([(lo + hi) / 2, details["temperature"][mask].mean(), entropy[mask].mean(), mask.sum()])
    arrays["focus_bins"] = np.array(rows)
    curve = details["curve"].reshape(-1, 3)
    ranges = {i: tuple(np.percentile(curve[:, i], bounds)) for i, bounds in enumerate([(3, 97), (2, 98), (2, 98)])}
    late = centers["Late"]
    arrays["late_center"] = late["curve"]
    for name, pair in [("vs", (0, 1)), ("vc", (0, 2)), ("sc", (1, 2))]:
        xx, yy, zz = old.score_surface(model.attention, late["context"], late["state"], pair,
                                       late["curve"], ranges, device)
        arrays[f"{name}_x"], arrays[f"{name}_y"], arrays[f"{name}_z"] = xx, yy, zz
    out = PACK / "data/attention_mechanism_cache.npz"
    np.savez_compressed(out, **arrays)
    meta = {
        "model": "adaptive-attention-v1", "seed": 42, "split": "Batch10 target_test",
        "purpose": "Frozen post-hoc visualization only; no training or model selection",
        "checkpoint_source": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "extraction_source": str(script), "torch_version": torch.__version__, "device": str(device),
        "n_samples": len(dataset), "n_tokens": len(tokens),
        "stage_definition": "p_norm=cycle/3000; Early<1/3; Middle in [1/3,2/3); Late>=2/3",
        "surface_definition": "Historical conditional surrogate: pooled attention context is fed to hidden_proj; other curve coordinate fixed at late-stage attention-weighted center; each surface independently min-max normalized.",
        "limitations": ["Not RMSE optimization or hyperparameter search", "Not softmax attention probability", "Pooled context is not a token hidden state", "Separate min-max normalization prevents absolute cross-surface comparison", "Synthetic feature combinations need not lie on observed voltage curves", "Historical last cycle bin excludes its maximum boundary"],
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(out, flush=True)


if __name__ == "__main__":
    main()
