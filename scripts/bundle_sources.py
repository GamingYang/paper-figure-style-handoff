"""Copy the explicitly selected historical assets; never modify their sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

PACK = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    codex = root / "codex"
    section = codex / "section4-5-new-attention-ablation-figures-v1"
    mech = codex / "adaptive-attention-mechanism-ablation-v1"
    adapt = codex / "adaptive-attention-v1"
    pairs = []
    for folder, filenames in [
        (section, ["plot_stagewise_pentagon_radar_reference_style.py",
                   "plot_section4_5_new_attention_ablation_figures.py"]),
        (mech, ["plot_adaptive_attention_mechanism_response_surface.py",
                "visualize_attention_mechanism_ablation_v1.py"]),
    ]:
        for filename in filenames:
            pairs.append((folder / filename, PACK / "originals" / "scripts" / filename))
    for filename in ["fig4_5a_effect_map_source.csv", "fig4_5b_stage_profile_pentagon_source.csv",
                     "fig4_5b_stage_radar_source.csv", "fig4_5c_multiseed_cell_violin_source.csv",
                     "fig4_5c_multiseed_seed_points_source.csv"]:
        pairs.append((section / "source_data" / filename, PACK / "data" / filename))
    for filename in ["attention_mechanism_seed42_per_cell_long.csv",
                     "attention_ablation_all_evidence_aggregate.csv",
                     "attention_mechanism_seed42_stage_interpretability_long.csv"]:
        pairs.append((mech / filename, PACK / "data" / filename))
    pred = "adaptive_attention_v1_seed42_target_test_predictions.csv"
    pairs.append((adapt / pred, PACK / "data" / pred))
    for source in sorted((mech / "attention_mechanism_response_surface_source").glob("*.csv")):
        pairs.append((source, PACK / "data" / "mechanism_tables" / source.name))
    for stem in ["fig4_5a_ablation_static_mmd_effect_map", "fig4_5b_stagewise_rmse_radar",
                 "fig4_5b_stagewise_rmse_radar_horizontal", "fig4_5c_multiseed_stability_violin"]:
        for suffix in [".png", ".pdf"]:
            pairs.append((section / "figures" / (stem + suffix), PACK / "originals" / "figures" / (stem + suffix)))
    for source in sorted((mech / "figures").glob("*.png")):
        pairs.append((source, PACK / "originals" / "figures" / source.name))
    for filename in ["SECTION_4_5_NEW_ATTENTION_FIGURE_NOTES_20260729.md",
                     "SECTION_4_5_REVISED_TEXT_AND_CAPTIONS_20260729.md"]:
        pairs.append((section / filename, PACK / "originals" / "notes" / filename))
    pairs.append((mech / "fig06_attention_mechanism_caption_text_zh.txt",
                  PACK / "originals" / "notes" / "fig06_attention_mechanism_caption_text_zh.txt"))
    missing = [str(source) for source, _ in pairs if not source.is_file()]
    if missing:
        raise FileNotFoundError("Missing historical sources:\n" + "\n".join(missing))
    records = []
    for source, dest in pairs:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        records.append({"source": str(source), "bundled": dest.relative_to(PACK).as_posix(),
                        "bytes": dest.stat().st_size,
                        "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()})
    (PACK / "provenance.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Copied {len(records)} assets into {PACK}")


if __name__ == "__main__":
    main()
