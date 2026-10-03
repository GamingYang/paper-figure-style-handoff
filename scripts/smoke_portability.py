"""Relocate plotting code and data, then render without the old project layout."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PACK = Path(__file__).resolve().parents[1]


def main():
    temp_root = Path(tempfile.mkdtemp(prefix="paper-figure-style-smoke-")).resolve()
    shutil.copy2(PACK / "style.json", temp_root / "style.json")
    shutil.copytree(PACK / "data", temp_root / "data")
    (temp_root / "scripts").mkdir()
    for name in ["render_figures.py", "figure_style.py"]:
        shutil.copy2(PACK / "scripts" / name, temp_root / "scripts" / name)
    archive = temp_root / "originals/scripts"
    archive.mkdir(parents=True)
    shutil.copy2(PACK / "originals/scripts/visualize_attention_mechanism_ablation_v1.py", archive)
    command = [sys.executable, "-B", str(temp_root / "scripts/render_figures.py"),
               "--formats", "png", "--dpi", "150"]
    result = subprocess.run(command, cwd=temp_root, text=True, capture_output=True, timeout=180)
    figures = list((temp_root / "outputs").glob("*.png"))
    report = {"status": "PASS" if result.returncode == 0 and len(figures) == 9 else "FAIL",
              "relocated_root": str(temp_root), "python": sys.executable,
              "returncode": result.returncode, "png_count": len(figures),
              "stdout": result.stdout, "stderr": result.stderr,
              "scope": "Bundled CSV/NPZ rendering in a new directory; no checkpoint or training files copied.",
              "cleanup": "Temporary test directory is retained for inspection; no deletion performed."}
    (PACK / "qa").mkdir(exist_ok=True)
    (PACK / "qa/portability.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
