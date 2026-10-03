---
name: paper-figure-style-handoff
description: Reuse this user's established scientific figure style across manuscripts, including Times New Roman, bottom panel labels, uncluttered radar panels, attention surfaces, violin distributions, and export validation. Use with this bundled style pack when reproducing or adapting these research plots.
---

# Paper Figure Style Handoff

Read `references/USER_REQUIREMENTS.md` before choosing a design. It separates the user's explicit requirements from implementation recommendations. This folder is self-contained for plotting; use paths relative to this SKILL.md.

## Required Visual Choices

- Use Times New Roman for all English, digits, legends, ticks, and compatible mathematical glyphs. Verify the actual font exists; do not silently substitute.
- Use black numeric annotations and black axes, tick labels, and axis titles. Keep multi-axis curves colored and distinguish them with a concise combined legend, line styles, or markers.
- Put panel identifiers at the bottom next to the x-axis description: `a) Token position`. Radar panels can use `a) MMD control` below the panel. Do not put identifiers near the y-axis or upper-left title.
- Omit the overall figure title and long in-figure explanatory footer. Put mechanism details and interpretation in a separate caption file. Keep any short local titles only if needed.
- Prevent label/errorbar, label/legend, subplot, and colorbar collisions. Anchor numbers above the full errorbar with a point offset. Reserve a separate colorbar area.
- Radar: prefer the reference pentagon appearance with distinct hues, light fills, clear outlines, and separate panels by comparison question. Only use five axes when five real, defined quantities are available. Never repeat or invent dimensions to obtain a pentagon.
- Keep radar values close to their own vertex/line and black. When close methods are hard to label, split panels or use short leader lines; do not scatter numbers across the figure.
- Attention curves: color actual line segments with a LineCollection. Do not use floating rectangular marks or visually unrelated dot clouds. Curves should remain visibly opaque.
- Response surfaces: do not add unexplained black point overlays. Preserve scientific data; remove only optional decoration with no required evidential role.

## Workflow

1. Select the relevant entry in `references/FIGURE_CATALOG.md`, then read its data contract in `references/DATA_AND_PROVENANCE.md`.
2. Prefer `scripts/render_figures.py`, `scripts/figure_style.py`, and `style.json`. Read archived code only for historical tracing. Avoid executing archived main routines against the user's old paths.
3. Verify units, actual repetitions, and aggregate definitions before substituting data. Use the new paper's names and evidence, not the bundled battery values.
4. Adapt and run the needed figure. Default to PNG 600 dpi + PDF + SVG; add TIFF when required. Use a separate output folder for a new paper.
5. Inspect exports using the checklist in `references/QUALITY_CHECKLIST.md`, fix collisions, and provide the actual figure, source data, runnable code, and separate caption. A successful script run or zero text-box collisions alone is insufficient.

## Evidence Constraints

This is a visual style package, not a training task or a source of new research results. The real sample data are for reproducibility only. Read the provenance notes before interpreting the attention surfaces, method difference errorbars, or pooled cell-by-seed violin samples. Do not fabricate seeds, interpolate data to manufacture evidence, smooth away model failures, or optimize aesthetics by selecting favorable experiments.

These are the user's manuscript preferences, not universal Nature requirements. A new explicit request can override an aesthetic choice. Do not infer a global prohibition on bars from the teacher's request to replace a particular ablation bar figure.
