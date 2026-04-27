# AD Showcase — Next Steps (verbose, junior-friendly handoff)

> **Read this first.** This document is a *step-by-step* recipe to take the current AD
> showcase from "tests pass" to "figures we are proud to show at EGU and put in a paper".
>
> The current state (`test_ad_I.py` + `showcase_ad_sphinx_gallery.py`) is functionally
> correct but **visually weak** for a scientific talk: bar charts of "influence" are
> not compelling on a 4K conference projector. We need **3D, model-overlaid,
> geologically intuitive figures** that any geologist in the audience can read in
> 5 seconds.
>
> If you are a junior engineer, follow this document **in order**, top to bottom.
> Do not skip Section 0 (definition of "scientifically compelling").

---

## 0) What "good" looks like (definition of done for the talk)

Before writing any code, internalize the visual targets. A good AD-sensitivity figure
for an EGU talk has **all** of the following:

1. **Geological context is visible.** The audience sees the actual geological model
   (lithology blocks, layer surfaces, surface points, orientations) — not a naked
   gradient array. Use `gempy_viewer` 3D / 2D plotters as the *base layer*.
2. **Sensitivity is overlaid, not isolated.** Gradients are encoded as:
   - color (diverging colormap, zero-centered) on a slice or a mesh, and/or
   - vectors/glyphs at surface points, scaled by gradient magnitude, oriented by
     gradient direction.
3. **One question per figure.** Each panel answers exactly one scientific question
   (e.g., "Which surface point most controls the position of the fold hinge?").
4. **Reproducibility is built-in.** Every figure file is paired with a small JSON
   sidecar describing backend, dtype, seed, perturbation, model, and git hash.
5. **Publication-grade styling.** 300 DPI, vector-friendly (`.pdf` + `.png`),
   consistent fonts, colorblind-safe palette (`cmocean.balance`, `RdBu_r`,
   `viridis` for magnitudes).
6. **Narrative caption.** Each figure ships with a 2–3 sentence caption that a
   geologist (not a numericist) can read.

If a generated figure does not satisfy all 6, it is **not done**.

---

## 1) Current status snapshot

- ✅ **Scenario A** (baseline AD smoke + regression guard): implemented.
- ✅ **Scenario B** (Jacobian spatial sensitivity): implemented; **visualization is
  weak** (only bar charts).
- ✅ **Scenario C** (autograd vs finite-difference check): implemented for a single
  sample.
- ✅ **Scenario D** (multi-target derivatives): implemented as numbers; **no figure**.
- ✅ **Scenario E** (dual contouring intermediate sensitivity): implemented at
  `dc_data.gradients`; **no figure**.
- ⏳ **Scenario F** (vertex-location sensitivity): pending; engine work required.

The honest gap: **we have numbers, we do not have figures**. This document fixes that.

---

## 2) Files you will touch (and where to put new ones)

Existing:

- `test/test_modules/test_ad/test_ad_I.py` — keep as **CI regression** tests only.
  Do *not* put figure code here. CI must stay fast and headless-safe.
- `test/test_modules/test_ad/showcase_ad_sphinx_gallery.py` — this is the **demo /
  figure-generation** entry point. It is the file that produces talk artifacts.

New folders/files to create:

- `test/test_modules/test_ad/figures/` — output directory for generated figures
  (`.png`, `.pdf`) and JSON sidecars. Add to `.gitignore` *or* commit small PNG
  previews; ask the lead before committing large binaries.
- `test/test_modules/test_ad/_ad_plot_utils.py` — shared plotting helpers
  (style, save-with-metadata, colorbar conventions). See Section 3.
- `test/test_modules/test_ad/_ad_model_zoo.py` — small fixtures returning
  pre-built models (fold, anticline, fault) so each scenario does not duplicate
  the model-building code.

Reference files (read-only, just to learn patterns):

- 3D plotting style example:
  `gempy_viewer/tests/test_private/test_terranigma/test_3d_colormaps.py`
  → especially `Test3DColormaps.test_3d_volume_input` shows the canonical
  `plot_3d(model, image=True, show_data=True, show_topography=False, ...)`
  call pattern. **Mirror this style** when adding 3D overlays.
- Probability-side gradient pattern:
  `gempy_probability/tests/test_gradients/test_gradients_I.py`.
- Engine autograd:
  `gempy_engine/tests/test_pytorch/test_pytorch_gradients.py`.
- Dual contouring API boundary (the autograd blocker for Scenario F):
  `gempy_engine/gempy_engine/API/dual_contouring/multi_scalar_dual_contouring.py`.

---

## 3) Environment & reproducibility scaffolding (do this first, before any plot)

Create `_ad_plot_utils.py` with at least the following utilities. **Every** figure
script must use them — non-negotiable for paper reproducibility.

### 3.1 Global style

```python
# _ad_plot_utils.py
import json
import os
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

PUBLICATION_RC = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 11,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "image.cmap": "RdBu_r",       # diverging default for signed gradients
}

def use_publication_style():
    mpl.rcParams.update(PUBLICATION_RC)
```

### 3.2 Seed & determinism

```python
def set_determinism(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed
```

### 3.3 Save with metadata sidecar

```python
@dataclass
class FigureMetadata:
    scenario: str
    backend: str
    dtype: str
    compute_grads: bool
    seed: int
    perturbation: float | None
    model_name: str
    n_surface_points: int
    n_voxels: int | None
    git_hash: str
    timestamp_utc: str
    notes: str = ""

def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(__file__),
        ).decode().strip()
    except Exception:
        return "unknown"

def save_figure(fig, name, meta: FigureMetadata, formats=("png", "pdf")):
    base = os.path.join(FIG_DIR, name)
    for ext in formats:
        fig.savefig(f"{base}.{ext}")
    with open(f"{base}.json", "w") as f:
        json.dump(asdict(meta), f, indent=2)
```

### 3.4 Colorbar convention for signed gradients

```python
def symmetric_norm(values: np.ndarray, percentile: float = 99.0):
    """Robust symmetric color limits centered on zero."""
    vmax = np.nanpercentile(np.abs(values), percentile)
    if vmax == 0:
        vmax = 1e-12
    return mpl.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
```

**Why this matters:** signed gradients with a non-zero-centered colormap are
*scientifically misleading*. Reviewers will catch this.

---

## 4) The 6 figures we want for the talk (priority-ordered)

Each figure has: **(a)** the scientific question, **(b)** the data pipeline,
**(c)** the visualization recipe, **(d)** acceptance checklist.

### Figure 1 — "Where is the model most sensitive?" (Scenario B upgrade) ⭐ headline

(a) **Question.** Across the whole 3D domain, which voxels respond most to *any*
small perturbation of *any* surface point?

(b) **Pipeline.**
1. Build the fold model (`_build_fold_model`).
2. Compute with PyTorch + `compute_grads=True`.
3. Compute the Jacobian
   `J[i, p, c] = d scalar_field[i] / d sp_coords[p, c]`. Reuse the loop already
   in `showcase_ad_sphinx_gallery.py`.
4. **New**: per-voxel total sensitivity
   `S[i] = sqrt(sum_{p,c} J[i, p, c]^2)`. This is a 3D scalar field of the
   *same shape as the regular grid* — that is the trick that makes it plottable.
5. Reshape `S` to `(nx, ny, nz)` using
   `geo_data.grid.regular_grid.resolution`.

(c) **Visualization (multi-panel, single figure).**
- Left: 3D PyVista volume of `S` overlaid with the lithology mesh
  (semi-transparent), produced via `gempy_viewer.plot_3d` + a custom volume
  actor. Use `viridis` (S is non-negative). Mirror the call style of
  `test_3d_volume_input` in `test_3d_colormaps.py`.
- Middle: 3 orthogonal slices (`XY` mid-Z, `XZ` mid-Y, `YZ` mid-X) of `S` using
  `matplotlib.pcolormesh` with the geological cross-section underneath
  (use `gpv.plot_2d(..., override_regular_grid=S_slice)` if available, else
  draw lithology contours from `final_block` on top).
- Right: top-K influential surface points highlighted as colored spheres on the
  3D model, size ∝ `sum_c |J[:, p, c]|`. Annotate top-3 with their `(x, y, z)`
  and surface name.

(d) **Acceptance.**
- [ ] Diverging-vs-magnitude colormap choice is correct (`viridis` for `S`,
      not `RdBu_r`).
- [ ] Colorbar has units and a label "‖∂φ/∂p‖ (model units / m)".
- [ ] Lithology context visible in every panel.
- [ ] Top-K points annotated with their surface name.
- [ ] Saved as PNG + PDF + JSON sidecar in
      `figures/fig01_jacobian_sensitivity_field.*`.

### Figure 2 — "Per-point gradient field" (Scenario B, point-conditioned)

(a) **Question.** If I move *this single surface point* by 1 m upward, where
does the model change?

(b) **Pipeline.** Pick `point_idx` (e.g., the top-1 from Figure 1). Slice the
Jacobian: `g[i, c] = J[i, point_idx, c]` for `c in {x, y, z}`. Reshape to grid.

(c) **Visualization.**
- 3 stacked 2D cross-sections (one per coordinate `c`) using a **diverging**
  colormap, zero-centered, symmetric range via `symmetric_norm`.
- Overlay lithology contours from the `final_block` slice.
- Mark the chosen surface point with a black star at its true `(x, y, z)`.

(d) **Acceptance.**
- [ ] Diverging colormap, zero-centered.
- [ ] The chosen point is visually identifiable.
- [ ] Caption explains "blue = scalar field decreases when we move the point in +c".

### Figure 3 — "Autograd is correct" (Scenario C upgrade)

(a) **Question.** Are AD gradients quantitatively trustworthy?

(b) **Pipeline.** Currently the test compares 1 sample. Extend to N≈30 random
`(point_idx, coord_idx, voxel_idx)` triplets with a *fixed* seed. For each:
- AD gradient `ad`
- Central FD gradient `num` with `epsilon ∈ {1e-2, 1e-3, 1e-4}` (sweep!)
- Record `(ad, num, epsilon)`.

(c) **Visualization (3-panel parity figure).**
- Left: parity scatter `ad` vs `num` with `y=x` reference line, log–log,
  colored by `epsilon`. This is the *money plot* for reviewer confidence.
- Middle: histogram of relative errors per `epsilon`.
- Right: ranking-agreement plot — Spearman correlation between `argsort(|ad|)`
  and `argsort(|num|)` over the N samples.

(d) **Acceptance.**
- [ ] Median relative error reported in the caption (target < 1e-3 at best
      `epsilon`).
- [ ] Outliers (rel_err > 1e-1) are listed in the JSON sidecar with their indices.
- [ ] At least 3 perturbation magnitudes shown.

### Figure 4 — "Different observables, different sensitivities" (Scenario D upgrade)

(a) **Question.** Does sensitivity depend on the geological observable
(scalar field vs lithology block vs hypothetical gravity)?

(b) **Pipeline.** Reuse code in `showcase_ad_sphinx_gallery.py` Scenario D.
Add a 3rd target: a *crude gravity surrogate* `g = sum(block * z_weight)` where
`z_weight` is the depth column. This keeps autograd intact and gives a third
interpretable observable without adding the geophysics package.

(c) **Visualization.** Single figure with 3 rows × `n_points` columns of small
heatmaps, one per `(target, point)` showing the per-coordinate gradient as a
3-cell strip. Add a final summary row showing cosine-similarity matrix between
the three targets.

(d) **Acceptance.**
- [ ] Cosine similarity matrix is shown and annotated with values.
- [ ] All three targets share a common colorbar scale (after per-target
      normalization).

### Figure 5 — "Surface-extraction sensitivity" (Scenario E upgrade)

(a) **Question.** How sensitive is the *extracted surface mesh* to surface points,
**without** breaking through the dual-contouring NumPy boundary?

(b) **Pipeline.**
1. Get `mesh_e.dc_data.gradients` (already a tensor in the AD path).
2. Compute `d ||grad_dc||_2 / d sp_coords` (already implemented; reuse).
3. **New**: project `dc_data.gradients` onto the *post-DC* mesh vertices by
   nearest-neighbor lookup from the regular grid edge centers to vertex
   positions. This gives a vertex-attached scalar that is *not autograd-bound*
   to vertices, but is *visually anchored on the mesh*.

(c) **Visualization.** PyVista 3D plot of the extracted surface, colored by the
projected sensitivity, with vector glyphs at surface points showing
`d(target)/d(sp)` as arrows.

(d) **Acceptance.**
- [ ] Mesh is the visual anchor (audience immediately sees "the geological surface").
- [ ] Caption clearly states "intermediate sensitivity (Tier A); see §8 of
      `SHOWCASE_AD.md` for vertex-position sensitivity (Tier B / research)".

### Figure 6 — "Optimization narrative" (new, killer slide)

(a) **Question.** Can we *use* AD to do something useful — e.g., move a single
surface point so the fold hinge passes through a target location?

(b) **Pipeline.**
1. Define a target voxel `i*` and target value `v*` for the scalar field
   (or a target Z for the iso-surface).
2. Define loss `L = (scalar_field[i*] - v*)^2`.
3. Run 20 steps of `torch.optim.Adam` on `sp_coords` (small learning rate;
   step the chosen surface point only — clone & freeze others).
4. Recompute the model every step.
5. Snapshot the model at steps `{0, 5, 10, 20}`.

(c) **Visualization.** 4-panel storyboard (one per snapshot) showing the
lithology cross-section + the moving surface point as a red dot + the loss
curve in an inset.

(d) **Acceptance.**
- [ ] Loss decreases monotonically (or near-monotonically); document otherwise.
- [ ] The animation/storyboard makes the message obvious without narration.
- [ ] **This is the slide that justifies AD in GemPy.** Treat it as the talk's
      climax.

---

## 5) Concrete implementation order (junior dev, ~5 working days)

**Day 1 — scaffolding (no figures yet).**
- [ ] Create `_ad_plot_utils.py` (Section 3).
- [ ] Create `_ad_model_zoo.py` with `build_fold_model()` returning `(geo_data, meta)`.
- [ ] Add `figures/` folder; add `*.png` and `*.pdf` to `.gitignore` if the team
      agrees.
- [ ] Confirm the existing 5 scenarios still pass:
      `pytest test/test_modules/test_ad -x`.

**Day 2 — Figure 1 (the headline figure).**
- [ ] Compute the per-voxel `S[i]` field. Verify shape equals the regular-grid
      resolution.
- [ ] First, get the 2D slice version working (easier).
- [ ] Then layer the 3D PyVista version on top, mirroring `test_3d_volume_input`.
- [ ] Iterate on color limits using `symmetric_norm` / percentile clipping.

**Day 3 — Figures 2 & 3.**
- [ ] Figure 2 reuses Figure 1's plumbing; only the slicing changes.
- [ ] Figure 3 needs a small loop running `gp.compute_model` 2N times for FD;
      cache results because each call is slow.

**Day 4 — Figures 4 & 5.**
- [ ] Figure 4 is mostly matplotlib; cosine-similarity is one
      `torch.nn.functional.cosine_similarity` call.
- [ ] Figure 5 needs `scipy.spatial.cKDTree` to map edge centers → vertices.

**Day 5 — Figure 6 (the climax) + polish.**
- [ ] Implement the Adam-on-surface-point loop. **Critical:** call
      `gp.compute_model` *inside* the loop and re-extract the scalar field
      tensor each step; do not cache.
- [ ] Make a 4-panel storyboard, then optionally a `matplotlib.animation`.
- [ ] Final pass: every figure has a JSON sidecar; every figure has a caption
      stored in `figures/captions.md`.

---

## 6) Common pitfalls (read these — each saves you a day)

1. **Calling `backward` twice without `retain_graph=True`.** The current code
   uses `retain_graph=True`; keep it. Otherwise the second call dies with
   "Trying to backward through the graph a second time".
2. **Forgetting to `zero_()` `sp_coords.grad`** before each per-voxel
   `backward`. The Jacobian loop in `showcase_ad_sphinx_gallery.py` already
   does this — copy the pattern.
3. **Comparing AD gradients (in *normalized* coordinates) vs FD gradients (in
   *real* coordinates).** The current Scenario C already multiplies by
   `geo_c.input_transform.scale[coord_idx]`. **Do not remove this.** It is the
   single most common AD-vs-FD mismatch in this codebase.
4. **Using `RdBu_r` for non-negative quantities** (e.g., `‖J‖`). Use `viridis`
   or `cmocean.thermal`. Diverging colormaps imply sign.
5. **Plotting gradients without geological context.** A bar chart of "point
   index vs influence" is *not* a scientific figure for a geology audience.
   Always put the model in the picture.
6. **Mesh vertices look differentiable but are not.** `dc_mesh.vertices` is a
   NumPy array. Calling `.requires_grad_()` on a recovered torch tensor will
   *not* reconnect it to `sp_coords`. See `SHOWCASE_AD.md` §8.2.
7. **Slow runtime in CI.** Keep `MAX_VOXELS_FOR_JACOBIAN = 350` for CI; bump to
   ≥ 4000 only in the figure-generation script (and document the runtime in the
   JSON sidecar).
8. **Running figure code in `test_ad_I.py`.** Don't. CI must stay headless and
   fast. Keep figures in `showcase_ad_sphinx_gallery.py` (or a sibling script).

---

## 7) Reproducibility & metadata (must be done for the paper)

For each saved figure (`figures/figXX_*.png`), write a sibling
`figures/figXX_*.json` with the `FigureMetadata` shown in §3.3. Reviewers
*will* ask for this. The git hash + perturbation + seed lets us regenerate
any figure exactly.

Also produce a single `figures/MANIFEST.md` listing every figure, its caption,
its inputs, and the command that produced it. This is what goes into the
supplementary materials of the paper.

---

## 8) Scenario F (research-track) — unchanged but unblocked by figures above

Figures 1–6 do **not** depend on Scenario F. Ship them first, then revisit F.

Prerequisites for end-to-end `dc_mesh.vertices` autograd, in order:

1. Tensor-preserving output mode in `multi_scalar_dual_contouring.py`
   (avoid `.numpy()` / `.cpu().numpy()` at the extraction boundary).
2. Public toggle (e.g., `InterpolationOptions.tensor_preserving_dc=True`),
   default off, opt-in for AD users only.
3. Memory/perf benchmark vs current path on the fold model and a fault model.
4. Define the vertex objective: L2 displacement against a reference mesh,
   directional projection on a target normal, or curvature proxy.
5. Integration test in `gempy_engine` proving `vertex.backward()` reaches
   `sp_coords.grad` with non-zero, finite values.
6. Replicate Figures 5–6 with vertex-level gradients; this becomes the paper's
   "future work realized" figure.

---

## 9) Definition of done for this whole track

- [ ] All Scenario A–E assertions pass in CI (`test_ad_I.py`).
- [ ] `showcase_ad_sphinx_gallery.py` produces Figures 1–6 deterministically.
- [ ] Each figure has a PNG + PDF + JSON sidecar in `figures/`.
- [ ] `figures/MANIFEST.md` exists and is up to date.
- [ ] Each figure caption is reviewed by a domain geologist (≠ the implementer).
- [ ] Scenario F is either implemented (with Figures 5–6 upgraded) or has a
      written, approved engine-side implementation plan.

---

## 10) Pointers for the next person to ask the lead

If any of the following is unclear, **ask before coding** — getting it wrong
costs more than a half-day call:

1. Is the fold model the right "hero" model for the talk, or should we use a
   fault model (more visually compelling for sensitivity)?
2. Is committing PNG/PDF figure binaries acceptable, or should we generate them
   in CI artifacts only?
3. Do we have a license-compatible colormap preference (`cmocean`,
   `cmcrameri`)? The latter is widely accepted in geosciences.
4. Is `gempy_viewer.plot_3d` happy in headless CI (Xvfb), or should figure
   generation be marked as an opt-in script (recommended)?
5. For Figure 6, what is the scientifically *correct* objective to optimize for
   the EGU narrative? (Suggestion: "move surface point so layer crosses a known
   borehole observation".)

Document the answers at the top of `showcase_ad_sphinx_gallery.py` so the next
engineer after you does not have to re-ask.
