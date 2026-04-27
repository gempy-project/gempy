# GemPy 3 AD Showcase Test Plan (Paper + EGU Demo)

## 1) Scientific Purpose and Audience

This plan defines an exhaustive and publication-ready showcase of automatic differentiation (AD)
capabilities in GemPy 3 for:

- **Scientific paper figures** (method validation + sensitivity interpretation).
- **EGU presentation/demo** (clear visual narratives for differentiable geological modeling).

Primary objective: demonstrate gradients from geological outputs back to geological input parameters,
with increasing complexity from AD smoke tests to surface-location sensitivity analyses.

## 2) Scope and Test Tiers

- **Tier A — Ready to implement now:** scenarios that can be built with current GemPy/GemPy Engine APIs.
- **Tier B — Research track:** scenarios requiring engine-side changes (tensor lineage preservation
  through dual contouring mesh outputs).

## 3) Current Baseline (Existing AD Smoke Path)

Current baseline lives in `test_generate_fold_model` in this file:

- Uses `GemPyEngineConfig(..., backend=PYTORCH, compute_grads=True)`.
- Computes model and calls `backward(...)` on a scalar-field target from
  `geo_data.solutions.octrees_output[0].last_output_center.exported_fields.scalar_field[0]`.
- Confirms proof-of-life AD flow from model outputs back to taped interpolation inputs.

Reference context in neighboring projects:

- `gempy_probability/tests/test_gradients/test_gradients_I.py`
- `gempy_engine/tests/test_pytorch/test_pytorch_gradients.py`
- `gempy_engine/tests/test_common/test_modules/test_dual.py`
- `gempy_engine/tests/test_common/test_integrations/test_multi_fields_dual_contouring.py`

## 4) Exhaustive Showcase Matrix (Planned Scenarios)

| Scenario ID | Tier | Scientific objective | Model/data setup | Differentiable target | Variables to differentiate | Metrics/assertions | Visualization output | Expected publication value |
|---|---|---|---|---|---|---|---|---|
| A. Baseline AD smoke + regression guard | A | Verify stable AD plumbing in GemPy 3 fold path | Current `test_generate_fold_model` fold model + regular grid | `scalar_field[0]` from `octrees_output[0].last_output_center.exported_fields` | All `surface_points` tensors (and optionally orientations) | Non-null gradients, finite values, deterministic gradient norm ranges | Simple table + one 3D model view (`gpv.plot_3d`) | Methods baseline figure and reproducibility statement |
| B. Jacobian over full grid wrt all surface-point coordinates | A | Quantify spatial sensitivity distribution over model domain | Fold model at moderate grid resolution; follow `gempy_probability` Jacobian pattern | Grid-wide scalar field / block (`final_block`-style target) | Every surface-point coordinate `(x, y, z)` | Jacobian shape checks, sparsity/density summary, top-k influential points | 2D gradient heatmaps via `gpv.plot_2d(..., override_regular_grid=...)`; sensitivity bar charts | Core methodological figure: "where model is most sensitive" |
| C. Numerical-vs-autograd gradient comparison | A | Demonstrate scientific correctness of AD derivatives | Same model as B, selected points/voxels for local checks | Scalar outputs at selected grid cells or aggregated objective | Chosen surface-point coordinates; finite-difference perturbations | Relative/absolute error thresholds, ranking agreement of influential parameters | Side-by-side error maps + parity plot (`autograd` vs `finite diff`) | Validation figure for reviewer confidence |
| D. Multi-target derivatives (scalar field + lithology block + optional geophysics objective) | A | Show AD supports multiple geological observables | Reuse fold model and optional geophysics-enabled setup when available | Scalar field, lithology probabilities/labels surrogate, optional gravity/magnetic objective | Surface points + optional nuisance parameters | Cross-target consistency trends, gradient norm decomposition by target | Multi-panel sensitivity dashboard (one panel per target) | Demonstrates breadth of differentiable outputs |
| E. Dual contouring edge/normal sensitivity | A | Connect AD to isosurface extraction intermediates | Engine dual contouring integration pattern from `test_multi_fields_dual_contouring` | Edge scalar gradients / intersection-related quantities in contouring path | Surface points affecting local scalar topology | Finite + stable sensitivities around extracted structures; perturbation response consistency | 3D mesh overlay with local sensitivity vectors/colors | Bridge figure from interpolation field to extracted surfaces |
| F. Dual contouring vertex location gradient (surface location sensitivity) | B | End-to-end derivative of mesh vertex positions wrt geological inputs | Requires preserving tensor graph in DC mesh outputs | Final `dc_mesh.vertices` (or equivalent tensorized vertex output) | Surface-point coordinates and optional orientation controls | Autograd path continuity tests, vertex displacement derivative sanity checks | 3D vertex sensitivity glyphs and response animations | Forward-looking figure for next engine milestone |

## 5) Visualization Catalog

- **2D gradient maps:** `gpv.plot_2d(..., override_regular_grid=...)` using per-parameter gradient slices.
- **Sensitivity ranking panels:** bar/violin plots of gradient magnitudes by point/parameter, in the style of
  `gempy_probability/tests/test_gradients/test_gradients_I.py`.
- **Autograd-vs-numerical validation plots:** parity scatter + error histogram + spatial error maps.
- **3D views for talks:** PyVista-based mesh and vector overlays (leveraging existing plotting helpers in
  `gempy_engine.plugins.plotting.helper_functions_pyvista` when applicable).

## 6) Initial Implementation Order and Effort

1. **A (0.5 day):** convert current smoke path into explicit regression guard assertions.
2. **B (1-2 days):** implement full-grid Jacobian-style workflow and first figure set.
3. **C (1 day):** add finite-difference comparison on selected points for numerical credibility.
4. **D (1 day):** extend to multi-target derivatives and consolidated visualization panels.
5. **E (1-2 days):** implement dual contouring intermediate sensitivity diagnostics.
6. **F (research milestone):** engine-level changes, then proof-of-concept tests/figures.

## 7) Notes and Clarifications

- This document is a test-and-visualization blueprint; it does not implement all scenarios now.
- Scenarios A-E are intended as implementation-ready using current API boundaries.
- Scenario F is research-track and depends on engine changes to keep tensor lineage on mesh outputs.

## 8) Dual Contouring Sensitivity Track (Feasibility Boundary + Phased Plan)

### 8.1 Current feasible path (ready now)

- In `gempy_engine/API/dual_contouring/multi_scalar_dual_contouring.py`, dual contouring uses
  scalar-field-derived gradients on edges/intersections during extraction.
- Therefore, sensitivity analyses based on edge/intersection quantities are feasible now and can be
  demonstrated as Tier A (Scenario E) without changing AD semantics.

### 8.2 Current autograd boundary (research constraint)

- The same dual contouring path currently converts mesh arrays (including vertices/edges outputs)
  to NumPy at the extraction boundary before returning.
- This conversion breaks end-to-end PyTorch autograd lineage from final mesh vertices back to
  interpolation inputs.
- Consequence: direct derivatives of final `dc_mesh.vertices` wrt surface points are not currently
  a ready-to-implement test in this repository and remain Tier B research track (Scenario F).

### 8.3 Phased implementation proposal for surface-location sensitivity

1. **Phase 1 (now):** implement edge/intersection sensitivity diagnostics and visual overlays
   to show where extracted surfaces are most responsive to input perturbations.
2. **Phase 2 (engine prototype):** add a tensor-preserving output mode in dual contouring APIs
   (defer/avoid NumPy conversion for selected outputs).
3. **Phase 3 (integration test):** implement a new test that backpropagates from vertex-position
   objectives to surface-point coordinates; include stability and sanity constraints.
4. **Phase 4 (publication figure):** generate vertex displacement sensitivity panels and
   comparative plots against Phase-1 edge sensitivities.
