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

## 9) Most Important Lessons Learned (for the next engineer)

This section distills the highest-value implementation lessons from the AD planning and code-review pass.
If you only read one part of this document before coding, read this one.

### 9.1 Lesson 1 — Start from AD plumbing invariants, not from plots

Before implementing any new scenario, lock down the minimum AD invariants in tests:

- `compute_grads=True` must be enabled in engine config.
- Backend must be `PYTORCH` for autograd behavior expected in this plan.
- The objective scalar/tensor used for `backward(...)` must be explicitly selected and documented.
- Gradients must be asserted as:
  - not `None`
  - finite (`isfinite`)
  - reproducible enough for CI tolerance windows.

Why this matters:

- Most failures in differentiable pipelines come from graph disconnects or silent non-differentiable boundaries,
  not from plotting code.
- If these invariants are not validated first, downstream sensitivity figures are not scientifically defensible.

### 9.2 Lesson 2 — Separate “scientific objective” from “differentiable objective”

For each scenario, write two statements:

1. Scientific question (e.g., “where is the model spatially most sensitive?”)
2. Computational objective (e.g., “sum/mean/selected voxel value from scalar field for `backward`”)

Why this matters:

- Reviewers and presentation audiences care about scientific interpretation.
- Autograd requires an explicit computational scalar/tensor path.
- Mixing both in one vague objective leads to unstable or non-comparable experiments.

### 9.3 Lesson 3 — Jacobian/full-grid scenarios need strict shape bookkeeping

For Scenario B (full-grid sensitivity wrt all surface points), define shape contracts early and assert them:

- Grid target shape (`n_voxels` or `(nx, ny, nz)` flattened convention).
- Parameter shape (`n_points x 3` for coordinates).
- Jacobian shape convention (`n_outputs x n_params` or structured equivalent).

Why this matters:

- Without fixed conventions, heatmaps and ranking panels become inconsistent between runs/branches.
- Shape mismatches can silently transpose interpretation (important for paper figures).

### 9.4 Lesson 4 — Numerical-vs-autograd comparisons are mandatory for credibility

Scenario C is not optional if the output is intended for publication.

Implementation guidance:

- Use finite differences on a selected subset of points/voxels first (not full combinatorial sweep).
- Fix perturbation magnitudes and record them in the test.
- Report both:
  - absolute/relative error
  - ranking agreement of influential parameters.

Why this matters:

- AD correctness is often assumed but not demonstrated.
- A compact numerical check dramatically increases confidence for paper reviewers and conference audiences.

### 9.5 Lesson 5 — Dual contouring has two tracks: feasible-now vs research-track

Do not treat dual contouring sensitivity as one homogeneous task.

Current practical split:

- **Ready now (Tier A / Scenario E):** sensitivity through edge/intersection and gradient-related intermediate quantities.
- **Research-track (Tier B / Scenario F):** end-to-end gradient from final mesh vertices to geological inputs.

Why this matters:

- Trying to directly implement vertex-position gradients now will likely fail due to graph breaks at output conversion boundaries.
- You can still deliver meaningful sensitivity science immediately via intermediate contouring diagnostics.

### 9.6 Lesson 6 — The autograd boundary in dual contouring is the critical blocker

The most important technical blocker identified is:

- Conversion of mesh outputs to NumPy at extraction boundary breaks PyTorch graph lineage.

Consequence:

- `dc_mesh.vertices` cannot currently serve as an end-to-end differentiable target in the present API path.

Actionable implication:

- Implement a tensor-preserving output mode in engine APIs before claiming Scenario F as “ready”.

### 9.7 Lesson 7 — Deliverables are tests + figures + narrative, not tests alone

For each scenario, maintain a 3-part deliverable:

1. Test assertions (technical correctness)
2. Visualization artifact (communication)
3. Interpretation text snippet (scientific meaning)

Why this matters:

- This project output is intended for both CI and scientific dissemination (paper + EGU).
- A passing test without interpretable visualization is insufficient for the stated objective.

### 9.8 Lesson 8 — Incremental rollout is safer than broad implementation

Recommended strict order remains:

1. A: baseline regression guards
2. B: Jacobian/full-grid
3. C: numerical validation
4. D: multi-target extension
5. E: dual contouring intermediates
6. F: vertex-gradient research milestone

Why this matters:

- Each stage de-risks the next.
- Failing to complete B/C before D/E/F increases debugging ambiguity and weakens scientific traceability.

### 9.9 Lesson 9 — Keep experiment metadata explicit for reproducibility

In each test and figure generation script, log/store:

- model setup and input dataset references
- backend + dtype + `compute_grads`
- objective definition used for `backward`
- perturbation scale (for finite differences)
- random seeds (if any stochastic component is introduced)

Why this matters:

- These details are often forgotten and later required for figure regeneration and reviewer responses.

### 9.10 Lesson 10 — Define “done” per scenario before coding

For every scenario, predefine:

- technical pass criteria (assertions + thresholds)
- expected artifact(s)
- interpretation sentence template

Why this matters:

- Avoids open-ended implementation loops.
- Enables handoff between engineers without losing direction.

## 10) File Address Map (high-priority references)

Use these as primary navigation anchors while implementing.

### 10.1 In this repository (`gempy`)

- Main showcase document (this file):
  - `/home/leguark/PycharmProjects/gempy/test/test_modules/test_ad/SHOWCASE_AD.md`
- Existing AD smoke test entry point:
  - `/home/leguark/PycharmProjects/gempy/test/test_modules/test_ad/test_ad_I.py`

### 10.2 In `gempy_probability` (gradient/Jacobian patterns)

- Gradient tests used as reference style:
  - `/home/leguark/PycharmProjects/gempy_probability/tests/test_gradients/test_gradients_I.py`

### 10.3 In `gempy_engine` (autograd and dual contouring internals)

- PyTorch gradient baseline tests:
  - `/home/leguark/PycharmProjects/gempy_engine/tests/test_pytorch/test_pytorch_gradients.py`
- Dual contouring module tests:
  - `/home/leguark/PycharmProjects/gempy_engine/tests/test_common/test_modules/test_dual.py`
- Dual contouring integration tests:
  - `/home/leguark/PycharmProjects/gempy_engine/tests/test_common/test_integrations/test_multi_fields_dual_contouring.py`
- Dual contouring API path (key feasibility boundary context):
  - `/home/leguark/PycharmProjects/gempy_engine/gempy_engine/API/dual_contouring/multi_scalar_dual_contouring.py`

## 11) Practical Implementation Checklist (handoff-ready)

Copy/paste this checklist into each PR description for AD showcase work.

- [ ] Scenario ID is explicitly stated (A/B/C/D/E/F).
- [ ] Scientific objective and differentiable objective are both written.
- [ ] `compute_grads=True` and `PYTORCH` backend usage is explicit.
- [ ] Gradient existence + finiteness assertions added.
- [ ] Shape contracts asserted for outputs/parameters/Jacobian.
- [ ] If Scenario C or validation step: finite-difference comparison reported.
- [ ] Figure artifact(s) generated and linked/saved.
- [ ] Interpretation note added (what the sensitivity means geologically).
- [ ] Tier classification respected (A-ready vs B-research).
- [ ] If dual contouring vertices are targeted, graph-lineage feasibility is explicitly documented.

## 12) Open Questions to Resolve Before Scenario F

These should be resolved before allocating a full sprint to vertex-gradient sensitivity:

1. What exact API contract should expose tensor-preserving mesh outputs?
2. Should tensor-preserving mode be default or opt-in?
3. What memory/performance budget is acceptable for keeping graph lineage in mesh extraction?
4. Which vertex objective(s) are scientifically most meaningful (L2 displacement, directional projection, curvature proxy)?
5. What numerical sanity checks are mandatory before claiming end-to-end differentiability?

## 13) Recommended First PR for the next engineer

To maximize impact quickly:

- Implement Scenario A as strict regression guard assertions in
  `/home/leguark/PycharmProjects/gempy/test/test_modules/test_ad/test_ad_I.py`.
- Implement Scenario B minimal Jacobian workflow (single model, moderate grid, stable shape assertions).
- Produce one reproducible 2D sensitivity map and one ranking panel.
- Add a short result note to this markdown with observed gradient statistics.

This gives immediate scientific value while establishing the foundation for C/D/E and future F.
