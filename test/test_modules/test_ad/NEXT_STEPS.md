# AD Showcase — Next Steps

This checklist summarizes what is still left after the current implementation in:

- `test/test_modules/test_ad/test_ad_I.py`
- `test/test_modules/test_ad/showcase_ad_sphinx_gallery.py`

## Current status snapshot

- ✅ **Scenario A** (baseline AD smoke + regression guard): implemented.
- ✅ **Scenario B** (Jacobian spatial sensitivity): implemented with shape checks and influence ranking.
- ✅ **Scenario C** (autograd vs finite-difference check): implemented for a selected point/voxel.
- ✅ **Scenario D** (multi-target derivatives): implemented (scalar field and differentiable block surrogate).
- ✅ **Scenario E** (dual contouring intermediate sensitivity): implemented at `dc_data.gradients` level.
- ⏳ **Scenario F** (vertex-location sensitivity): pending; requires engine-side tensor-lineage-preserving outputs.

## High-priority remaining work

1. **Harden reproducibility for paper/demo artifacts**
   - Fix random seeds where relevant.
   - Record runtime metadata with each generated figure:
     - backend, dtype, `compute_grads`, perturbation size, model setup.
   - Save all figures with deterministic names and paths.

2. **Upgrade Scenario B visual outputs**
   - Add spatial sensitivity maps (slice-wise) in addition to global ranking bars.
   - Add top-k point annotations directly on plots.
   - Export figure set for publication quality (high DPI + consistent style).

3. **Expand Scenario C validation coverage**
   - Compare autograd vs numerical gradients on multiple `(point, coord, voxel)` samples.
   - Report both:
     - absolute and relative errors
     - ranking agreement of influential parameters.
   - Define and document acceptance thresholds for CI vs publication runs.

4. **Scenario D dashboarding**
   - Add a compact multi-panel figure:
     - norm comparison
     - cosine similarity
     - per-point target sensitivity differences.
   - Introduce optional geophysics target if available in current setup.

5. **Scenario E interpretability improvements**
   - Add 3D overlays linking dual-contouring intermediate sensitivity to surface geometry.
   - Validate stability around high-curvature or topology-change zones.

## Scenario F (research-track) prerequisites

Before implementing end-to-end `dc_mesh.vertices` gradients:

1. Add/agree on an API mode that preserves tensor outputs through dual contouring.
2. Avoid NumPy conversion at extraction boundaries for selected outputs.
3. Benchmark memory/performance impact of lineage-preserving mode.
4. Define scientifically meaningful vertex objectives (e.g., displacement-based metrics).
5. Add integration tests proving autograd continuity from vertex objectives to surface-point coordinates.

## Recommended execution order

1. Reproducibility hardening and metadata logging.
2. Scenario B + D publication-quality visualization upgrades.
3. Scenario C multi-sample numerical validation.
4. Scenario E 3D interpretability panels.
5. Scenario F engine prototype and integration tests.

## Done criteria for this track

- All Scenario A–E assertions pass reliably.
- Gallery script generates reproducible figure artifacts.
- Numerical validation includes representative multi-sample coverage.
- Open technical boundary for Scenario F is either:
  - resolved in engine APIs, or
  - explicitly documented with an approved implementation plan.
