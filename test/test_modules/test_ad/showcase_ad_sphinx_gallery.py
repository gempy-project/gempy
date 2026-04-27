"""
Automatic Differentiation Sensitivity Showcase (GemPy 3)
=========================================================

This Sphinx-Gallery style script mirrors the AD showcase content currently implemented in
``test_ad_I.py`` and reorganizes it as a publication/demo narrative.

It focuses on:

1. **Scenario A**: baseline AD smoke + gradient quality checks.
2. **Scenario B**: Jacobian-based spatial sensitivity and parameter influence ranking.
3. **Scenario C**: numerical-vs-autograd gradient comparison for scientific credibility.
4. **Scenario D**: multi-target derivatives (scalar field vs block surrogate).
5. **Scenario E**: dual contouring intermediate sensitivity.

Research-track **Scenario F** (end-to-end vertex location sensitivity) is not included here,
as it depends on tensor-lineage-preserving outputs in the dual contouring pipeline.
"""

# %%
# Imports and runtime controls
# ----------------------------

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

import gempy as gp
from gempy.optional_dependencies import require_gempy_viewer

os.environ.setdefault("DEFAULT_BACKEND", "PYTORCH")

PLOT_3D = False
MAX_VOXELS_FOR_JACOBIAN = 350


# %%
# Helper utilities
# ----------------

def _build_fold_model(project_name: str) -> gp.data.GeoModel:
    data_path = "https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/"
    path_to_data = data_path + "/data/input_data/jan_models/"

    geo_data: gp.data.GeoModel = gp.create_geomodel(
        project_name=project_name,
        extent=[0, 1000, 0, 1000, 0, 1000],
        refinement=3,
        importer_helper=gp.data.ImporterHelper(
            path_to_orientations=path_to_data + "model2_orientations.csv",
            path_to_surface_points=path_to_data + "model2_surface_points.csv",
        ),
    )

    gp.map_stack_to_surfaces(
        gempy_model=geo_data,
        mapping_object={"Strat_Series": ("rock2", "rock1")},
    )
    return geo_data


def _compute_with_ad(geo_data: gp.data.GeoModel) -> None:
    gp.compute_model(
        gempy_model=geo_data,
        engine_config=gp.data.GemPyEngineConfig(
            backend=gp.data.AvailableBackends.PYTORCH,
            use_gpu=False,
            dtype="float64",
            compute_grads=True,
        ),
    )


def _get_scalar_field(geo_data: gp.data.GeoModel):
    scalar_field = geo_data.solutions.octrees_output[0].last_output_center.exported_fields.scalar_field
    return scalar_field[0] if isinstance(scalar_field, list) else scalar_field


def _beautiful_sensitivity_plot(jacobian: torch.Tensor, save_path: str) -> None:
    point_influence = torch.sum(torch.abs(jacobian), dim=(0, 2)).detach().numpy()
    coord_influence = torch.sum(torch.abs(jacobian), dim=(0, 1)).detach().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(np.arange(point_influence.shape[0]), point_influence, color="#5B8FF9", alpha=0.9)
    axes[0].set_title("Surface-point influence ranking")
    axes[0].set_xlabel("Surface point index")
    axes[0].set_ylabel("$\\sum |\\partial f / \\partial p|$")
    axes[0].grid(alpha=0.25)

    coord_labels = ["X", "Y", "Z"]
    axes[1].bar(coord_labels, coord_influence, color=["#61DDAA", "#65789B", "#F6BD16"], alpha=0.9)
    axes[1].set_title("Coordinate-wise sensitivity budget")
    axes[1].set_ylabel("$\\sum |\\partial f / \\partial c|$")
    axes[1].grid(alpha=0.25)

    fig.suptitle("Scenario B — Jacobian sensitivity summary", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


# %%
# Scenario A — Baseline AD smoke and gradient quality
# ----------------------------------------------------

geo_a = _build_fold_model("fold_gallery_scenario_a")
_compute_with_ad(geo_a)

target_a = _get_scalar_field(geo_a)[0]
target_a.backward(retain_graph=True, create_graph=True)

sp_grad_a = geo_a.taped_interpolation_input.surface_points.sp_coords.grad
orient_grad_a = geo_a.taped_interpolation_input.orientations.dip_positions.grad

assert sp_grad_a is not None, "Surface points gradients should not be None"
assert orient_grad_a is not None, "Orientation gradients should not be None"
assert torch.isfinite(sp_grad_a).all(), "Surface points gradients should be finite"
assert torch.isfinite(orient_grad_a).all(), "Orientation gradients should be finite"

grad_norm_a = torch.norm(sp_grad_a).item()
assert 0.0 < grad_norm_a < 1e6, f"Gradient norm {grad_norm_a} out of expected range"
print(f"Scenario A | ||grad(surface_points)|| = {grad_norm_a:.6e}")

if PLOT_3D:
    gpv = require_gempy_viewer()
    gpv.plot_3d(geo_a, image=True)


# %%
# Scenario B — Jacobian spatial sensitivity and richer visual summaries
# ---------------------------------------------------------------------

geo_b = _build_fold_model("fold_gallery_scenario_b")
_compute_with_ad(geo_b)

scalar_field_b = _get_scalar_field(geo_b)
sp_coords_b = geo_b.taped_interpolation_input.surface_points.sp_coords

n_voxels_full = scalar_field_b.shape[0]
n_voxels = min(n_voxels_full, MAX_VOXELS_FOR_JACOBIAN)
n_points = sp_coords_b.shape[0]

print(f"Scenario B | Computing Jacobian for {n_voxels}/{n_voxels_full} voxels and {n_points} points")

jacobian_b = torch.zeros((n_voxels, n_points, 3), dtype=torch.float64)

for i in range(n_voxels):
    if sp_coords_b.grad is not None:
        sp_coords_b.grad.zero_()
    scalar_field_b[i].backward(retain_graph=True)
    jacobian_b[i] = sp_coords_b.grad

assert jacobian_b.shape == (n_voxels, n_points, 3)
assert torch.isfinite(jacobian_b).all(), "Jacobian contains non-finite values"

density_b = (torch.count_nonzero(jacobian_b) / jacobian_b.numel()).item()
point_influence_b = torch.sum(torch.abs(jacobian_b), dim=(0, 2))
top_k = min(5, n_points)
top_k_idx = torch.topk(point_influence_b, k=top_k).indices.tolist()

print(f"Scenario B | Jacobian density = {density_b:.4f}")
print(f"Scenario B | Top-{top_k} influential points = {top_k_idx}")
assert point_influence_b.max() > 0

_beautiful_sensitivity_plot(
    jacobian=jacobian_b,
    save_path="test/test_modules/test_ad/scenario_b_sensitivity_dashboard.png",
)


# %%
# Scenario C — Numerical vs autograd consistency check
# -----------------------------------------------------

geo_c = _build_fold_model("fold_gallery_scenario_c")
_compute_with_ad(geo_c)

scalar_field_c = _get_scalar_field(geo_c)
sp_coords_c = geo_c.taped_interpolation_input.surface_points.sp_coords

point_idx = 0
coord_idx = 2
voxel_idx = 0
epsilon = 1e-3

if sp_coords_c.grad is not None:
    sp_coords_c.grad.zero_()
scalar_field_c[voxel_idx].backward(retain_graph=True)
ad_grad = sp_coords_c.grad[point_idx, coord_idx].item()

original_z = geo_c.surface_points_copy.df.iloc[point_idx]["Z"]

gp.modify_surface_points(geo_c, slice=point_idx, Z=original_z + epsilon)
sol_p = gp.compute_model(
    geo_c,
    engine_config=gp.data.GemPyEngineConfig(
        backend=gp.data.AvailableBackends.numpy,
        compute_grads=False,
    ),
)
val_p = sol_p.octrees_output[0].last_output_center.exported_fields.scalar_field[voxel_idx].item()

gp.modify_surface_points(geo_c, slice=point_idx, Z=original_z - epsilon)
sol_n = gp.compute_model(
    geo_c,
    engine_config=gp.data.GemPyEngineConfig(
        backend=gp.data.AvailableBackends.numpy,
        compute_grads=False,
    ),
)
val_n = sol_n.octrees_output[0].last_output_center.exported_fields.scalar_field[voxel_idx].item()

num_grad = (val_p - val_n) / (2 * epsilon)
scale = geo_c.input_transform.scale[coord_idx]
ad_grad_real = ad_grad * scale

abs_error = abs(ad_grad_real - num_grad)
rel_error = abs_error / (abs(ad_grad_real) + 1e-10)

print(f"Scenario C | AD(real)={ad_grad_real:.6e}, Num={num_grad:.6e}, abs={abs_error:.3e}, rel={rel_error:.3e}")
assert rel_error < 1e-2 or abs_error < 1e-2

gp.modify_surface_points(geo_c, slice=point_idx, Z=original_z)


# %%
# Scenario D — Multi-target derivative comparison
# ------------------------------------------------

geo_d = _build_fold_model("fold_gallery_scenario_d")
geo_d.interpolation_options.sigmoid_slope = 100
_compute_with_ad(geo_d)

scalar_field_d = _get_scalar_field(geo_d)
block_d = geo_d.solutions.octrees_output[0].last_output_center.block
sp_coords_d = geo_d.taped_interpolation_input.surface_points.sp_coords

if sp_coords_d.grad is not None:
    sp_coords_d.grad.zero_()
scalar_field_d.mean().backward(retain_graph=True)
grad_scalar_d = sp_coords_d.grad.clone()

if sp_coords_d.grad is not None:
    sp_coords_d.grad.zero_()
block_d.mean().backward(retain_graph=True)
grad_block_d = sp_coords_d.grad.clone()

assert torch.isfinite(grad_scalar_d).all()
assert torch.isfinite(grad_block_d).all()

cos_sim = torch.nn.functional.cosine_similarity(grad_scalar_d.flatten(), grad_block_d.flatten(), dim=0).item()
print(f"Scenario D | ||grad_scalar||={torch.norm(grad_scalar_d).item():.6e}")
print(f"Scenario D | ||grad_block|| ={torch.norm(grad_block_d).item():.6e}")
print(f"Scenario D | cosine_similarity={cos_sim:.6f}")
assert cos_sim < 1.0


# %%
# Scenario E — Dual contouring intermediate sensitivity
# ------------------------------------------------------

geo_e = _build_fold_model("fold_gallery_scenario_e")
_compute_with_ad(geo_e)

mesh_e = geo_e.solutions.dc_meshes[0]
dc_gradients_e = mesh_e.dc_data.gradients
target_e = dc_gradients_e.mean()

sp_coords_e = geo_e.taped_interpolation_input.surface_points.sp_coords
if sp_coords_e.grad is not None:
    sp_coords_e.grad.zero_()
target_e.backward(retain_graph=True)
grad_dc_e = sp_coords_e.grad

assert grad_dc_e is not None
assert torch.isfinite(grad_dc_e).all()
assert torch.norm(grad_dc_e).item() > 0
print(f"Scenario E | ||grad_dc||={torch.norm(grad_dc_e).item():.6e}")


# %%
# Final notes
# -----------
# - A richer visual artifact is written to:
#   ``test/test_modules/test_ad/scenario_b_sensitivity_dashboard.png``
# - This script is designed as a demonstrator for Sphinx-Gallery narrative flow.
# - For pending work (especially Scenario F), see ``NEXT_STEPS.md`` in this folder.

# sphinx_gallery_thumbnail_number = 2
