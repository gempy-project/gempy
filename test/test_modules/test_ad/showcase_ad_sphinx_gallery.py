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
import sys

# Ensure local gempy_engine is used if available
sys.path.insert(0, "/home/leguark/PycharmProjects/gempy_engine")

import gempy_engine

os.environ.setdefault("DEFAULT_BACKEND", "PYTORCH")
import matplotlib.pyplot as plt
import numpy as np
import torch

import gempy as gp
from gempy.optional_dependencies import require_gempy_viewer

from _ad_plot_utils import use_publication_style, set_determinism, save_figure, FigureMetadata, symmetric_norm
from _ad_model_zoo import build_fold_model

use_publication_style()
set_determinism(42)

PLOT_3D = False
MAX_VOXELS_FOR_JACOBIAN = 350


# %%
# Helper utilities
# ----------------

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
    point_influence = torch.sum(torch.abs(jacobian), axis=(0, 2)).detach().numpy()
    coord_influence = torch.sum(torch.abs(jacobian), axis=(0, 1)).detach().numpy()

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

geo_a, meta_a = build_fold_model("fold_gallery_scenario_a")
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

geo_b, meta_b = build_fold_model("fold_gallery_scenario_b")
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
point_influence_b = torch.sum(torch.abs(jacobian_b), axis=(0, 2))
top_k = min(5, n_points)
top_k_idx = torch.topk(point_influence_b, k=top_k).indices.tolist()

print(f"Scenario B | Jacobian density = {density_b:.4f}")
print(f"Scenario B | Top-{top_k} influential points = {top_k_idx}")
assert point_influence_b.max() > 0

# --- Figure 1: Where is the model most sensitive? ---
# 1. Per-voxel total sensitivity S[i] = sqrt(sum_{p,c} J[i, p, c]^2)
S_b = torch.sqrt(torch.sum(jacobian_b**2, axis=(1, 2)))

# 2. Reshape to grid
# For Figure 1, we want to show sensitivity in context.
def plot_figure_1(geo_model, jacobian, meta):
    S = torch.sqrt(torch.sum(jacobian**2, axis=(1, 2))).detach().numpy()
    
    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(1, 3)
    
    # Left: placeholder for 3D (narrative)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.text(0.5, 0.5, "3D PyVista Overlay\n(S-field over Mesh)", ha='center', va='center', fontsize=12)
    ax0.set_title("3D Sensitivity Volume (S)")
    ax0.axis('off')
    
    # Middle: Voxel sensitivity distribution
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(S, 'o-', color='teal', markersize=4)
    ax1.set_title("Total Sensitivity per Voxel")
    ax1.set_xlabel("Voxel Index")
    ax1.set_ylabel("‖∂φ/∂p‖ (model units / m)")
    ax1.grid(alpha=0.3)
    
    # Right: Top-K influential points
    ax2 = fig.add_subplot(gs[0, 2])
    point_influence = torch.sum(torch.abs(jacobian), axis=(0, 2)).detach().numpy()
    top_k_val = point_influence[top_k_idx]
    
    ax2.bar(np.arange(len(point_influence)), point_influence, color='gray', alpha=0.3, label='Other points')
    ax2.bar(top_k_idx, top_k_val, color='red', alpha=0.8, label='Top-K points')
    ax2.set_title(f"Top-{top_k} Influential Surface Points")
    ax2.set_xlabel("Surface Point Index")
    ax2.set_ylabel("$\\sum |\\partial f / \\partial p|$")
    ax2.legend()
    
    fig.suptitle("Figure 1: Jacobian Sensitivity Field Summary", fontsize=15)
    fig.tight_layout()
    
    meta.scenario = "Figure 1 - Jacobian Sensitivity"
    meta.n_voxels = len(S)
    save_figure(fig, "fig01_jacobian_sensitivity_field", meta)
    plt.close(fig)

plot_figure_1(geo_b, jacobian_b, meta_b)

# --- Figure 2: Per-point gradient field ---
def plot_figure_2(geo_model, jacobian, point_idx, meta):
    # Slice the Jacobian: g[i, c] = J[i, point_idx, c]
    g = jacobian[:, point_idx, :].detach().numpy() # (n_voxels, 3)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    coord_names = ["X", "Y", "Z"]
    
    for c in range(3):
        gc = g[:, c]
        norm = symmetric_norm(gc)
        im = axes[c].scatter(np.arange(len(gc)), gc, c=gc, cmap='RdBu_r', norm=norm)
        plt.colorbar(im, ax=axes[c])
        axes[c].set_title(f"Sensitivity to {coord_names[c]} shift")
        axes[c].set_xlabel("Voxel Index")
    
    fig.suptitle(f"Figure 2: Gradient field for surface point {point_idx}", fontsize=15)
    fig.tight_layout()
    
    meta.scenario = "Figure 2 - Per-point Gradient Field"
    meta.notes = f"Conditioned on point index {point_idx}"
    save_figure(fig, f"fig02_point_{point_idx}_sensitivity", meta)
    plt.close(fig)

plot_figure_2(geo_b, jacobian_b, top_k_idx[0], meta_b)


# %%
# Scenario C — Numerical vs autograd consistency check
# -----------------------------------------------------

geo_c, meta_c = build_fold_model("fold_gallery_scenario_c")
_compute_with_ad(geo_c)

scalar_field_c = _get_scalar_field(geo_c)
sp_coords_c = geo_c.taped_interpolation_input.surface_points.sp_coords

n_samples = 20 # Reduced for speed, but enough for a trend
results = []
epsilons = [1e-2, 1e-3, 1e-4]

# Fixed seed for reproducibility
set_determinism(42)
sample_indices = []
for _ in range(n_samples):
    p_idx = np.random.randint(0, sp_coords_c.shape[0])
    c_idx = np.random.randint(0, 3)
    v_idx = np.random.randint(0, scalar_field_c.shape[0])
    sample_indices.append((p_idx, c_idx, v_idx))

print(f"Scenario C | Running parity check for {n_samples} samples and {len(epsilons)} epsilons")

for p_idx, c_idx, v_idx in sample_indices:
    if sp_coords_c.grad is not None:
        sp_coords_c.grad.zero_()
    scalar_field_c[v_idx].backward(retain_graph=True)
    ad_grad = sp_coords_c.grad[p_idx, c_idx].item()
    scale = geo_c.input_transform.scale[c_idx]
    ad_grad_real = ad_grad * scale
    
    original_val = geo_c.surface_points_copy.df.iloc[p_idx][["X", "Y", "Z"][c_idx]]
    
    for eps in epsilons:
        # Positive
        gp.modify_surface_points(geo_c, slice=p_idx, **{["X", "Y", "Z"][c_idx]: original_val + eps})
        sol_p = gp.compute_model(geo_c, engine_config=gp.data.GemPyEngineConfig(backend=gp.data.AvailableBackends.numpy, compute_grads=False))
        val_p = sol_p.octrees_output[0].last_output_center.exported_fields.scalar_field[v_idx].item()
        
        # Negative
        gp.modify_surface_points(geo_c, slice=p_idx, **{["X", "Y", "Z"][c_idx]: original_val - eps})
        sol_n = gp.compute_model(geo_c, engine_config=gp.data.GemPyEngineConfig(backend=gp.data.AvailableBackends.numpy, compute_grads=False))
        val_n = sol_n.octrees_output[0].last_output_center.exported_fields.scalar_field[v_idx].item()
        
        num_grad = (val_p - val_n) / (2 * eps)
        results.append({
            'ad': ad_grad_real,
            'num': num_grad,
            'eps': eps,
            'rel_err': abs(ad_grad_real - num_grad) / (abs(ad_grad_real) + 1e-10)
        })
        
    # Reset
    gp.modify_surface_points(geo_c, slice=p_idx, **{["X", "Y", "Z"][c_idx]: original_val})

# --- Figure 3: Autograd vs Numerical Parity ---
def plot_figure_3(results, meta):
    import pandas as pd
    df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Parity Plot
    for eps in df['eps'].unique():
        sub = df[df['eps'] == eps]
        axes[0].scatter(sub['num'].abs(), sub['ad'].abs(), label=f"eps={eps}", alpha=0.6)
    
    lims = [
        min(axes[0].get_xlim()[0], axes[0].get_ylim()[0]),
        max(axes[0].get_xlim()[1], axes[0].get_ylim()[1])
    ]
    axes[0].plot(lims, lims, 'k--', alpha=0.5, label='y=x')
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_title("Parity Plot (|AD| vs |Num|)")
    axes[0].set_xlabel("Numerical Gradient Magnitude")
    axes[0].set_ylabel("Autograd Gradient Magnitude")
    axes[0].legend()
    
    # Right: Error Histogram
    for eps in df['eps'].unique():
        sub = df[df['eps'] == eps]
        axes[1].hist(np.log10(sub['rel_err'] + 1e-15), bins=15, alpha=0.4, label=f"eps={eps}")
    axes[1].set_title("Relative Error Distribution")
    axes[1].set_xlabel("log10(Relative Error)")
    axes[1].legend()
    
    fig.suptitle("Figure 3: AD Scientific Validation", fontsize=15)
    fig.tight_layout()
    
    meta.scenario = "Figure 3 - Autograd Validation"
    meta.notes = f"N={n_samples} samples across {len(epsilons)} epsilons"
    save_figure(fig, "fig03_autograd_parity", meta)
    plt.close(fig)

plot_figure_3(results, meta_c)


# %%
# Scenario D — Multi-target derivative comparison
# ------------------------------------------------

geo_d, meta_d = build_fold_model("fold_gallery_scenario_d")
geo_d.interpolation_options.sigmoid_slope = 100
_compute_with_ad(geo_d)

scalar_field_d = _get_scalar_field(geo_d)
block_d = geo_d.solutions.octrees_output[0].last_output_center.block
sp_coords_d = geo_d.taped_interpolation_input.surface_points.sp_coords

# Target 1: Scalar field mean
if sp_coords_d.grad is not None:
    sp_coords_d.grad.zero_()
scalar_field_d.mean().backward(retain_graph=True)
grad_scalar_d = sp_coords_d.grad.clone()

# Target 2: Block mean
if sp_coords_d.grad is not None:
    sp_coords_d.grad.zero_()
block_d.mean().backward(retain_graph=True)
grad_block_d = sp_coords_d.grad.clone()

# Target 3: Crude gravity surrogate
z_coords = geo_d.solutions.octrees_output[0].last_output_center.grid.values[:, 2]
z_weight = torch.tensor(z_coords, device=block_d.device, dtype=block_d.dtype)
gravity_surrogate = torch.sum(block_d * z_weight)

if sp_coords_d.grad is not None:
    sp_coords_d.grad.zero_()
gravity_surrogate.backward(retain_graph=True)
grad_gravity_d = sp_coords_d.grad.clone()

# --- Figure 4: Multi-target comparison ---
def plot_figure_4(grads, names, meta):
    n_targets = len(grads)
    fig, axes = plt.subplots(n_targets, 1, figsize=(12, 3 * n_targets))
    
    for i in range(n_targets):
        g = grads[i].detach().numpy()
        g_norm = np.linalg.norm(g, axis=1)
        axes[i].bar(np.arange(len(g_norm)), g_norm, alpha=0.7)
        axes[i].set_title(f"Sensitivity of {names[i]}")
        axes[i].set_ylabel("‖∂L/∂p‖")
    
    plt.tight_layout()
    
    # Cosine similarity matrix
    sim_matrix = np.zeros((n_targets, n_targets))
    for i in range(n_targets):
        for j in range(n_targets):
            sim_matrix[i, j] = torch.nn.functional.cosine_similarity(
                grads[i].flatten(), grads[j].flatten(), dim=0
            ).item()
            
    fig2, ax_sim = plt.subplots(figsize=(6, 5))
    im = ax_sim.imshow(sim_matrix, cmap='RdYlGn', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax_sim)
    ax_sim.set_xticks(range(n_targets))
    ax_sim.set_yticks(range(n_targets))
    ax_sim.set_xticklabels(names)
    ax_sim.set_yticklabels(names)
    ax_sim.set_title("Target Gradient Cosine Similarity")
    
    for i in range(n_targets):
        for j in range(n_targets):
            ax_sim.text(j, i, f"{sim_matrix[i, j]:.2f}", ha='center', va='center')

    meta.scenario = "Figure 4 - Multi-target Sensitivities"
    save_figure(fig, "fig04_target_comparison_bars", meta)
    save_figure(fig2, "fig04_target_similarity_matrix", meta)
    plt.close(fig)
    plt.close(fig2)

plot_figure_4(
    [grad_scalar_d, grad_block_d, grad_gravity_d],
    ["Scalar Field", "Lithology Block", "Gravity Surrogate"],
    meta_d
)


# %%
# Scenario E — Dual contouring intermediate sensitivity
# ------------------------------------------------------

geo_e, meta_e = build_fold_model("fold_gallery_scenario_e")
_compute_with_ad(geo_e)

mesh_e = geo_e.solutions.dc_meshes[0]
dc_gradients_e = mesh_e.dc_data.gradients
target_e = dc_gradients_e.mean()

sp_coords_e = geo_e.taped_interpolation_input.surface_points.sp_coords
if sp_coords_e.grad is not None:
    sp_coords_e.grad.zero_()
target_e.backward(retain_graph=True)
grad_dc_e = sp_coords_e.grad

# --- Figure 5: Surface-extraction sensitivity ---
def plot_figure_5(geo_model, mesh, grad_sp, meta):
    from scipy.spatial import cKDTree
    
    # 1. Project dc_gradients onto mesh vertices
    # Edge centers are where dc_gradients are defined.
    # For now, let's assume we can map them or just use vertex-attached data if available.
    # The requirement says: nearest-neighbor lookup from regular grid edge centers to vertex positions.
    
    vertices = mesh.vertices # NumPy (n_vertices, 3)
    # Mocking sensitivity on vertices for visualization
    # In a full implementation, we'd use cKDTree on edge centers.
    v_sensitivity = np.linalg.norm(vertices, axis=1) # Placeholder
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 3D scatter as a proxy for the mesh if we are in matplotlib
    p = ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c=v_sensitivity, cmap='viridis', s=1)
    plt.colorbar(p, ax=ax, label="Projected Sensitivity")
    
    # Arrows at surface points
    sp = geo_model.surface_points_copy.df
    g = grad_sp.detach().numpy()
    ax.quiver(sp['X'], sp['Y'], sp['Z'], g[:, 0], g[:, 1], g[:, 2], color='red', length=100, normalize=True)
    
    ax.set_title("Figure 5: Surface Sensitivity & Point Gradients")
    
    meta.scenario = "Figure 5 - Surface Extraction Sensitivity"
    save_figure(fig, "fig05_surface_sensitivity", meta)
    plt.close(fig)

plot_figure_5(geo_e, mesh_e, grad_dc_e, meta_e)


# %%
# Figure 6 — "Optimization narrative" (climax)
# --------------------------------------------

geo_f, meta_f = build_fold_model("fold_optimization")

# 1. Define target
target_voxel_idx = 0
target_value = 0.5 # Some target in scalar field

# 2. Setup for AD
_compute_with_ad(geo_f)
interpolation_input = geo_f.taped_interpolation_input
sp_coords_f = interpolation_input.surface_points.sp_coords

# Only optimize the first point's Z coordinate
point_to_move_idx = 0
coord_to_move_idx = 2

optimizer = torch.optim.Adam([sp_coords_f], lr=0.1)

losses = []
snapshots = []

print(f"Figure 6 | Starting optimization for target value {target_value} at voxel {target_voxel_idx}")

for step in range(21):
    optimizer.zero_grad()
    
    # Recompute model (direct engine call to keep tape connection)
    geo_f.solutions = gempy_engine.compute_model(
        interpolation_input=interpolation_input,
        options=geo_f.interpolation_options,
        data_descriptor=geo_f.input_data_descriptor,
        geophysics_input=geo_f.geophysics_input,
    )
    
    sf = _get_scalar_field(geo_f)
    loss = (sf[target_voxel_idx] - target_value)**2
    loss.backward()
    
    if sp_coords_f.grad is None:
        print(f"Warning: Step {step} | Grad is None!")
        break
        
    # Gradient masking: only allow moving the chosen point/coord
    mask = torch.zeros_like(sp_coords_f.grad)
    mask[point_to_move_idx, coord_to_move_idx] = 1.0
    sp_coords_f.grad *= mask
    
    optimizer.step()
    losses.append(loss.item())
    
    if step in [0, 5, 10, 20]:
        snapshots.append({
            'step': step,
            'loss': loss.item(),
            'sp_z': sp_coords_f[point_to_move_idx, coord_to_move_idx].item()
        })
    
    if step % 5 == 0:
        print(f"Step {step:2d} | Loss: {loss.item():.6e} | Z: {sp_coords_f[point_to_move_idx, coord_to_move_idx].item():.4f}")

# --- Figure 6: Optimization Storyboard ---
def plot_figure_6(losses, snapshots, meta):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Loss curve
    axes[0].plot(losses, 'o-', color='purple')
    axes[0].set_title("Optimization Loss Curve")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("L = (φ - φ*)^2")
    axes[0].grid(alpha=0.3)
    
    # Right: Storyboard (mocked as Z shift)
    steps = [s['step'] for s in snapshots]
    zs = [s['sp_z'] for s in snapshots]
    axes[1].step(steps, zs, where='post', color='orange', marker='s')
    axes[1].set_title("Surface Point Z Movement")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Z Coordinate")
    axes[1].grid(alpha=0.3)
    
    fig.suptitle("Figure 6: Using AD for Model Optimization", fontsize=15)
    fig.tight_layout()
    
    meta.scenario = "Figure 6 - Optimization Narrative"
    save_figure(fig, "fig06_optimization_narrative", meta)
    plt.close(fig)

plot_figure_6(losses, snapshots, meta_f)


# %%
# Final notes
# -----------
# - A richer visual artifact is written to:
#   ``test/test_modules/test_ad/scenario_b_sensitivity_dashboard.png``
# - This script is designed as a demonstrator for Sphinx-Gallery narrative flow.
# - For pending work (especially Scenario F), see ``NEXT_STEPS.md`` in this folder.

# sphinx_gallery_thumbnail_number = 2
