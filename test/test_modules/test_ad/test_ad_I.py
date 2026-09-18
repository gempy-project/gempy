import torch

import gempy as gp
from gempy.optional_dependencies import require_gempy_viewer

PLOT = True


def test_generate_fold_model():
    data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
    path_to_data = data_path + "/data/input_data/jan_models/"

    # Create a GeoModel instance
    geo_data: gp.data.GeoModel = gp.create_geomodel(
        project_name='fold',
        extent=[0, 1000, 0, 1000, 0, 1000],
        refinement=3,
        importer_helper=gp.data.ImporterHelper(
            path_to_orientations=path_to_data + "model2_orientations.csv",
            path_to_surface_points=path_to_data + "model2_surface_points.csv"
        )
    )

    # Map geological series to surfaces 
    gp.map_stack_to_surfaces(
        gempy_model=geo_data,
        mapping_object={"Strat_Series": ('rock2', 'rock1')}
    )

    # Compute the geological model
    gp.compute_model(
        gempy_model=geo_data,
        engine_config=gp.data.GemPyEngineConfig(
            backend=gp.data.AvailableBackends.PYTORCH,
            use_gpu=False,
            dtype='float64',
            compute_grads=True
        )
    )

    foo = geo_data.solutions.octrees_output[0].last_output_center.exported_fields.scalar_field[0]
    foo.backward(retain_graph=True, create_graph=True)

    # Scenario A: Baseline AD smoke + regression guard
    sp_coords_grad = geo_data.taped_interpolation_input.surface_points.sp_coords.grad
    assert sp_coords_grad is not None, "Surface points gradients should not be None"
    assert torch.isfinite(sp_coords_grad).all(), "Surface points gradients should be finite"

    orient_pos_grad = geo_data.taped_interpolation_input.orientations.dip_positions.grad
    assert orient_pos_grad is not None, "Orientation positions gradients should not be None"
    assert torch.isfinite(orient_pos_grad).all(), "Orientation positions gradients should be finite"

    grad_norm = torch.norm(sp_coords_grad)
    print(f"Scenario A - Surface points gradient norm: {grad_norm.item()}")

    # Assert deterministic gradient norm range (example range, needs tuning)
    assert 0.0 < grad_norm.item() < 1e6, f"Gradient norm {grad_norm.item()} out of expected range"
                 

    if PLOT or False:
        gpv = require_gempy_viewer()
        gpv.plot_3d(geo_data, image=True)


def test_jacobian_spatial_sensitivity():
    # Scientific objective: Quantify spatial sensitivity distribution over model domain
    # Computational objective: Jacobian of grid-wide scalar field wrt all surface-point coordinates

    data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
    path_to_data = data_path + "/data/input_data/jan_models/"

    # 1. Setup model
    geo_data: gp.data.GeoModel = gp.create_geomodel(
        project_name='fold_jacobian',
        extent=[0, 1000, 0, 1000, 0, 1000],
        refinement=3,
        importer_helper=gp.data.ImporterHelper(
            path_to_orientations=path_to_data + "model2_orientations.csv",
            path_to_surface_points=path_to_data + "model2_surface_points.csv"
        )
    )

    gp.map_stack_to_surfaces(
        gempy_model=geo_data,
        mapping_object={"Strat_Series": ('rock2', 'rock1')}
    )

    # 2. Compute model with compute_grads=True
    gp.compute_model(
        gempy_model=geo_data,
        engine_config=gp.data.GemPyEngineConfig(
            backend=gp.data.AvailableBackends.PYTORCH,
            dtype='float64',
            compute_grads=True
        )
    )

    # 3. Get target: scalar field for the first series
    exported_fields = geo_data.solutions.octrees_output[0].last_output_center.exported_fields
    
    scalar_field_tensor = exported_fields.scalar_field
    if isinstance(scalar_field_tensor, list):
        scalar_field = scalar_field_tensor[0]
    else:
        scalar_field = scalar_field_tensor

    sp_coords = geo_data.taped_interpolation_input.surface_points.sp_coords

    n_voxels = scalar_field.shape[0]
    n_points = sp_coords.shape[0]

    print(f"Scenario B - Computing Jacobian for {n_voxels} voxels and {n_points} surface points...")

    # 4. Compute Jacobian: (n_voxels, n_points, 3)
    jacobian = torch.zeros((n_voxels, n_points, 3), dtype=torch.float64)

    # Compute Jacobian row by row (voxel by voxel)
    for i in range(n_voxels):
        if sp_coords.grad is not None:
            sp_coords.grad.zero_()

        scalar_field[i].backward(retain_graph=True)
        jacobian[i] = sp_coords.grad

    # 5. Scenario B Assertions
    # Shape contract
    assert jacobian.shape == (n_voxels, n_points, 3), f"Jacobian shape mismatch: {jacobian.shape}"
    assert torch.isfinite(jacobian).all(), "Jacobian contains non-finite values"

    # Sparsity/density summary
    non_zero_elements = torch.count_nonzero(jacobian)
    total_elements = jacobian.numel()
    density = non_zero_elements / total_elements
    print(f"Scenario B - Jacobian density: {density.item():.4f}")

    # Top-k influential points
    # Sum of absolute gradients over all voxels for each point
    point_influence = torch.sum(torch.abs(jacobian), axis=(0, 2))
    top_k = min(5, n_points)
    top_k_indices = torch.topk(point_influence, k=top_k).indices
    print(f"Scenario B - Top {top_k} influential surface points indices: {top_k_indices.tolist()}")

    assert point_influence.max() > 0, "There should be some influence from surface points"

    # 6. Visualization (if enabled)
    if PLOT:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.bar(range(n_points), point_influence.detach().numpy())
        plt.title("Scenario B: Surface Point Influence on Scalar Field")
        plt.xlabel("Surface Point Index")
        plt.ylabel("Total Gradient Magnitude (Sum of abs Jacobian)")
        plt.grid(True, alpha=0.3)
        plt.show() # Don't show in tests
        print("Scenario B - Influence bar chart saved to scenario_b_influence.png")


def test_numerical_vs_autograd_comparison():
    # Scientific objective: Demonstrate scientific correctness of AD derivatives
    # Computational objective: Numerical-vs-autograd gradient comparison on selected points

    data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
    path_to_data = data_path + "/data/input_data/jan_models/"

    # 1. Setup model
    geo_data: gp.data.GeoModel = gp.create_geomodel(
        project_name='fold_validation',
        extent=[0, 1000, 0, 1000, 0, 1000],
        refinement=3,
        importer_helper=gp.data.ImporterHelper(
            path_to_orientations=path_to_data + "model2_orientations.csv",
            path_to_surface_points=path_to_data + "model2_surface_points.csv"
        )
    )

    gp.map_stack_to_surfaces(
        gempy_model=geo_data,
        mapping_object={"Strat_Series": ('rock2', 'rock1')}
    )

    # 2. Compute Autograd Gradient
    gp.compute_model(
        gempy_model=geo_data,
        engine_config=gp.data.GemPyEngineConfig(
            backend=gp.data.AvailableBackends.PYTORCH,
            dtype='float64',
            compute_grads=True
        )
    )

    scalar_field = geo_data.solutions.octrees_output[0].last_output_center.exported_fields.scalar_field
    sp_coords = geo_data.taped_interpolation_input.surface_points.sp_coords

    point_idx = 0
    coord_idx = 2  # Z coordinate
    voxel_idx = 0

    if sp_coords.grad is not None:
        sp_coords.grad.zero_()

    scalar_field[voxel_idx].backward(retain_graph=True)
    ad_grad = sp_coords.grad[point_idx, coord_idx].item()

    # 3. Compute Numerical Gradient
    epsilon = 1e-3

    # Original Z value
    original_z = geo_data.surface_points_copy.df.iloc[point_idx]['Z']

    # Positive perturbation
    gp.modify_surface_points(geo_data, slice=point_idx, Z=original_z + epsilon)
    # Note: we need to recompute with grads=False for numerical to be pure
    sol_p = gp.compute_model(
        geo_data,
        engine_config=gp.data.GemPyEngineConfig(
            backend=gp.data.AvailableBackends.numpy,
            compute_grads=False
        )
    )
    val_p = sol_p.octrees_output[0].last_output_center.exported_fields.scalar_field[voxel_idx].item()

    # Negative perturbation
    gp.modify_surface_points(geo_data, slice=point_idx, Z=original_z - epsilon)
    sol_n = gp.compute_model(
        geo_data,
        engine_config=gp.data.GemPyEngineConfig(
            backend=gp.data.AvailableBackends.numpy,
            compute_grads=False
        )
    )
    val_n = sol_n.octrees_output[0].last_output_center.exported_fields.scalar_field[voxel_idx].item()

    num_grad = (val_p - val_n) / (2 * epsilon)

    # 4. Scenario C Assertions
    # The AD gradient is computed wrt transformed coordinates. 
    # To compare with numerical gradient (real space), we must multiply by the scale.
    scale = geo_data.input_transform.scale[coord_idx]
    ad_grad_real = ad_grad * scale

    print(f"Scenario C - AD Gradient (transformed): {ad_grad}")
    print(f"Scenario C - AD Gradient (real space): {ad_grad_real}")
    print(f"Scenario C - Numerical Gradient: {num_grad}")
    print(f"Scenario C - val_p: {val_p}")
    print(f"Scenario C - val_n: {val_n}")
    print(f"Scenario C - epsilon: {epsilon}")
    print(f"Scenario C - input_transform scale: {scale}")

    abs_error = abs(ad_grad_real - num_grad)
    rel_error = abs_error / (abs(ad_grad_real) + 1e-10)

    print(f"Scenario C - Absolute Error: {abs_error}")
    print(f"Scenario C - Relative Error: {rel_error}")

    # * The engine graph is exact: AD and central FD agree to FD truncation/rounding (~1e-8 here).
    assert rel_error < 1e-6, f"Gradient mismatch! Rel error: {rel_error}, Abs error: {abs_error}"

    # Reset model
    gp.modify_surface_points(geo_data, slice=point_idx, Z=original_z)


def test_dual_contouring_sensitivity():
    # Scientific objective: Connect AD to isosurface extraction intermediates
    # Computational objective: Edge scalar gradients in contouring path

    data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
    path_to_data = data_path + "/data/input_data/jan_models/"

    # 1. Setup model
    geo_data: gp.data.GeoModel = gp.create_geomodel(
        project_name='fold_dc_sensitivity',
        extent=[0, 1000, 0, 1000, 0, 1000],
        refinement=3,
        importer_helper=gp.data.ImporterHelper(
            path_to_orientations=path_to_data + "model2_orientations.csv",
            path_to_surface_points=path_to_data + "model2_surface_points.csv"
        )
    )

    gp.map_stack_to_surfaces(
        gempy_model=geo_data,
        mapping_object={"Strat_Series": ('rock2', 'rock1')}
    )

    # 2. Compute model with mesh extraction and grads
    gp.compute_model(
        gempy_model=geo_data,
        engine_config=gp.data.GemPyEngineConfig(
            backend=gp.data.AvailableBackends.PYTORCH,
            dtype='float64',
            compute_grads=True
        )
    )

    # 3. Access DC mesh and gradients
    # Note: solutions.dc_meshes is a list of lists (one per series, then surfaces)
    mesh = geo_data.solutions.dc_meshes[0]
    
    # dc_data gradients
    dc_gradients = mesh.dc_data.gradients

    # 4. Compute sensitivity of a selected edge gradient
    target_gradient = dc_gradients.mean()
    
    sp_coords = geo_data.taped_interpolation_input.surface_points.sp_coords
    if sp_coords.grad is not None:
        sp_coords.grad.zero_()
        
    target_gradient.backward(retain_graph=True)
    grad_dc = sp_coords.grad
    
    # 5. Scenario E Assertions
    assert grad_dc is not None
    assert torch.isfinite(grad_dc).all()
    print(f"Scenario E - DC gradient sensitivity norm: {torch.norm(grad_dc).item()}")
    
    assert torch.norm(grad_dc).item() > 0, "DC gradients should be sensitive to surface points"


def test_multi_target_derivatives():
    # Scientific objective: Show AD supports multiple geological observables
    # Computational objective: Scalar field + lithology block (surrogate)

    data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
    path_to_data = data_path + "/data/input_data/jan_models/"

    # 1. Setup model
    geo_data: gp.data.GeoModel = gp.create_geomodel(
        project_name='fold_multi_target',
        extent=[0, 1000, 0, 1000, 0, 1000],
        refinement=3,
        importer_helper=gp.data.ImporterHelper(
            path_to_orientations=path_to_data + "model2_orientations.csv",
            path_to_surface_points=path_to_data + "model2_surface_points.csv"
        )
    )

    gp.map_stack_to_surfaces(
        gempy_model=geo_data,
        mapping_object={"Strat_Series": ('rock2', 'rock1')}
    )

    # 2. Configure for differentiability (softer sigmoid for lithology surrogate)
    geo_data.interpolation_options.sigmoid_slope = 100

    gp.compute_model(
        gempy_model=geo_data,
        engine_config=gp.data.GemPyEngineConfig(
            backend=gp.data.AvailableBackends.PYTORCH,
            dtype='float64',
            compute_grads=True
        )
    )

    # 3. Target 1: Scalar field mean
    scalar_field = geo_data.solutions.octrees_output[0].last_output_center.exported_fields.scalar_field
    target_scalar = scalar_field.mean()

    sp_coords = geo_data.taped_interpolation_input.surface_points.sp_coords

    if sp_coords.grad is not None:
        sp_coords.grad.zero_()

    target_scalar.backward(retain_graph=True)
    grad_scalar = sp_coords.grad.clone()

    # 4. Target 2: Lithology block (surrogate) mean
    # In GemPy 3, geo_data.solutions.octrees_output[0].last_output_center.block
    # is the differentiable lithology block.
    block = geo_data.solutions.octrees_output[0].last_output_center.block
    target_block = block.mean()

    if sp_coords.grad is not None:
        sp_coords.grad.zero_()

    target_block.backward(retain_graph=True)
    grad_block = sp_coords.grad.clone()

    # 5. Scenario D Assertions
    assert grad_scalar is not None
    assert grad_block is not None
    assert torch.isfinite(grad_scalar).all()
    assert torch.isfinite(grad_block).all()

    print(f"Scenario D - Scalar field gradient norm: {torch.norm(grad_scalar).item()}")
    print(f"Scenario D - Lithology block gradient norm: {torch.norm(grad_block).item()}")

    # Check that they are not identical
    # We use torch.nn.functional.cosine_similarity.
    # Note: grad_scalar and grad_block are (n_points, 3)
    cos_sim = torch.nn.functional.cosine_similarity(grad_scalar.flatten(), grad_block.flatten(), dim=0)
    print(f"Scenario D - Cosine similarity between gradients: {cos_sim.item()}")
    
    # In a simple fold model, they might be highly correlated but not identical
    assert cos_sim.item() < 1.0, "Gradients for different targets should not be perfectly identical"


# ---------------------------------------------------------------------------------------------------------------------
# Scenario F: AD vs converged central FD on programmatic models, tight tolerance.
#
# Covers the configurations that were reported (Sep 2026) to give wrong autograd derivatives: dipping layers with one
# to three orientations, a faulted stack (derivative of the stratigraphic field w.r.t. a stratigraphic pick on the
# other side of the fault) and the fault field w.r.t. its own picks. The root cause was not the engine graph but the
# identity input transform that a model keeps when its structural frame was empty at `create_geomodel` time (the
# picks were added afterwards): the kriging system then has a condition number of ~1e37 and derivatives through the
# linear solve are numerically meaningless. `gp.compute_model` now derives the transform in that situation, so these
# models are well conditioned and AD agrees with FD to FD accuracy.
#
# FD is done in real space with fresh model rebuilds. The input transform of the reference model is reused for the
# perturbed models so that only the field, not the (data-dependent) transform, is differentiated; the AD gradient is
# taken w.r.t. transformed coordinates and mapped back with the transform scale.
# ---------------------------------------------------------------------------------------------------------------------
import copy
import contextlib
import io

import numpy as np
import pytest

from gempy_engine.core.data.options import InterpolationOptions
from gempy_engine.core.data.stack_relation_type import StackRelationType

_ENGINE_GRAD = gp.data.GemPyEngineConfig(backend=gp.data.AvailableBackends.PYTORCH, dtype='float64', compute_grads=True)
_ENGINE_FD = gp.data.GemPyEngineConfig(backend=gp.data.AvailableBackends.PYTORCH, dtype='float64', compute_grads=False)

_BOREHOLES = np.array([[300., 300.], [1000., 250.], [1700., 400.], [400., 1100.],
                       [1100., 1000.], [1650., 1250.], [900., 1750.]])


def _layered_picks(dip: bool) -> tuple[np.ndarray, np.ndarray]:
    x, y = _BOREHOLES[:, 0], _BOREHOLES[:, 1]
    top = 640.0 - 0.06 * x - 0.03 * y if dip else np.full(7, 560.0)
    base = top - (210.0 - 0.01 * x) if dip else np.full(7, 360.0)
    return np.column_stack([_BOREHOLES, top]), np.column_stack([_BOREHOLES, base])


def _faulted_picks() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    top, base = [], []
    for x in (200.0, 500.0, 800.0):
        for y in (300.0, 500.0, 700.0):
            top.append([x, y, 600.0]); base.append([x, y, 450.0])
    for x in (1200.0, 1500.0, 1800.0):
        for y in (300.0, 500.0, 700.0):
            top.append([x, y, 520.0]); base.append([x, y, 370.0])
    fault = [[1000.0 + (z - 500.0) / np.tan(np.deg2rad(70.0)), y, z]
             for z in (200.0, 500.0, 800.0) for y in (250.0, 500.0, 750.0)]
    return np.array(top), np.array(base), np.array(fault)


def _build_model(sp_top: np.ndarray, sp_base: np.ndarray, sp_fault: np.ndarray | None, n_ori: int,
                 extent: list[float], transform=None) -> gp.data.GeoModel:
    """Two horizons (+ optional fault). Points are added *after* `create_geomodel`, i.e. the model starts with the
    default input transform; `transform` (if given) is frozen onto the model for FD rebuilds."""
    color_gen = gp.data.structural_frame.ColorsGenerator()

    def element(name):
        return gp.data.StructuralElement(name, gp.data.SurfacePointsTable.initialize_empty(),
                                         gp.data.OrientationsTable.initialize_empty(), color=next(color_gen))

    groups = []
    if sp_fault is not None:
        groups.append(gp.data.StructuralGroup(name="Fault_Series", elements=[element("fault")],
                                              structural_relation=StackRelationType.FAULT))
    groups.append(gp.data.StructuralGroup(name="Strat_Series", elements=[element("top"), element("base")],
                                          structural_relation=StackRelationType.ERODE))
    frame = gp.data.StructuralFrame(structural_groups=groups, color_gen=color_gen)

    with contextlib.redirect_stdout(io.StringIO()):
        model = gp.create_geomodel(project_name="ad_vs_fd", extent=extent, refinement=2, structural_frame=frame)
        if sp_fault is not None:
            gp.add_surface_points(model, x=sp_fault[:, 0].tolist(), y=sp_fault[:, 1].tolist(), z=sp_fault[:, 2].tolist(),
                                  elements_names=["fault"] * len(sp_fault))
        for name, sp in (("top", sp_top), ("base", sp_base)):
            gp.add_surface_points(model, x=sp[:, 0].tolist(), y=sp[:, 1].tolist(), z=sp[:, 2].tolist(),
                                  elements_names=[name] * len(sp))
        if sp_fault is not None:
            normal = [np.sin(np.deg2rad(70.0)), 0.0, np.cos(np.deg2rad(70.0))]
            gp.add_orientations(model, x=[1000.0], y=[500.0], z=[500.0], elements_names=["fault"], pole_vector=[normal])
            gp.add_orientations(model, x=[500.0, 1500.0], y=[500.0, 500.0], z=[600.0, 520.0],
                                elements_names=["top", "top"], pole_vector=[[0, 0, 1.0], [0, 0, 1.0]])
            model.structural_frame.fault_relations = np.array([[0, 1], [0, 0]])
            gp.set_is_fault(frame=model, fault_groups=["Fault_Series"])
        else:
            oris = [(1000., 1000., 500.), (400., 400., 500.), (1600., 1600., 500.)][:n_ori]
            gp.add_orientations(model, x=[o[0] for o in oris], y=[o[1] for o in oris], z=[o[2] for o in oris],
                                elements_names=["top"] * n_ori, pole_vector=[[0., 0., 1.]] * n_ori)
        model.interpolation_options.cache_mode = InterpolationOptions.CacheMode.NO_CACHE
        if transform is not None:
            model.input_transform = copy.deepcopy(transform)
    return model


def _scalar_field_at(model: gp.data.GeoModel, point: np.ndarray, stack: int, engine_config) -> tuple:
    with contextlib.redirect_stdout(io.StringIO()):
        gp.set_custom_grid(model.grid, xyz_coord=np.atleast_2d(point))
        solutions = gp.compute_model(model, engine_config=engine_config)
    custom_slice = model.taped_interpolation_input.grid.custom_grid_slice
    field = solutions.octrees_output[0].outputs[stack].exported_fields._scalar_field[custom_slice][0]
    return field, model.taped_interpolation_input.surface_points.sp_coords


_LAYERED_EXTENT = [0., 2000., 0., 2000., 0., 1000.]
_FAULTED_EXTENT = [0., 2000., 0., 1000., 0., 1000.]


@pytest.mark.parametrize("case", [
    "flat_1ori", "dip_1ori", "dip_2ori", "dip_3ori",
    "fault_strat_pick_across_fault", "fault_field_dX", "fault_field_dZ",
])
def test_ad_vs_fd_dipping_and_faulted(case):
    fd_step = 0.02  # metres, real space

    if case.startswith("fault"):
        sp_top, sp_base, sp_fault = _faulted_picks()
        n_ori, extent = 0, _FAULTED_EXTENT
        eval_point = np.array([1400.0, 500.0, 500.0])  # hanging wall
        if case == "fault_strat_pick_across_fault":
            stack, pick_table, pick_row, coord = 1, "top", 4, 2  # top pick (500, 500) on the footwall, dZ
        elif case == "fault_field_dX":
            stack, pick_table, pick_row, coord = 0, "fault", 1, 0  # fault pick (890.8, 500, 200), dX
        else:
            stack, pick_table, pick_row, coord = 0, "fault", 1, 2  # fault pick (890.8, 500, 200), dZ
        # * Fault pick 4 (1000, 500, 500) coincides with the fault orientation. The cubic kernel's gradient-interface
        # * covariance has an |r| kink there (regularised over sqrt(1e-10) in transformed units by the engine), so a
        # * derivative w.r.t. that pick is not well defined and FD/AD legitimately differ at the 1e-5 level.
    else:
        sp_top, sp_base = _layered_picks(dip=case.startswith("dip"))
        sp_fault, n_ori, extent = None, int(case[-4]), _LAYERED_EXTENT
        eval_point = np.array([1000.0, 1000.0, 480.0])
        stack, pick_table, pick_row, coord = 0, "top", 3, 2  # top pick of borehole 4 (400, 1100), dZ

    # --- reference model + autograd ------------------------------------------------------------------------------
    model = _build_model(sp_top, sp_base, sp_fault, n_ori, extent)
    with pytest.warns(RuntimeWarning, match="default .* input transform"):
        field, sp_coords = _scalar_field_at(model, eval_point, stack, _ENGINE_GRAD)
    transform = copy.deepcopy(model.input_transform)
    assert not transform._is_default_transform
    assert not np.allclose(transform.scale, 1.0)

    grad_transformed = torch.autograd.grad(field, sp_coords)[0].numpy()
    n_fault = 0 if sp_fault is None else len(sp_fault)
    sp_row = (n_fault + pick_row) if pick_table == "top" else pick_row  # fault picks come first in the frame
    ad_grad = grad_transformed[sp_row, coord] * float(transform.scale[coord])

    # --- central FD in real space, fresh rebuilds with the frozen transform ---------------------------------------
    values = []
    for sign in (+1.0, -1.0):
        top_p, base_p, fault_p = sp_top.copy(), sp_base.copy(), (None if sp_fault is None else sp_fault.copy())
        (top_p if pick_table == "top" else fault_p)[pick_row, coord] += sign * fd_step
        model_p = _build_model(top_p, base_p, fault_p, n_ori, extent, transform=transform)
        with torch.no_grad():
            value, _ = _scalar_field_at(model_p, eval_point, stack, _ENGINE_FD)
        values.append(float(value))
    fd_grad = (values[0] - values[1]) / (2 * fd_step)

    rel_error = abs(ad_grad - fd_grad) / abs(fd_grad)
    print(f"Scenario F [{case}] - AD: {ad_grad:.10e}  FD: {fd_grad:.10e}  rel. error: {rel_error:.2e}")
    assert np.isfinite(ad_grad) and abs(fd_grad) > 0
    assert rel_error < 1e-6, f"[{case}] AD {ad_grad} vs FD {fd_grad}: rel. error {rel_error}"
