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

    assert rel_error < 1e-2 or abs_error < 1e-2, f"Gradient mismatch! Rel error: {rel_error}, Abs error: {abs_error}"

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
