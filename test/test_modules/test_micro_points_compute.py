import numpy as np
import pytest

import gempy as gp
from gempy.modules.data_manipulation import interpolation_input_from_structural_frame
from gempy_engine.core.backend_tensor import BackendTensor
from test.test_modules.test_micro_points_serialization import _model_with_micro_points


def test_micro_supports_follow_composed_model_transforms_and_surface_order():
    model = _model_with_micro_points()
    elements = model.structural_frame.structural_groups[0].elements
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    authored = np.stack([np.eye(4), np.eye(4)])
    authored[0, :3, :3] = rotation @ np.diag([5.0, 2.0, 0.4])
    authored[0, :3, 3] = [120.0, 230.0, 340.0]
    authored[1, :3, :3] = rotation @ np.diag([3.0, 7.0, 0.25])
    authored[1, :3, 3] = [420.0, 530.0, 640.0]
    elements[0].micro_points = gp.data.MicroPointsTable.from_transforms(
        authored, names=elements[0].name, nugget=np.array([0.02, 0.04]),
        name_id_map=model.structural_frame.element_name_id_map,
    )
    model.grid.transform.rotation = np.array([0.0, 0.0, 25.0])
    model.grid.transform.scale = np.array([1.4, 0.8, 1.2])
    model.input_transform.rotation = np.array([10.0, 0.0, -15.0])
    model.input_transform.scale = np.array([0.004, 0.002, 0.006])

    result = interpolation_input_from_structural_frame(model).micro_points
    assert result is not None
    original = np.concatenate([authored, elements[1].micro_points.support_transforms])
    centers = original[:, :3, 3]
    grid = model.grid.transform
    inp = model.input_transform
    grid_matrix = grid.get_transform_matrix()
    pivot = grid.cached_pivot
    grid_matrix[:3, 3] += pivot - grid_matrix[:3, :3] @ pivot
    linear = (inp.get_transform_matrix() @ grid_matrix)[:3, :3]
    expected_centers = inp.apply(grid.apply_with_cached_pivot(centers))
    expected_supports = linear @ original[:, :3, :3]

    np.testing.assert_allclose(result.points, expected_centers)
    np.testing.assert_allclose(result.anisotropy_matrices, np.linalg.inv(expected_supports), atol=1e-9)
    np.testing.assert_allclose(result.support_to_engine, linear)
    np.testing.assert_allclose(
        np.einsum('nij,njk->nik', result.anisotropy_matrices, expected_supports),
        np.broadcast_to(np.eye(3), (3, 3, 3)), atol=1e-12,
    )
    np.testing.assert_array_equal(result.surface_indices, [0, 0, 1])
    np.testing.assert_array_equal(result.nuggets, [0.02, 0.04, 0.0])
    np.testing.assert_array_equal(elements[0].micro_points.support_transforms, authored)


def _compute_ready_model():
    model = _model_with_micro_points()
    for element in model.structural_frame.structural_groups[0].elements[:2]:
        center = element.surface_points.xyz[0] + [0.0, 0.0, 30.0]
        transforms = np.stack([np.eye(4), np.eye(4)])
        transforms[:, :3, :3] = np.diag([100.0, 80.0, 20.0])
        transforms[:, :3, 3] = [center, center + [10.0, 5.0, 0.0]]
        element.micro_points = gp.data.MicroPointsTable.from_transforms(
            transforms, names=element.name, nugget=np.array([0.02, 0.04]),
            name_id_map=model.structural_frame.element_name_id_map,
        )
    model.interpolation_options.evaluation_options.number_octree_levels = 1
    model.interpolation_options.micro_options.enabled = True
    model.interpolation_options.micro_options.kernel_range = 0.37
    return model


def _elements_by_id(model):
    return {element.id: element for element in model.structural_frame.structural_elements}


def test_aligned_micro_supports_write_back_to_original_rows_with_composed_transform():
    model = _compute_ready_model()
    model.grid.transform.rotation = np.array([0.0, 0.0, 25.0])
    model.grid.transform.scale = np.array([1.4, 0.8, 1.2])
    model.input_transform.rotation = np.array([10.0, 0.0, -15.0])
    model.input_transform.scale = np.array([0.004, 0.002, 0.006])
    before = model.structural_frame.micro_points_copy.data.copy()
    root = interpolation_input_from_structural_frame(model).micro_points

    solution = gp.compute_model(model)
    result = solution.micro_point_results
    assert result is not None
    np.testing.assert_array_equal(np.sort(result.source_indices), np.arange(4))
    assert result.macro_gradients.shape == (4, 3)
    assert np.isfinite(result.macro_gradients).all()
    after = model.structural_frame.micro_points_copy.data
    # Reordering may change element order; compare by ownership rather than concatenated row order.
    for element_id, element in _elements_by_id(model).items():
        old = before[before['element_id'] == element_id]
        np.testing.assert_array_equal(element.micro_points.xyz, old['support_transform'][:, :3, 3])
        np.testing.assert_array_equal(element.micro_points.nugget, old['nugget'])
        np.testing.assert_array_equal(element.micro_points.ids, old['element_id'])
        np.testing.assert_array_equal(element.micro_points.support_transforms[:, 3, :], old['support_transform'][:, 3, :])
    assert len(after) == 4
    for index, matrix in zip(result.source_indices, result.anisotropy_matrices):
        old_id = before['element_id'][index]
        row = np.count_nonzero(before['element_id'][:index] == old_id)
        expected = np.linalg.solve(root.support_to_engine, np.linalg.inv(matrix))
        np.testing.assert_allclose(_elements_by_id(model)[old_id].micro_points.support_transforms[row, :3, :3], expected)
        np.testing.assert_allclose(np.linalg.norm(expected, axis=0), np.linalg.norm(before['support_transform'][index, :3, :3], axis=0))
    assert any(not np.allclose(_elements_by_id(model)[element_id].micro_points.support_transforms[:, :3, :3],
                               before[before['element_id'] == element_id]['support_transform'][:, :3, :3])
               for element_id in _elements_by_id(model) if len(_elements_by_id(model)[element_id].micro_points))


@pytest.mark.parametrize('transformed', [False, True])
def test_aligned_micro_compute_save_load_recompute_is_stable(tmp_path, transformed):
    model = _compute_ready_model()
    if transformed:
        model.grid.transform.rotation = np.array([0.0, 0.0, 25.0])
        model.grid.transform.scale = np.array([1.4, 0.8, 1.2])
        model.input_transform.rotation = np.array([10.0, 0.0, -15.0])
        model.input_transform.scale = np.array([0.004, 0.002, 0.006])
    model.interpolation_options.evaluation_options.mesh_extraction = True
    first = gp.compute_model(model)
    assert first.micro_point_results is not None
    aligned = {i: e.micro_points.data.copy() for i, e in _elements_by_id(model).items()}
    path = tmp_path / 'aligned.gempy'
    gp.save_model(model, path=str(path))
    loaded = gp.load_model(str(path))
    assert loaded.interpolation_options.micro_options.enabled
    assert loaded.interpolation_options.micro_options.align_to_macro
    assert loaded.interpolation_options.micro_options.kernel_range == 0.37
    for i, element in _elements_by_id(loaded).items():
        np.testing.assert_array_equal(element.micro_points.data, aligned[i])

    repeated = gp.compute_model(model)
    restored = gp.compute_model(loaded)
    for i, element in _elements_by_id(model).items():
        np.testing.assert_allclose(element.micro_points.support_transforms, aligned[i]['support_transform'], atol=1e-9)
        np.testing.assert_allclose(_elements_by_id(loaded)[i].micro_points.support_transforms,
                                   aligned[i]['support_transform'], atol=1e-9)
    for subsequent in (repeated, restored):
        np.testing.assert_allclose(subsequent.raw_arrays.scalar_field_matrix, first.raw_arrays.scalar_field_matrix, atol=1e-7)
        for mesh, reference in zip(subsequent.dc_meshes, first.dc_meshes):
            if reference is not None:
                np.testing.assert_allclose(mesh.vertices, reference.vertices, atol=1e-7)
                np.testing.assert_array_equal(mesh.edges, reference.edges)


def test_micro_writeback_maps_sparse_results_before_element_reordering():
    from types import SimpleNamespace
    from gempy_engine.core.data.micro_points import MicroPointResults

    model = _compute_ready_model()
    group = model.structural_frame.structural_groups[0]
    original_elements = group.elements.copy()
    before = [element.micro_points.support_transforms.copy() for element in original_elements]
    root = interpolation_input_from_structural_frame(model).micro_points
    rotation = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    supports = np.stack([rotation @ before[1][1, :3, :3], rotation.T @ before[0][0, :3, :3]])
    model.solutions = SimpleNamespace(
        micro_point_results=MicroPointResults(
            source_indices=np.array([3, 0]),
            macro_gradients=np.zeros((2, 3)),
            anisotropy_matrices=np.linalg.inv(root.support_to_engine @ supports),
        ),
        scalar_field_at_surface_points=np.array([0., 1.]),
        dc_meshes=None,
        _ordered_elements=[np.array([1, 0])],
    )
    assert group.elements[0] is original_elements[1]
    assert group.elements[1] is original_elements[0]
    np.testing.assert_allclose(original_elements[1].micro_points.support_transforms[1, :3, :3], supports[0])
    np.testing.assert_allclose(original_elements[0].micro_points.support_transforms[0, :3, :3], supports[1])
    np.testing.assert_array_equal(original_elements[0].micro_points.support_transforms[1], before[0][1])
    np.testing.assert_array_equal(original_elements[1].micro_points.support_transforms[0], before[1][0])


@pytest.mark.parametrize('enabled,align', [(False, True), (True, False)])
def test_micro_alignment_opt_out_preserves_authored_supports(enabled, align):
    model = _compute_ready_model()
    model.interpolation_options.micro_options.enabled = enabled
    model.interpolation_options.micro_options.align_to_macro = align
    original = {i: e.micro_points.data.copy() for i, e in _elements_by_id(model).items()}
    result = gp.compute_model(model)
    assert result.micro_point_results is None
    for i, element in _elements_by_id(model).items():
        np.testing.assert_array_equal(element.micro_points.data, original[i])


@pytest.fixture
def torch_backend():
    torch = pytest.importorskip('torch')
    grad_enabled = torch.is_grad_enabled()
    try:
        yield torch
    finally:
        torch.set_default_device('cpu')
        BackendTensor._change_backend(gp.data.AvailableBackends.numpy)
        torch.set_grad_enabled(grad_enabled)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_authored_numpy_micro_points_compute_with_torch_and_macro_gradients(torch_backend, device):
    torch = torch_backend
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    model = _model_with_micro_points()
    elements = model.structural_frame.structural_groups[0].elements
    for element in elements[:2]:
        center = element.surface_points.xyz[0] + np.array([0.0, 0.0, 30.0])
        support = np.eye(4)
        support[:3, :3] = np.diag([100.0, 100.0, 20.0])
        support[:3, 3] = center
        element.micro_points = gp.data.MicroPointsTable.from_transforms(
            np.stack([support]), names=element.name,
            name_id_map=model.structural_frame.element_name_id_map,
        )
    model.interpolation_options.evaluation_options.number_octree_levels = 1
    model.interpolation_options.evaluation_options.mesh_extraction = False
    model.interpolation_options.micro_options.enabled = True

    solution = gp.compute_model(model, engine_config=gp.data.GemPyEngineConfig(
        backend=gp.data.AvailableBackends.PYTORCH, use_gpu=device == 'cuda',
        dtype='float64', compute_grads=True,
    ))
    micro = model.taped_interpolation_input.micro_points
    assert isinstance(solution.micro_point_results.anisotropy_matrices, torch.Tensor)
    for element in elements[:2]:
        assert isinstance(element.micro_points.support_transforms, np.ndarray)
    assert micro is not None
    np.testing.assert_array_equal(micro.surface_indices, [0, 1])
    for values in (micro.points, micro.anisotropy_matrices, micro.nuggets):
        assert isinstance(values, np.ndarray)
        assert np.isfinite(values).all()

    macro = model.taped_interpolation_input.surface_points.sp_coords
    assert macro.device.type == device
    assert macro.requires_grad
    scalar = solution.octrees_output[0].last_output_center.exported_fields.scalar_field
    field = scalar[0] if isinstance(scalar, list) else scalar
    assert isinstance(field, torch.Tensor)
    assert field.device.type == device
    assert torch.isfinite(field).all()
    field.sum().backward()
    assert macro.grad is not None
    assert torch.isfinite(macro.grad).all()
