import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import gempy as gp
from gempy.core.data.enumerators import ExampleModel
from gempy.modules.json_io.json_operations import JsonIO
from gempy.modules.serialization.save_load import model_to_bytes, _load_model_from_bytes
from gempy_engine.core.data.dual_contouring_mesh import DualContouringMesh
from gempy_engine.core.backend_tensor import BackendTensor


@pytest.fixture(autouse=True)
def restore_backend():
    backend = BackendTensor.engine_backend
    dtype, gpu, grads = BackendTensor.dtype, BackendTensor.use_gpu, BackendTensor.COMPUTE_GRADS
    keops = BackendTensor.use_pykeops
    torch = sys.modules.get('torch')
    grad_enabled = torch.is_grad_enabled() if torch is not None else True
    yield
    BackendTensor._change_backend(backend, use_pykeops=keops, use_gpu=gpu, dtype=dtype, grads=grads)
    if torch is not None:
        torch.set_grad_enabled(grad_enabled)


def model_and_config():
    model = gp.generate_example_model(ExampleModel.HORIZONTAL_STRAT, compute_model=False)
    evaluation = model.interpolation_options.evaluation_options
    evaluation.number_octree_levels = 2
    evaluation.number_octree_levels_surface = 2
    return model, gp.data.GemPyEngineConfig(backend=gp.data.AvailableBackends.numpy, dtype='float64')


@pytest.mark.parametrize('overlap', [None, 'joint', gp.data.DualContouringOverlap.joint_contacts])
def test_meshing_options_round_trip_in_both_formats(tmp_path, overlap):
    model, _ = model_and_config()
    evaluation = model.interpolation_options.evaluation_options
    evaluation.octree_refinement_mode = gp.data.OctreeRefinementMode.PRECISE
    evaluation.deduplicate_octree_corners = True
    evaluation.triangulation_method = gp.data.TriangulationMethod.QUADS
    evaluation.mesh_extraction_extent_capping = gp.data.MeshExtentCapping.SCALAR_LESS_EQUAL
    evaluation.mesh_extraction_overlap = overlap
    evaluation.octree_curvature_threshold = 0.2
    evaluation.octree_error_threshold = 0.7
    evaluation.evaluation_chunk_size = 2000
    evaluation.number_octree_levels = 3
    expected = model.interpolation_options.model_dump(mode='json')['evaluation_options']
    restored = _load_model_from_bytes(model_to_bytes(model))
    assert restored.interpolation_options.model_dump(mode='json')['evaluation_options'] == expected
    path = tmp_path / 'model.json'
    JsonIO.save_model_to_json(model, str(path))
    restored = JsonIO.load_model_from_json(str(path))
    assert restored.interpolation_options.model_dump(mode='json')['evaluation_options'] == expected

    data = json.loads(path.read_text())
    del data['interpolation_options']['evaluation_options']
    data['interpolation_options']['mesh_extraction'] = False
    path.write_text(json.dumps(data))
    assert not JsonIO.load_model_from_json(str(path)).interpolation_options.evaluation_options.mesh_extraction


@pytest.mark.parametrize('mode', ['none', 'pretty', 'watertight', 'joint', 'joint_contacts', 'quads', 'capped'])
def test_mesh_modes_and_clearing_previous_outputs(mode):
    model, config = model_and_config()
    evaluation = model.interpolation_options.evaluation_options
    if mode == 'quads':
        evaluation.triangulation_method = gp.data.TriangulationMethod.QUADS
        evaluation.deduplicate_octree_corners = True
        evaluation.octree_refinement_mode = gp.data.OctreeRefinementMode.BALANCED
    elif mode == 'capped':
        evaluation.mesh_extraction_extent_capping = gp.data.MeshExtentCapping.SCALAR_LESS_EQUAL
    else:
        evaluation.mesh_extraction_overlap = mode
    solutions = gp.compute_model(model, config)
    assert len(solutions.dc_meshes) == 2
    for mesh in solutions.dc_meshes:
        element = next(e for e in model.structural_frame.structural_groups[mesh.stack_index].elements
                       if np.isclose(e.scalar_field_at_interface, mesh.isovalue))
        assert mesh.edges.shape[1] == 3
        assert len(mesh.edges) > 0
        assert element.mesh is not mesh
        np.testing.assert_allclose(element.vertices, model.grid.transform.apply_inverse_with_cached_pivot(
            model.input_transform.apply_inverse(mesh.vertices)))
        assert element.mesh.contact_report is mesh.contact_report
        if mode.startswith('joint'):
            np.testing.assert_array_equal(element.mesh.joint_vertex_ids, mesh.joint_vertex_ids)
        if mode == 'capped':
            assert element.mesh.capping_report is not None
    evaluation.mesh_extraction = False
    gp.compute_model(model, config)
    assert all(e.mesh is None and e.vertices is None and e.edges is None
               for e in model.structural_frame.structural_elements_filtered[:-1])


def test_mesh_mapping_uses_metadata_before_reordering_and_skips_null_space():
    model, _ = model_and_config()
    original = model.structural_frame.structural_groups[0].elements
    null_element, exported_element = original
    model.structural_frame.structural_groups = [
        gp.data.StructuralGroup(name='null', elements=[null_element], structural_relation=gp.data.StackRelationType.NULL_SPACE),
        gp.data.StructuralGroup(name='layers', elements=[exported_element], structural_relation=gp.data.StackRelationType.ERODE),
    ]
    mesh = DualContouringMesh(np.zeros((3, 3)), np.array([[0, 1, 2]]), stack_index=1, surface_index=0)
    model.solutions = SimpleNamespace(micro_point_results=None, scalar_field_at_surface_points=[10., 20.],
                                     dc_meshes=[mesh], _ordered_elements=[[0], [0]])
    assert null_element.scalar_field_at_interface == 10.
    assert null_element.mesh is None
    assert exported_element.scalar_field_at_interface == 20.
    assert exported_element.mesh is not None


@pytest.mark.parametrize('backend', [gp.data.AvailableBackends.numpy, gp.data.AvailableBackends.PYTORCH])
def test_fixed_fields_use_compute_time_transforms_options_and_surface_order(monkeypatch, backend):
    if backend == gp.data.AvailableBackends.PYTORCH:
        pytest.importorskip('torch')
    model, config = model_and_config()
    config.backend = backend
    model.interpolation_options.evaluation_options.mesh_extraction = False
    # Reverse input order so the solutions setter reorders it after computing.
    model.structural_frame.structural_groups[0].elements.reverse()
    model.grid.transform.rotation = np.array([0., 0., 25.])
    model.grid.transform.scale = np.array([1.4, 0.8, 1.2])
    model.input_transform.rotation = np.array([10., 0., -15.])
    points = model.surface_points_copy.xyz.copy()
    solution = gp.compute_model(model, config)
    reference = np.repeat(solution.scalar_field_at_surface_points, 6)[None, :]

    def no_compute(*args, **kwargs):
        raise AssertionError('Field queries must not recompute the model')

    import gempy_engine
    monkeypatch.setattr(gempy_engine, 'compute_model', no_compute)
    model.input_transform.scale *= 2
    model.grid.transform.rotation[:] = 0
    model.interpolation_options.kernel_options.range = 99.
    evaluator = gp.prepare_field_evaluator(model)
    np.testing.assert_allclose(evaluator(points), reference, atol=1e-7)
    np.testing.assert_allclose(gp.evaluate_fields(model, points), reference, atol=1e-7)
    assert evaluator.fault_values(points).shape == (0, len(points))
    assert evaluator.classify_units(points)[0].shape == (len(points),)
    with pytest.raises(ValueError, match='shape'):
        evaluator([1., 2., 3.])


def test_field_evaluation_requires_computation():
    model, _ = model_and_config()
    with pytest.raises(ValueError, match='Compute the model'):
        gp.prepare_field_evaluator(model)


def test_fault_side_queries_match_engine_in_transformed_coordinates():
    import gempy_engine
    model = gp.generate_example_model(ExampleModel.ONE_FAULT, compute_model=False)
    evaluation = model.interpolation_options.evaluation_options
    evaluation.number_octree_levels = 2
    evaluation.number_octree_levels_surface = 2
    evaluation.mesh_extraction = False
    model.grid.transform.rotation = np.array([0., 0., 20.])
    config = gp.data.GemPyEngineConfig(backend=gp.data.AvailableBackends.numpy, dtype='float64')
    gp.compute_model(model, config)
    evaluator = gp.prepare_field_evaluator(model)
    points = np.array([[300., 400., 450.], [650., 550., 600.]])
    context = model._field_evaluation_context
    reference = gempy_engine.prepare_field_evaluator(
        model.solutions, context.interpolation_input, context.descriptor, context.options)
    transformed = model.input_transform.apply(model.grid.transform.apply_with_cached_pivot(points))
    assert evaluator.fault_stacks == (0,)
    for sides in (None, np.array([[-1, 1]]), np.array([[1, -1]])):
        np.testing.assert_allclose(evaluator(points, fault_sides=sides), reference(transformed, fault_sides=sides))
    np.testing.assert_allclose(evaluator.fault_values(points), reference.fault_values(transformed))


def test_joint_rejects_capping_and_stale_backend_is_rejected():
    model, config = model_and_config()
    evaluation = model.interpolation_options.evaluation_options
    evaluation.mesh_extraction_overlap = 'joint'
    evaluation.mesh_extraction_extent_capping = gp.data.MeshExtentCapping.SCALAR_LESS_EQUAL
    with pytest.raises((ValueError, NotImplementedError), match='capping'):
        gp.compute_model(model, config)
    evaluation.mesh_extraction_extent_capping = gp.data.MeshExtentCapping.NONE
    gp.compute_model(model, config)
    BackendTensor._change_backend(gp.data.AvailableBackends.numpy, dtype='float32')
    with pytest.raises(RuntimeError, match='backend changed'):
        gp.prepare_field_evaluator(model)
