"""Null space as a flag on any group, groups backed by a mesh, and named custom-grid segments."""
import io
import json
import warnings
import zipfile

import numpy as np
import pytest

import gempy as gp
from gempy.core.data.enumerators import ExampleModel
from gempy.core.data.orientations import OrientationsTable
from gempy.core.data.structural_element import StructuralElement
from gempy.core.data.surface_points import SurfacePointsTable
from gempy.modules import mesh_fields
from gempy.modules.serialization.save_load import _load_model_from_bytes, _validate_serialization, model_to_bytes
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.interpolation_functions import CustomInterpolationFunctions
from gempy_engine.core.data.stack_relation_type import StackRelationType

COLUMN = np.column_stack([np.full(50, 500.), np.full(50, 500.), np.arange(10., 1000., 20.)])
NUMPY = gp.data.GemPyEngineConfig(backend=gp.data.AvailableBackends.numpy, dtype="float64")


@pytest.fixture(autouse=True)
def no_backend():
    """Each test registers the backend it needs."""
    previous = mesh_fields.mesh_field_backend()
    mesh_fields.register_mesh_field_backend(None)
    yield
    mesh_fields.register_mesh_field_backend(previous)


def _model():
    """rock2 above 600, rock1 between 400 and 600, basement below; the custom grid is a vertical column."""
    model = gp.generate_example_model(ExampleModel.HORIZONTAL_STRAT, compute_model=False)
    gp.set_custom_grid(model.grid, COLUMN)
    return model


def _kriged_group(name, height, relation=StackRelationType.ERODE, null_space=False):
    x, y = np.array([100., 900., 100., 900.]), np.array([100., 100., 900., 900.])
    element = StructuralElement(
        name=f"{name}_surface", color="#015482",
        surface_points=SurfacePointsTable.from_arrays(x, y, np.full(4, height), names=f"{name}_surface"),
        orientations=OrientationsTable.from_arrays(np.array([500.]), np.array([500.]), np.array([height]),
                                                   np.zeros(1), np.zeros(1), np.ones(1), names=f"{name}_surface"),
    )
    return gp.data.StructuralGroup(name=name, elements=[element], structural_relation=relation, null_space=null_space)


def _column(model):
    """Element name (or ``void``) along the column, bottom up, and where it changes."""
    solutions = gp.compute_model(model, engine_config=NUMPY)
    ids = np.rint(BackendTensor.t.to_numpy(solutions.raw_arrays.custom)).astype(int)
    names = [e.name for e in model.structural_frame.structural_elements_filtered]
    labels = ["void" if i == -1 else names[i - 1] for i in ids]
    return [(round(float((COLUMN[k, 2] + COLUMN[k + 1, 2])/2)), labels[k], labels[k + 1])
            for k in range(len(labels) - 1) if labels[k] != labels[k + 1]]


def _square(height):
    """A horizontal square mesh over the extent, normals up."""
    vertices = np.array([[-50., -50., height], [1050., -50., height], [1050., 1050., height], [-50., 1050., height]])
    return vertices, np.array([[0, 1, 2], [0, 2, 3]])


def _plane_backend(calls):
    """A stand-in for the server's Warp backend: the signed distance to the mesh's (horizontal) plane."""
    def backend(vertices, faces, invert):
        calls.append(vertices)
        height, sign = float(np.mean(vertices[:, 2])), -1. if invert else 1.
        zeros = lambda xyz: BackendTensor.t.zeros(len(xyz), dtype=BackendTensor.dtype_obj)
        return CustomInterpolationFunctions(
            np.array([0.0]), lambda xyz: sign*(xyz[:, 2] - height), zeros, zeros,
            lambda xyz: BackendTensor.t.full((len(xyz),), sign, dtype=BackendTensor.dtype_obj))
    return backend


LAYERS = [(400, "basement", "rock1"), (600, "rock1", "rock2")]


def test_a_kriged_eroding_null_group_voids_above_its_surface_and_takes_no_ids():
    model = _model()
    model.structural_frame.insert_group(0, _kriged_group("air", 800., null_space=True))
    assert _column(model) == LAYERS + [(800, "rock2", "void")]


def test_an_onlapping_null_group_voids_from_the_older_top():
    model = _model()
    model.structural_frame.insert_group(0, _kriged_group("air", 800., relation=StackRelationType.ONLAP, null_space=True))
    assert _column(model) == [(400, "basement", "rock1"), (600, "rock1", "void")]


def test_a_legacy_null_space_relation_loads_as_an_eroding_null_group():
    model = _model()
    model.structural_frame.insert_group(0, _kriged_group("air", 800., null_space=True))
    archive = zipfile.ZipFile(io.BytesIO(model_to_bytes(model)))
    header = json.loads(archive.read("header.json"))
    group = header["structural_frame"]["structural_groups"][0]
    assert (group["structural_relation"], group["null_space"]) == (StackRelationType.ERODE.value, True)
    group["structural_relation"], _ = StackRelationType.NULL_SPACE.value, group.pop("null_space")
    rewritten = io.BytesIO()
    with zipfile.ZipFile(rewritten, "w") as out:
        for name in archive.namelist():
            out.writestr(name, json.dumps(header) if name == "header.json" else archive.read(name))
    loaded = _load_model_from_bytes(rewritten.getvalue()).structural_frame.structural_groups[0]
    assert (loaded.structural_relation, loaded.null_space) == (StackRelationType.ERODE, True)


@pytest.mark.parametrize("null_space, expected", [
    (True, LAYERS + [(800, "rock2", "void")]),
    (False, LAYERS + [(800, "rock2", "lid")]),
])
def test_a_mesh_group_takes_its_surface_from_the_mesh(null_space, expected):
    calls = []
    mesh_fields.register_mesh_field_backend(_plane_backend(calls))
    model = _model()
    group = gp.add_mesh_group(model, "lid", "le://space/static_mesh/lid", null_space=null_space)
    group.mesh_source.load(*_square(800.))
    assert _column(model) == expected
    # Built once per mesh, transform and backend.
    _column(model)
    assert len(calls) == 1
    np.testing.assert_allclose(calls[0], model.input_transform.apply(_square(800.)[0]))


def test_an_inverted_mesh_group_claims_the_other_side():
    mesh_fields.register_mesh_field_backend(_plane_backend([]))
    model = _model()
    gp.add_mesh_group(model, "lid", gp.data.MeshScalarField("le://space/static_mesh/lid", invert=True),
                      null_space=True).mesh_source.load(*_square(800.))
    # The void is below the plane; nothing older shows there, and above it the older groups remain.
    assert _column(model) == [(800, "void", "rock2")]


@pytest.mark.parametrize("loaded", [False, True])
def test_without_a_backend_or_mesh_an_eroding_mesh_group_is_left_out(loaded):
    if loaded:
        mesh_fields.register_mesh_field_backend(None)
    else:
        mesh_fields.register_mesh_field_backend(_plane_backend([]))
    model = _model()
    group = gp.add_mesh_group(model, "lid", "le://space/static_mesh/lid", null_space=True)
    if loaded:
        group.mesh_source.load(*_square(800.))
    with pytest.warns(mesh_fields.MeshFieldUnavailableWarning, match="lid"):
        assert _column(model) == LAYERS


def test_without_a_backend_an_onlapping_mesh_group_raises():
    model = _model()
    gp.add_mesh_group(model, "lid", "le://space/static_mesh/lid", null_space=True,
                      structural_relation=StackRelationType.ONLAP).mesh_source.load(*_square(800.))
    with pytest.raises(mesh_fields.MeshFieldUnavailableError, match="lid"):
        gp.compute_model(model, engine_config=NUMPY)


def test_mesh_groups_and_null_flags_round_trip_without_the_mesh():
    model = _model()
    gp.add_mesh_group(model, "lid", gp.data.MeshScalarField("le://space/static_mesh/lid", invert=True, sha256="ab"),
                      null_space=True).mesh_source.load(*_square(800.))
    loaded = _load_model_from_bytes(model_to_bytes(model))
    _validate_serialization(model, loaded)
    group = loaded.structural_frame.structural_groups[0]
    assert group.null_space and group.structural_relation is StackRelationType.ERODE
    assert (group.mesh_source.uri, group.mesh_source.invert, group.mesh_source.sha256) == (
        "le://space/static_mesh/lid", True, "ab")
    assert not group.mesh_source.is_loaded


def test_a_null_group_of_several_surfaces_voids_above_its_lowest():
    model = _model()
    group = _kriged_group("air", 850., null_space=True)
    group.elements.append(_kriged_group("air_base", 750.).elements[0])
    model.structural_frame.insert_group(0, group)
    assert _column(model) == LAYERS + [(760, "rock2", "void")]


#: A null bottom group with its lowest surface at 200: the basement under it, void up to rock1.
BOTTOM = [(200, "basement", "void"), (400, "void", "rock1"), (600, "rock1", "rock2")]


def test_a_null_bottom_group_voids_above_its_lowest_surface_over_the_basement():
    model = _model()
    group = _kriged_group("deep", 300., null_space=True)
    group.elements.append(_kriged_group("deep_base", 200.).elements[0])
    model.structural_frame.append_group(group)
    assert _column(model) == BOTTOM


@pytest.mark.parametrize("ignored, expected", [((), BOTTOM), (("custom_grid",), LAYERS)])
def test_a_null_bottom_mesh_group_leaves_the_grids_it_ignores_to_the_basement(ignored, expected):
    mesh_fields.register_mesh_field_backend(_plane_backend([]))
    model = _model()
    group = gp.add_mesh_group(model, "floor", "le://space/static_mesh/floor", null_space=True,
                              group_index=len(model.structural_frame.structural_groups))
    group.mesh_source.load(*_square(200.))
    group.ignored_grid_types = ignored
    assert _column(model) == expected


def test_without_a_backend_a_null_bottom_mesh_group_is_left_out_even_when_it_onlaps():
    model = _model()
    gp.add_mesh_group(model, "floor", "le://space/static_mesh/floor", null_space=True,
                      structural_relation=StackRelationType.ONLAP,
                      group_index=len(model.structural_frame.structural_groups)).mesh_source.load(*_square(200.))
    with pytest.warns(mesh_fields.MeshFieldUnavailableWarning, match="floor"):
        assert _column(model) == LAYERS


def test_a_null_group_above_the_last_one_must_erode_or_onlap():
    model = _model()
    model.structural_frame.insert_group(0, _kriged_group("air", 800., relation=StackRelationType.FAULT, null_space=True))
    with pytest.raises(gp.data.ModelValidationError) as error:
        model.validate()
    assert error.value.reason == "null_space_group_relation"


@pytest.mark.parametrize("change, reason", [
    (lambda m, g: g.elements.append(_kriged_group("x", 900.).elements[0]), "mesh_group_elements"),
    (lambda m, g: setattr(g, "structural_relation", StackRelationType.FAULT), "mesh_group_relation"),
    (lambda m, g: m.structural_frame.structural_groups.reverse(), "mesh_group_relation"),
])
def test_mesh_groups_have_one_surface_and_only_a_null_one_is_last(change, reason):
    model = _model()
    group = gp.add_mesh_group(model, "lid", "le://space/static_mesh/lid")
    change(model, group)
    with pytest.raises(gp.data.ModelValidationError) as error:
        model.validate()
    assert error.value.reason == reason


def test_a_mesh_group_has_no_input_data():
    model = _model()
    group = _kriged_group("lid", 800.)
    group.mesh_source = gp.data.MeshScalarField("le://space/static_mesh/lid")
    model.structural_frame.insert_group(0, group)
    with pytest.raises(gp.data.ModelValidationError) as error:
        model.validate()
    assert error.value.reason == "mesh_group_input_data"


def test_custom_grid_segments_split_results_and_round_trip():
    model = gp.generate_example_model(ExampleModel.HORIZONTAL_STRAT, compute_model=False)
    gp.set_custom_grid(model.grid, {"borehole": COLUMN[:30], "mesh": COLUMN[30:40]})
    gp.add_custom_grid_segment(model.grid, "cloud", COLUMN[40:])
    custom = model.grid.custom_grid
    assert custom.segments == [("borehole", 30), ("mesh", 10), ("cloud", 10)]
    assert custom.segment_slice("cloud") == slice(40, 50)
    np.testing.assert_array_equal(custom.values, COLUMN)
    with pytest.raises(ValueError, match="already"):
        gp.add_custom_grid_segment(model.grid, "mesh", COLUMN[:1])

    loaded = _load_model_from_bytes(model_to_bytes(model))
    assert loaded.grid.custom_grid.segments == custom.segments
    np.testing.assert_array_equal(loaded.grid.custom_grid.values, COLUMN)

    solutions = gp.compute_model(model, engine_config=NUMPY)
    split = custom.split(BackendTensor.t.to_numpy(solutions.raw_arrays.custom))
    assert [len(values) for values in split.values()] == [30, 10, 10]


def test_a_null_group_can_leave_one_segment_out():
    model = _model()
    gp.set_custom_grid(model.grid, {"topography": COLUMN[40:45], "borehole": COLUMN[45:]})
    group = _kriged_group("air", 800., null_space=True)
    group.ignored_grid_types = ("custom_grid/topography",)
    model.structural_frame.insert_group(0, group)
    solutions = gp.compute_model(model, engine_config=NUMPY)
    split = model.grid.custom_grid.split(np.rint(BackendTensor.t.to_numpy(solutions.raw_arrays.custom)).astype(int))
    np.testing.assert_array_equal(split["topography"], [1, 1, 1, 1, 1])  # rock2: the geology below the void
    np.testing.assert_array_equal(split["borehole"], [-1, -1, -1, -1, -1])


def test_the_null_flag_round_trips_through_json(tmp_path):
    from gempy.modules.json_io import JsonIO

    model = _model()
    model.structural_frame.insert_group(0, _kriged_group("air", 800., relation=StackRelationType.ONLAP, null_space=True))
    path = tmp_path/"model.json"
    JsonIO.save_model_to_json(model, str(path))
    restored = JsonIO.load_model_from_json(str(path)).structural_frame.get_group_by_name("air")
    assert (restored.structural_relation, restored.null_space) == (StackRelationType.ONLAP, True)
