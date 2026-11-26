"""GeoModel.validate() names the group or surface for input the engine would reject by index."""
import numpy as np
import pytest

import gempy as gp
from gempy.core.data.enumerators import ExampleModel
from gempy.core.data.orientations import OrientationsTable
from gempy.core.data.structural_element import StructuralElement
from gempy.core.data.structural_group import FaultsRelationSpecialCase
from gempy.core.data.surface_points import SurfacePointsTable
from gempy_engine.core.data.interpolation_functions import CustomInterpolationFunctions
from gempy_engine.core.data.stack_relation_type import StackRelationType

X, Y = np.array([100., 900., 100., 900.]), np.array([100., 100., 900., 900.])


def _element(name, height, points=True, orientations=True):
    return StructuralElement(
        name=name, color="#015482",
        surface_points=(SurfacePointsTable.from_arrays(X, Y, np.full(4, height), names=name) if points
                        else SurfacePointsTable(data=np.zeros(0, dtype=SurfacePointsTable.dt))),
        orientations=(OrientationsTable.from_arrays(np.array([500.]), np.array([500.]), np.array([height]),
                                                    np.zeros(1), np.zeros(1), np.ones(1), names=name) if orientations
                      else OrientationsTable(data=np.zeros(0, dtype=OrientationsTable.dt))),
    )


def _model():
    return gp.generate_example_model(ExampleModel.HORIZONTAL_STRAT, compute_model=False)


def _raises(model, reason):
    with pytest.raises(gp.data.ModelValidationError) as error:
        model.validate()
    assert error.value.reason == reason
    assert error.value.message in str(error.value)
    return error.value


def test_the_example_model_is_valid():
    _model().validate()


def test_a_surface_without_points_is_named():
    model = _model()
    group = model.structural_frame.structural_groups[0]
    group.elements.append(_element("new_surface", 0., points=False, orientations=False))
    error = _raises(model, "surface_without_points")
    assert error.context == {"group_name": group.name, "surface_name": "new_surface"}
    assert error.field == f"structural_groups[0].elements[{len(group.elements) - 1}]"
    assert "'new_surface'" in error.message and f"'{group.name}'" in error.message


def test_a_group_without_orientations_is_named():
    model = _model()
    group = gp.data.StructuralGroup(name="Cover", elements=[_element("cover_top", 900., orientations=False)],
                                    structural_relation=StackRelationType.ERODE)
    model.structural_frame.insert_group(0, group)
    error = _raises(model, "missing_orientations")
    assert error.context == {"group_name": "Cover"}


def test_compute_model_reports_the_group_before_the_engine():
    model = _model()
    model.structural_frame.insert_group(0, gp.data.StructuralGroup(
        name="Cover", elements=[_element("cover_top", 900., orientations=False)],
        structural_relation=StackRelationType.ERODE))
    with pytest.raises(gp.data.ModelValidationError, match="Group 'Cover' has no orientations"):
        gp.compute_model(model, engine_config=gp.data.GemPyEngineConfig(backend=gp.data.AvailableBackends.numpy))


def test_a_custom_interpolation_group_needs_no_input_data():
    model = _model()
    group = gp.data.StructuralGroup(name="External", elements=[_element("external_top", 900., points=False,
                                                                        orientations=False)],
                                    structural_relation=StackRelationType.ERODE)
    group.custom_interpolation = CustomInterpolationFunctions(scalar_field_at_surface_points=np.zeros(1),
                                                              implicit_function=lambda xyz, extent: xyz[:, 2])
    model.structural_frame.insert_group(0, group)
    model.validate()


def _faulted():
    model = _model()
    fault = gp.data.StructuralGroup(name="Fault1", elements=[_element("fault1", 500.)],
                                    structural_relation=StackRelationType.FAULT)
    model.structural_frame.insert_group(0, fault)
    return model, fault


def test_a_fault_offsetting_all_groups_below_is_valid():
    model, fault = _faulted()
    fault.fault_relations = FaultsRelationSpecialCase.OFFSET_ALL
    model.validate()


def test_a_fault_offsetting_an_older_group_is_named():
    model, fault = _faulted()
    above = gp.data.StructuralGroup(name="Cover", elements=[_element("cover_top", 900.)],
                                    structural_relation=StackRelationType.ERODE)
    model.structural_frame.insert_group(0, above)
    fault.fault_relations = [above]
    error = _raises(model, "fault_offsets_older_group")
    assert error.context == {"fault_name": "Fault1", "group_name": "Cover"}
    # Callers match the earlier ValueError text.
    assert "Fault Fault1 cannot affect older fault Cover" in str(error)


@pytest.mark.parametrize("relations", [None, []])
def test_a_fault_without_a_valid_offset_setting_is_named(relations):
    model, fault = _faulted()
    fault.fault_relations = relations
    error = _raises(model, "invalid_fault_relation")
    assert error.context == {"fault_name": "Fault1"}
    assert "Fault Fault1 has an invalid fault relation" in str(error)
