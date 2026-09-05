import numpy as np
import pytest

import gempy as gp
from gempy.core.data.enumerators import ExampleModel
from gempy.modules.data_manipulation import input_data_descriptor_from_geo_model


def test_finite_fault_is_transformed_for_engine_without_mutating_model():
    model = gp.generate_example_model(ExampleModel.ONE_FAULT, compute_model=False)
    fault_group = model.structural_frame.structural_groups[0]
    finite_fault = gp.data.FiniteFault(
        center=(500.0, 400.0, 300.0),
        strike_radius=(200.0, 100.0),
        dip_radius=150.0,
        taper=gp.data.TaperType.QUADRATIC,
    )
    fault_group.set_finite_fault(finite_fault)

    descriptor = input_data_descriptor_from_geo_model(model)
    transformed = descriptor.stack_structure.faults_input_data[0].finite_fault

    expected_center = model.grid.transform.apply_with_cached_pivot(np.atleast_2d(finite_fault.center))
    expected_center = model.input_transform.apply(expected_center)[0]
    scale = float((model.input_transform.scale * model.grid.transform.scale)[0])
    assert transformed.center == pytest.approx(expected_center)
    assert transformed.strike_radius == pytest.approx((200.0 * scale, 100.0 * scale))
    assert transformed.dip_radius == pytest.approx(150.0 * scale)
    assert transformed.taper is finite_fault.taper
    assert fault_group.faults_input_data.finite_fault is finite_fault


def test_finite_fault_rejects_anisotropic_engine_transform():
    model = gp.generate_example_model(ExampleModel.ONE_FAULT, compute_model=False)
    model.structural_frame.structural_groups[0].set_finite_fault(
        gp.data.FiniteFault(center=(500.0, 400.0, 300.0))
    )
    model.input_transform.scale = np.array([1.0, 2.0, 1.0])

    with pytest.raises(ValueError, match="isotropic model transform"):
        input_data_descriptor_from_geo_model(model)
