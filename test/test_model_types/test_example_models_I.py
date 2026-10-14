import pytest
import numpy as np
from gempy_viewer import GemPyToVista

import gempy as gp
from gempy.core.data.enumerators import ExampleModel
from gempy.optional_dependencies import require_gempy_viewer
from gempy_engine.core.data.interp_output import InterpOutput
from test.conftest import TEST_SPEED, TestSpeed
from test.verify_helper import gempy_verify_array

# ! When importing the model is computed

# TODO []: Use the generator and do some approval testing
PLOT = True

pytestmark = pytest.mark.skipif(TEST_SPEED.value < TestSpeed.MINUTES.value, reason="Global test speed below this test value.")


def _verify_scalar_field(model, name):
    outputs_centers_: InterpOutput = model.solutions.octrees_output[-1].outputs[0]
    scalar_field = outputs_centers_.exported_fields.scalar_field
    scalar_field = scalar_field[::int(len(scalar_field) / 50)]  # Pick 50 values from the scalar field array
    gempy_verify_array(scalar_field, name)


def test_generate_horizontal_stratigraphic_model():
    model = gp.generate_example_model(ExampleModel.HORIZONTAL_STRAT, compute_model=True)
    print(model.structural_frame)

    if PLOT:
        gpv = require_gempy_viewer()
        gpv.plot_3d(model, show_data=True, image=True)

    if False:
        _verify_scalar_field(
            model=model,
            name="Horizontal Stratigraphic Scalar Field"
        )


def test_generate_fold_model():
    model = gp.generate_example_model(ExampleModel.ANTICLINE, compute_model=True)
    print(model.structural_frame)

    if PLOT or False:
        gpv = require_gempy_viewer()
        gpv.plot_3d(model, image=True)

    _verify_scalar_field(
        model=model,
        name="Anticline Scalar Field"
    )


def test_generate_fault_model():
    model = gp.generate_example_model(ExampleModel.ONE_FAULT, compute_model=True)
    print(model.structural_frame)

    if PLOT:
        gpv = require_gempy_viewer()
        gpv.plot_3d(model, image=True)

    _verify_scalar_field(
        model=model,
        name="Fault Scalar Field"
    )


def test_generate_combination_model():

    model = gp.generate_example_model(ExampleModel.COMBINATION, compute_model=False)
    # Adaptive leaf counts/order may change without changing the interpolant.
    # Compare every stack at fixed world coordinates instead of striding leaves.
    axis = np.array([50., 500., 950.])
    points = np.array(np.meshgrid([125., 1250., 2375.], axis, axis, indexing='ij')).reshape(3, -1).T
    gp.set_custom_grid(model.grid, points)
    gp.compute_model(model)
    print(model.structural_frame)

    if PLOT:
        gpv = require_gempy_viewer()
        gpv.plot_2d(model)
        pd3 = gpv.plot_3d(
            model,
            image=True,
            show_lith=True,
            show_octree=True,
            show=True
        )

    root = model.solutions.root_output
    scalar_fields = np.stack([
        output.exported_fields.scalar_field[root.grid.custom_grid_slice]
        for output in root.outputs
    ])
    gempy_verify_array(scalar_fields, "Combination Scalar Field")


def test_generate_greenstone_model():
    model = gp.generate_example_model(ExampleModel.GREENSTONE, compute_model=True)
    print(model.structural_frame)

    if PLOT:
        gpv = require_gempy_viewer()
        gpv.plot_3d(model, image=True)
