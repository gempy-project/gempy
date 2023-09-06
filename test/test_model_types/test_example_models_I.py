import gempy as gp
from gempy.core.data.enumerators import ExampleModel
from gempy.optional_dependencies import require_gempy_viewer

from test.verify_helper import gempy_verify_array
import pytest
from test.conftest import TEST_SPEED, TestSpeed
# ! When importing the model is computed

# TODO []: Use the generator and do some approval testing
PLOT = True

pytestmark = pytest.mark.skipif(TEST_SPEED.value < TestSpeed.MINUTES.value, reason="Global test speed below this test value.")


def _verify_scalar_field(model, name):
    scalar_field = model.solutions.octrees_output[-1].outputs_centers[0].exported_fields.scalar_field
    scalar_field = scalar_field[::int(len(scalar_field) / 50)] # Pick 50 values from the scalar field array
    gempy_verify_array(scalar_field, name)



def test_generate_horizontal_stratigraphic_model():
    model = gp.generate_example_model(ExampleModel.HORIZONTAL_STRAT, compute_model=True)
    print(model.structural_frame)

    _verify_scalar_field(
        model=model,
        name="Horizontal Stratigraphic Scalar Field"
    )

    if PLOT: 
        gpv = require_gempy_viewer()
        gpv.plot_3d(model, image=False)



def test_generate_fold_model():
    model = gp.generate_example_model(ExampleModel.ANTICLINE, compute_model=True)
    print(model.structural_frame)
    
    _verify_scalar_field(
        model=model,
        name="Anticline Scalar Field"
    )
    
    if PLOT:
        gpv = require_gempy_viewer()
        gpv.plot_3d(model, image=True)



def test_generate_fault_model():
    model = gp.generate_example_model(ExampleModel.ONE_FAULT, compute_model=True)
    print(model.structural_frame)
    
    _verify_scalar_field(
        model=model,
        name="Fault Scalar Field"
    )
    
    if PLOT:
        gpv = require_gempy_viewer()
        gpv.plot_3d(model, image=True)



def test_generate_combination_model():
    model = gp.generate_example_model(ExampleModel.COMBINATION, compute_model=True)
    print(model.structural_frame)
    
    _verify_scalar_field(
        model=model,
        name="Combination Scalar Field"
    )
    
    if PLOT:
        gpv = require_gempy_viewer()
        gpv.plot_3d(model, image=True)
