import gempy as gp
from gempy.core.data.enumerators import ExampleModel
from gempy.optional_dependencies import require_gempy_viewer

from test.verify_helper import gempy_verify_array

# ! When importing the model is computed

# TODO []: Use the generator and do some approval testing
PLOT = True


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
        gpv.plot_3d(model, image=True)




def test_generate_fold_model():
    from  examples.examples.geometries.b02_fold import generate_anticline_model
    model = generate_anticline_model()
    assert isinstance(model, gp.data.GeoModel)


def test_generate_recumbent_fold_model():
    from examples.examples.geometries.c03_recumbent_fold import generate_recumbent_fold_model
    model = generate_recumbent_fold_model()
    assert isinstance(model, gp.data.GeoModel)


def test_generate_pinchout_model():
    from examples.examples.geometries.d04_pinchout import generate_pinchout_model
    model = generate_pinchout_model()
    assert isinstance(model, gp.data.GeoModel)


def test_generate_fault_model():
    from examples.examples.geometries.e05_fault import generate_fault_model
    model = generate_fault_model()
    assert isinstance(model, gp.data.GeoModel)


def test_generate_unconformity_model():
    from examples.examples.geometries.f06_unconformity import generate_unconformity_model
    model = generate_unconformity_model()
    assert isinstance(model, gp.data.GeoModel)


def test_generate_combination_model():
    from examples.examples.geometries.g07_combination import generate_combination_model
    model = generate_combination_model()
    assert isinstance(model, gp.data.GeoModel)
