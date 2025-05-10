import pprint

import gempy as gp
from gempy.core.data.enumerators import ExampleModel
from gempy_engine.core.data import InterpolationOptions


def test_generate_horizontal_stratigraphic_model():
    model: gp.data.GeoModel = gp.generate_example_model(ExampleModel.HORIZONTAL_STRAT, compute_model=False)
    print(model.structural_frame)
    options: InterpolationOptions = model.interpolation_options
    pass

def test_interpolation_options():
    options = InterpolationOptions.from_args(
        range=1.7,
        c_o=10.
    )
    json = options.model_dump()
    # Pretty print json
    pprint.pp(json)
    
    # Deserialize with pydantic
    options2 = InterpolationOptions.model_validate(json)
    assert options.__str__() == options2.__str__()
