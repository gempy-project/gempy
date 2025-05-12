import pprint

import gempy as gp
from gempy.core.data.enumerators import ExampleModel
from gempy_engine.core.data import InterpolationOptions

from approvaltests import Options, verify

from verify_helper import verify_json


def test_generate_horizontal_stratigraphic_model():
    model: gp.data.GeoModel = gp.generate_example_model(ExampleModel.HORIZONTAL_STRAT, compute_model=False)

    json = model.model_dump(mode='json')
    # Pretty print json
    pprint.pp(json)
    
    # write json to disk
    # with open("horizontal_stratigraphic_model.json", "w") as f:
    #     f.write(json)
    # 
    # # Validate json against schema
    # verify_json(json, name="Horizontal Stratigraphic Model serialization")
    
    

def test_interpolation_options():
    options = InterpolationOptions.from_args(
        range=1.7,
        c_o=10.
    )
    json = options.model_dump(mode="json")
    # Pretty print json
    pprint.pp(json)
    
    # Deserialize with pydantic
    options2 = InterpolationOptions.model_validate(json)
    assert options.__str__() == options2.__str__()
