import os

import pprint
from pydantic_core import from_json

import gempy as gp
from gempy.core.data.enumerators import ExampleModel
from gempy_engine.core.data import InterpolationOptions

from approvaltests import Options, verify

from verify_helper import verify_json


def test_generate_horizontal_stratigraphic_model():
    model: gp.data.GeoModel = gp.generate_example_model(ExampleModel.HORIZONTAL_STRAT, compute_model=False)

    model_json = model.model_dump_json(by_alias=True)
    # Pretty print JSON
    pprint.pp(model_json)
    

    # Ensure the 'verify/' directory exists
    os.makedirs("verify", exist_ok=True)

    # Write the JSON to disk
    file_path = os.path.join("temp", "horizontal_stratigraphic_model.json")
    with open(file_path, "w") as f:
        f.write(model_json)

    if False: # * Use this to debug which fields are giving problems
        # model_deserialized = gp.data.GeoModel.model_validate(from_json(model_json, allow_partial=True))
        pass
    else:
        model_deserialized = gp.data.GeoModel.model_validate_json(model_json)
        
    # TODO: [ ] Structural frame?
    # TODO: [ ] Input transform?
    assert model_deserialized.__str__() == model.__str__()
    
    # # Validate json against schema
    if False:
        verify_json(model_json, name="verify/Horizontal Stratigraphic Model serialization")


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
