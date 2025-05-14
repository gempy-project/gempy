import json
import numpy as np

import os
import pprint

import gempy as gp
from gempy.core.data.encoders.converters import loading_model_injection
from gempy.core.data.enumerators import ExampleModel
from gempy_engine.core.data import InterpolationOptions
from verify_helper import verify_json, verify_json_ignoring_dates


def test_generate_horizontal_stratigraphic_model():
    model: gp.data.GeoModel = gp.generate_example_model(ExampleModel.HORIZONTAL_STRAT, compute_model=False)
    model_json = model.model_dump_json(by_alias=True, indent=4, exclude={"*data"})

    # Write the JSON to disk
    file_path = os.path.join("temp", "horizontal_stratigraphic_model.json")
    with open(file_path, "w") as f:
        f.write(model_json)

    with loading_model_injection(
            surface_points_binary=model.structural_frame.surface_points_copy.data,  # TODO: Here we need to pass the binary array
            orientations_binary=model.structural_frame.orientations_copy.data
    ):
        model_deserialized = gp.data.GeoModel.model_validate_json(model_json)

    a = hash(model.structural_frame.structural_elements[1].surface_points.data.tobytes())
    b = hash(model_deserialized.structural_frame.structural_elements[1].surface_points.data.tobytes())

    o_a = hash(model.structural_frame.structural_elements[1].orientations.data.tobytes())
    o_b = hash(model_deserialized.structural_frame.structural_elements[1].orientations.data.tobytes())

    assert a == b, "Hashes for surface points are not equal"
    assert o_a == o_b, "Hashes for orientations are not equal"
    assert model_deserialized.__str__() == model.__str__()

    # # Validate json against schema
    if True:
        # Ensure the 'verify/' directory exists
        os.makedirs("verify", exist_ok=True)
        verify_model = json.loads(model_json)
        verify_model["meta"]["creation_date"] = "<DATE_IGNORED>"
        verify_json(json.dumps(verify_model, indent=4), name="verify/Horizontal Stratigraphic Model serialization")
        


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
