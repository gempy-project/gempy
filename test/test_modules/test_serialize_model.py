import tempfile

import json
import os
import pprint

from gempy_engine.core.data import InterpolationOptions

import gempy as gp
from gempy.core.data.encoders.converters import loading_model_from_binary
from gempy.core.data.enumerators import ExampleModel
from gempy.modules.serialization.save_load import save_model, load_model
from test.verify_helper import verify_json


def test_generate_horizontal_stratigraphic_model():
    model: gp.data.GeoModel = gp.generate_example_model(ExampleModel.HORIZONTAL_STRAT, compute_model=False)
    model_json = model.model_dump_json(by_alias=True, indent=4)

    # Write the JSON to disk
    if False:
        file_path = os.path.join("temp", "horizontal_stratigraphic_model.json")
        with open(file_path, "w") as f:
            f.write(model_json)

    with loading_model_from_binary(
            input_binary=model.structural_frame.input_tables_binary,
            grid_binary=model.grid.grid_binary
            # binary_body=model.structural_frame.input_tables_binary
    ):
        model_deserialized = gp.data.GeoModel.model_validate_json(model_json)

    _validate_serialization(model, model_deserialized)

    # # Validate json against schema
    if True:
        # Ensure the 'verify/' directory exists
        os.makedirs("verify", exist_ok=True)
        verify_model = json.loads(model_json)
        verify_model["meta"]["creation_date"] = "<DATE_IGNORED>"
        verify_json(json.dumps(verify_model, indent=4), name="verify/Horizontal Stratigraphic Model serialization")


def test_save_model_to_disk():
    model = gp.generate_example_model(ExampleModel.COMBINATION, compute_model=False)
    with tempfile.NamedTemporaryFile(mode='w+', delete=True) as tmp:
        tmp_name = tmp.name + ".gempy"  # Store the name to use it later
        save_model(model, tmp_name)

        # Load the model from disk
        loaded_model = load_model(tmp_name)
    _validate_serialization(model, loaded_model)

    gp.compute_model(loaded_model)
    if True:
        import gempy_viewer as gpv
        gpv.plot_3d(loaded_model, image=True)

    # Test save after compute
    with tempfile.NamedTemporaryFile(mode='w+', delete=True) as tmp:
        tmp_name = tmp.name + ".gempy"  # Store the name to use it later
        save_model(
            model=model,
            path=tmp_name,
            validate_serialization=True
        )


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


def _validate_serialization(original_model, model_deserialized):
    a = hash(original_model.structural_frame.surface_points_copy.data.tobytes())
    b = hash(model_deserialized.structural_frame.surface_points_copy.data.tobytes())
    o_a = hash(original_model.structural_frame.orientations_copy.data.tobytes())
    o_b = hash(model_deserialized.structural_frame.orientations_copy.data.tobytes())
    assert a == b, "Hashes for surface points are not equal"
    assert o_a == o_b, "Hashes for orientations are not equal"
    assert model_deserialized.__str__() == original_model.__str__()
