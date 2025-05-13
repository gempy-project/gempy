import numpy as np
import os
import pandas as pd
import pprint
import tempfile

import gempy as gp
from gempy.core.data import SurfacePointsTable
from gempy.core.data.encoders.converters import loading_model_injection
from gempy.core.data.enumerators import ExampleModel
from gempy_engine.core.data import InterpolationOptions
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

    if False:  # * Use this to debug which fields are giving problems
        # model_deserialized = gp.data.GeoModel.model_validate(from_json(model_json, allow_partial=True))
        pass
    else:
        with loading_model_injection(
            surface_points_binary = model.structural_frame.surface_points_copy.data,# TODO: Here we need to pass the binary array
            orientations_binary = model.structural_frame.orientations_copy.data
        ):
            model_deserialized = gp.data.GeoModel.model_validate_json(model_json)

    a = hash(model.structural_frame.structural_elements[1].surface_points.data.tobytes())
    b = hash(model_deserialized.structural_frame.structural_elements[1].surface_points.data.tobytes())
    
    o_a = hash(model.structural_frame.structural_elements[1].orientations.data.tobytes())
    o_b = hash(model_deserialized.structural_frame.structural_elements[1].orientations.data.tobytes())
    
    assert a == b, "Hashes for surface points are not equal"
    assert o_a == o_b, "Hashes for orientations are not equal"
    
    # TODO: [ ] Structural frame?
    # TODO: [ ] Input transform?
    assert model_deserialized.__str__() == model.__str__()

    # # Validate json against schema
    if False:
        verify_json(model_json, name="verify/Horizontal Stratigraphic Model serialization")




def test_generate_horizontal_stratigraphic_model_binary():
    model: gp.data.GeoModel = gp.generate_example_model(ExampleModel.HORIZONTAL_STRAT, compute_model=False)
    sp = model.surface_points_copy
    element_id_map: dict[int, str] = model.structural_frame.element_id_name_map

    # * So basically we have 12 elements of a very complex type
    data = sp.data
    assert data.shape[0] == 12
    assert data.dtype == np.dtype([('X', 'f8'), ('Y', 'f8'), ('Z', 'f8'), ('id', 'i4'), ('nugget', 'f8')])  #: The custom data type for the data array.
    
    df = pd.DataFrame(data)
    ds = df.to_xarray()
    pass


def test_split_input_data_tables():
    model: gp.data.GeoModel = gp.generate_example_model(ExampleModel.HORIZONTAL_STRAT, compute_model=False)
    sp: gp.data.SurfacePointsTable = model.surface_points_copy
    # Temp save sp numpy array 

    with tempfile.NamedTemporaryFile(delete=False, suffix='.npy') as temp:
        np.save(
            file=temp,
            arr=sp.data
        )
        temp_path = temp.name
    
    # Load 
    loaded_array = np.load(temp_path)
    loaded_table = SurfacePointsTable(
        data=loaded_array,
        name_id_map=sp.name_id_map
    )
    
    model.structural_frame.surface_points = loaded_table


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
