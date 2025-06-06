import re

import warnings

from ...core.data import GeoModel
from ...core.data.encoders.converters import loading_model_from_binary
from ...optional_dependencies import require_zlib
import pathlib
import os
import io
import zipfile


def save_model(model: GeoModel, path: str | None = None, validate_serialization: bool = True):
    """
    Save a GeoModel to a file with proper extension validation.
    
    Parameters:
    -----------
    model : GeoModel
        The geological model to save
    path : str
        The file path where to save the model
    
    Raises:
    -------
    ValueError
        If the file has an extension other than .gempy
    """

    # Warning about preview
    warnings.warn("This function is still in development. It may not work as expected.")

    # Define the valid extension for gempy models
    VALID_EXTENSION = ".gempy"
    if path is None:
        path = model.meta.name + VALID_EXTENSION

    # Check if path has an extension
    path_obj = pathlib.Path(path)
    if path_obj.suffix:
        # If extension exists but is not valid, raise error
        if path_obj.suffix.lower() != VALID_EXTENSION:
            raise ValueError(f"Invalid file extension: {path_obj.suffix}. Expected: {VALID_EXTENSION}")
    else:
        # If no extension, add the valid extension
        path = str(path_obj) + VALID_EXTENSION

    binary_file = model_to_bytes(model)

    if validate_serialization:
        model_deserialized = _load_model_from_bytes(binary_file)
        _validate_serialization(model, model_deserialized)

    # Create directory if it doesn't exist
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    with open(path, 'wb') as f:
        f.write(binary_file)

    return path  # Return the actual path used (helpful if extension was added)


def model_to_binary(model: GeoModel) -> bytes:

    # Compress the binary data
    zlib = require_zlib()
    compressed_binary_input = zlib.compress(model.structural_frame.input_tables_binary)
    compressed_binary_grid = zlib.compress(model.grid.grid_binary)

    compressed_binary_grid = zlib.compress(model.grid.grid_binary, level=6)

    import hashlib
    print("len raw bytes:", len(model.grid.grid_binary))
    
    print("raw bytes hash:", hashlib.sha256(model.grid.grid_binary).hexdigest())
    print("compressed length:", len(compressed_binary_grid))
    print("zlib version:", zlib.ZLIB_VERSION)

    # * Add here the serialization meta parameters like: len_bytes
    model.structural_frame._input_binary_size = len(compressed_binary_input)
    model.grid._grid_binary_size = len(compressed_binary_grid)
    
    model_json = model.model_dump_json(by_alias=True, indent=4)
    binary_file = _to_binary(
        header_json=model_json,
        body_input=compressed_binary_input,
        body_grid=compressed_binary_grid
    )
    return binary_file


def load_model(path: str) -> GeoModel:
    """
    Load a GeoModel from a file with extension validation.
    
    Parameters:
    -----------
    path : str
        Path to the gempy model file
        
    Returns:
    --------
    GeoModel
        The loaded geological model
        
    Raises:
    -------
    ValueError
        If the file doesn't have the proper .gempy extension
    FileNotFoundError
        If the file doesn't exist
    """

    # Warning about preview
    warnings.warn("This function is still in development. It may not work as expected.")

    VALID_EXTENSION = ".gempy"

    # Check if path has the valid extension
    path_obj = pathlib.Path(path)
    if not path_obj.suffix or path_obj.suffix.lower() != VALID_EXTENSION:
        raise ValueError(f"Invalid file extension: {path_obj.suffix}. Expected: {VALID_EXTENSION}")

    # Check if file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, 'rb') as f:
        binary_file = f.read()

    return _load_model_from_bytes(binary_file)

def model_to_bytes(model: GeoModel) -> bytes:
    # 1) Make a fully deterministic JSON header
    # header_dict = model.model_dump(by_alias=True)
    # header_json = json.dumps(
    #     header_dict,
    #     sort_keys=True,          # always sort object keys
    #     separators=(",", ":"),   # no extra whitespace
    # ).encode("utf-8")
    
    header_json = model.model_dump_json(by_alias=True, indent=4)

    # 2) Raw binary chunks (no additional zlib.compress here)
    input_raw = model.structural_frame.input_tables_binary
    grid_raw  = model.grid.grid_binary

    # 3) Pack into a ZIP archive in a fixed order:
    
    buf = io.BytesIO()
    with zipfile.ZipFile(
            buf, mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6
    ) as zf:
        # Force a fixed timestamp (1980-01-01) so the file headers don't vary
        def make_info(name):
            zi = zipfile.ZipInfo(name, date_time=(1980,1,1,0,0,0))
            zi.external_attr = 0  # clear OS-specific file permissions
            return zi

        zf.writestr(make_info("header.json"), header_json)
        zf.writestr(make_info("input.bin"),   input_raw)
        zf.writestr(make_info("grid.bin"),    grid_raw)

    return buf.getvalue()

def _load_model_from_bytes(data: bytes) -> GeoModel:
    from ...core.data.encoders.converters import loading_model_from_binary

    buf = io.BytesIO(data)
    with zipfile.ZipFile(buf, "r") as zf:
        header_json = zf.read("header.json").decode("utf-8")
        # header      = json.loads(header_json)
        input_raw   = zf.read("input.bin")
        grid_raw    = zf.read("grid.bin")

    # If you want to validate or decompress further, do it here…
    with loading_model_from_binary(
            input_binary=input_raw,
            grid_binary= grid_raw
    ):
        model = GeoModel.model_validate_json(header_json)

    return model

def _deserialize_binary_file(binary_file):
    import json
    # Get header length from first 4 bytes
    header_length = int.from_bytes(binary_file[:4], byteorder='little')
    # Split header and body
    header_json= binary_file[4:4 + header_length].decode('utf-8')
    header = json.loads(header_json)
    input_metadata = header["structural_frame"]["binary_meta_data"]
    input_size = input_metadata["input_binary_size"]
    
    grid_metadata = header["grid"]["binary_meta_data"]
    grid_size = grid_metadata["grid_binary_size"]
    
    input_binary = binary_file[4 + header_length: 4 + header_length + input_size]
    all_sections_length = 4 + header_length + input_size + grid_size
    if all_sections_length != len(binary_file):
        raise  ValueError("Binary file is corrupted")
    
    grid_binary = binary_file[4 + header_length + input_size: all_sections_length]
    zlib = require_zlib()
    
    with loading_model_from_binary(
            input_binary=(zlib.decompress(input_binary)),
            grid_binary=(zlib.decompress(grid_binary))
    ):
        model = GeoModel.model_validate_json(header_json)
    return model


def _to_binary(header_json, body_input, body_grid) -> bytes:
    header_json_bytes = header_json.encode('utf-8')
    header_json_length = len(header_json_bytes)
    header_json_length_bytes = header_json_length.to_bytes(4, byteorder='little')
    file = header_json_length_bytes + header_json_bytes + body_input + body_grid
    return file


def _validate_serialization(original_model, model_deserialized):
    a = hash(original_model.structural_frame.surface_points_copy.data.tobytes())
    b = hash(model_deserialized.structural_frame.surface_points_copy.data.tobytes())
    o_a = hash(original_model.structural_frame.orientations_copy.data.tobytes())
    o_b = hash(model_deserialized.structural_frame.orientations_copy.data.tobytes())
    assert a == b, "Hashes for surface points are not equal"
    assert o_a == o_b, "Hashes for orientations are not equal"
    original_model___str__ = re.sub(r'\s+', ' ', original_model.__str__())
    deserialized___str__ = re.sub(r'\s+', ' ', model_deserialized.__str__())
    if original_model___str__ != deserialized___str__:
        # Find first char that is not the same
        for i in range(min(len(original_model___str__), len(deserialized___str__))):
            if original_model___str__[i] != deserialized___str__[i]:
                break
        print(f"First difference at index {i}:")
        i1 = 10
        print(f"Original: {original_model___str__[i - i1:i + i1]}")
        print(f"Deserialized: {deserialized___str__[i - i1:i + i1]}")

    assert deserialized___str__ == original_model___str__


