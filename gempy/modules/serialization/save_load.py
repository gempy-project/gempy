from ...core.data import GeoModel
from ...core.data.encoders.converters import loading_model_from_binary
from ...optional_dependencies import require_zlib
import pathlib
import os


def save_model(model: GeoModel, path: str, validate_serialization: bool = True):
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
    # Define the valid extension for gempy models
    VALID_EXTENSION = ".gempy"

    # Check if path has an extension
    path_obj = pathlib.Path(path)
    if path_obj.suffix:
        # If extension exists but is not valid, raise error
        if path_obj.suffix.lower() != VALID_EXTENSION:
            raise ValueError(f"Invalid file extension: {path_obj.suffix}. Expected: {VALID_EXTENSION}")
    else:
        # If no extension, add the valid extension
        path = str(path_obj) + VALID_EXTENSION

    model_json = model.model_dump_json(by_alias=True, indent=4)

    # Compress the binary data
    zlib = require_zlib()
    compressed_binary = zlib.compress(model.structural_frame.input_tables_binary)

    binary_file = _to_binary(model_json, compressed_binary)

    if validate_serialization:
        model_deserialized = _deserialize_binary_file(binary_file)
        _validate_serialization(model, model_deserialized)

    # Create directory if it doesn't exist
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    with open(path, 'wb') as f:
        f.write(binary_file)

    return path  # Return the actual path used (helpful if extension was added)


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

    return _deserialize_binary_file(binary_file)


def _deserialize_binary_file(binary_file):
    # Get header length from first 4 bytes
    header_length = int.from_bytes(binary_file[:4], byteorder='little')
    # Split header and body
    header_json = binary_file[4:4 + header_length].decode('utf-8')
    binary_body = binary_file[4 + header_length:]
    zlib = require_zlib()
    decompressed_binary = zlib.decompress(binary_body)
    with loading_model_from_binary(
            binary_body=decompressed_binary,
    ):
        model = GeoModel.model_validate_json(header_json)
    return model


def _to_binary(header_json, body_) -> bytes:
    header_json_bytes = header_json.encode('utf-8')
    header_json_length = len(header_json_bytes)
    header_json_length_bytes = header_json_length.to_bytes(4, byteorder='little')
    file = header_json_length_bytes + header_json_bytes + body_
    return file


def _validate_serialization(original_model, model_deserialized):
    a = hash(original_model.structural_frame.surface_points_copy.data.tobytes())
    b = hash(model_deserialized.structural_frame.surface_points_copy.data.tobytes())
    o_a = hash(original_model.structural_frame.orientations_copy.data.tobytes())
    o_b = hash(model_deserialized.structural_frame.orientations_copy.data.tobytes())
    assert a == b, "Hashes for surface points are not equal"
    assert o_a == o_b, "Hashes for orientations are not equal"
    assert model_deserialized.__str__() == original_model.__str__()
