import json

import numpy as np

from ...core.data import GeoModel
from ...core.data.encoders.converters import loading_model_injection
from ...optional_dependencies import require_zlib


def save_model(model: GeoModel, path: str):
    import zlib
    
    # TODO: Serialize to json
    model_json = model.model_dump_json(by_alias=True, indent=4)

    # TODO: Serialize to binary
    data: np.ndarray = model.structural_frame.surface_points_copy.data
    sp_binary = data.tobytes()
    ori_binary = model.structural_frame.orientations_copy.data.tobytes()

    # Compress the binary data
    compressed_binary = zlib.compress(sp_binary + ori_binary)

    # Add compression info to metadata
    model_dict = model.model_dump(by_alias=True)
    model_dict["_binary_metadata"] = {
            "sp_shape"   : model.structural_frame.surface_points_copy.data.shape,
            "sp_dtype"   : str(model.structural_frame.surface_points_copy.data.dtype),
            "ori_shape"  : model.structural_frame.orientations_copy.data.shape,
            "ori_dtype"  : str(model.structural_frame.orientations_copy.data.dtype),
            "compression": "zlib",
            "sp_length"  : len(sp_binary)  # Need this to split the arrays after decompression
    }
    
    # TODO: Putting both together
    binary_file = _to_binary(model_json, compressed_binary)
    with open(path, 'wb') as f:
        f.write(binary_file)
        
def load_model(path: str) -> GeoModel:
    with open(path, 'rb') as f:
        binary_file = f.read()

    # Get header length from first 4 bytes
    header_length = int.from_bytes(binary_file[:4], byteorder='little')

    # Split header and body
    header_json = binary_file[4:4 + header_length].decode('utf-8')
    header_dict = json.loads(header_json)

    metadata = header_dict.pop("_binary_metadata")

    # Decompress the binary data
    ori_data, sp_data = _foo(binary_file, header_length, metadata)

    with loading_model_injection(
            surface_points_binary=sp_data,
            orientations_binary=ori_data
    ):
        model = GeoModel.model_validate_json(header_json)

    return model


def _foo(binary_file, header_length, metadata):
    zlib = require_zlib()
    body = binary_file[4 + header_length:]
    decompressed_binary = zlib.decompress(body)
    # Split the decompressed data using the stored length
    sp_binary = decompressed_binary[:metadata["sp_length"]]
    ori_binary = decompressed_binary[metadata["sp_length"]:]
    # Reconstruct arrays
    sp_data = np.frombuffer(sp_binary, dtype=np.dtype(metadata["sp_dtype"])).reshape(metadata["sp_shape"])
    ori_data = np.frombuffer(ori_binary, dtype=np.dtype(metadata["ori_dtype"])).reshape(metadata["ori_shape"])
    return ori_data, sp_data


def _to_binary(header_json, body_) -> bytes:
    header_json_bytes = header_json.encode('utf-8')
    header_json_length = len(header_json_bytes)
    header_json_length_bytes = header_json_length.to_bytes(4, byteorder='little')
    file = header_json_length_bytes + header_json_bytes + body_
    return file


