from ...core.data import GeoModel
from ...core.data.encoders.converters import loading_model_from_binary
from ...optional_dependencies import require_zlib


def save_model(model: GeoModel, path: str):
    
    model_json = model.model_dump_json(by_alias=True, indent=4)

    # Compress the binary data
    zlib = require_zlib()
    compressed_binary = zlib.compress(model.structural_frame.input_tables_binary)

    binary_file = _to_binary(model_json, compressed_binary)
    
    # TODO: Add validation
    
    with open(path, 'wb') as f:
        f.write(binary_file)
        
    
        
def load_model(path: str) -> GeoModel:
    with open(path, 'rb') as f:
        binary_file = f.read()

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


