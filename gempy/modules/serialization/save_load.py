from ...core.data import GeoModel
from ...core.data.encoders.converters import loading_model_injection


def save_model(model: GeoModel, path: str):
    
    # TODO: Serialize to json
    model_json = model.model_dump_json(by_alias=True, indent=4)

    # TODO: Serialize to binary
    sp_binary = model.structural_frame.surface_points_copy.data.tobytes()
    ori_binary = model.structural_frame.orientations_copy.data.tobytes()
    
    # TODO: Putting both together
    binary_file = _to_binary(model_json, sp_binary + ori_binary)
    with open(path, 'wb') as f:
        f.write(binary_file)
        
def load_model(path: str) -> GeoModel:
    with open(path, 'rb') as f:
        binary_file = f.read()

    # Get header length from first 4 bytes
    header_length = int.from_bytes(binary_file[:4], byteorder='little')

    # Split header and body
    header_json = binary_file[4:4 + header_length].decode('utf-8')
    body = binary_file[4 + header_length:]

    # Split body into surface points and orientations
    # They are equal size so we can split in half
    sp_binary = body[:len(body) // 2]
    ori_binary = body[len(body) // 2:]

    with loading_model_injection(
            surface_points_binary=sp_binary,
            orientations_binary=ori_binary
    ):
        model = GeoModel.model_validate_json(header_json)

    return model


def _to_binary(header_json, body_) -> bytes:
    header_json_bytes = header_json.encode('utf-8')
    header_json_length = len(header_json_bytes)
    header_json_length_bytes = header_json_length.to_bytes(4, byteorder='little')
    file = header_json_length_bytes + header_json_bytes + body_
    return file


