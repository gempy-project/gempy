import numpy as np

from ..surface_points import SurfacePointsTable
from ..orientations import OrientationsTable
from ..micro_points import MicroPointsTable


def deserialize_input_data_tables(binary_array: bytes, name_id_map: dict, 
                                  sp_binary_length_: int, ori_binary_length_: int) -> tuple[OrientationsTable, SurfacePointsTable]:
    """
    Deserializes binary data into two tables: OrientationsTable and SurfacePointsTable.

    This function takes a binary array, a mapping of names to IDs, and lengths for
    specific parts of the binary data to extract and deserialize two distinct data
    tables: OrientationsTable and SurfacePointsTable. It uses the provided lengths
    to split the binary data accordingly and reconstructs the table contents from
    their respective binary representations.

    Args:
        binary_array (bytes): A bytes array containing the serialized data for
            both the OrientationsTable and SurfacePointsTable.
        name_id_map (dict): A dictionary mapping names to IDs which is used to
            help reconstruct the table objects.
        sp_binary_length_ (int): The length of the binary segment corresponding
            to the SurfacePointsTable data.
        ori_binary_length_ (int): The length of the binary segment corresponding
            to the OrientationsTable data.

    Returns:
        tuple[OrientationsTable, SurfacePointsTable]: A tuple containing two table
        objects: first the OrientationsTable, and second the SurfacePointsTable.
    """
    sp_binary = binary_array[:sp_binary_length_]
    ori_binary = binary_array[sp_binary_length_:sp_binary_length_+ori_binary_length_]
    # Reconstruct arrays
    sp_data: np.ndarray = np.frombuffer(sp_binary, dtype=SurfacePointsTable.dt)
    ori_data: np.ndarray = np.frombuffer(ori_binary, dtype=OrientationsTable.dt)
    surface_points_table = SurfacePointsTable(data=sp_data, name_id_map=name_id_map)
    orientations_table = OrientationsTable(data=ori_data, name_id_map=name_id_map)
    return orientations_table, surface_points_table


def deserialize_micro_points(
        binary_array: bytes,
        metadata: dict,
        name_id_map: dict[str, int],
) -> MicroPointsTable:
    """Deserialize and validate a versioned micro-point binary section."""
    if not isinstance(metadata, dict):
        raise ValueError("Micro-point binary metadata is missing")
    if metadata.get("dtype_version") != 1:
        raise ValueError(f"Unsupported micro-point dtype version: {metadata.get('dtype_version')}")

    row_count = metadata.get("row_count")
    byte_length = metadata.get("byte_length")
    if not isinstance(row_count, int) or row_count < 0:
        raise ValueError("Micro-point row_count must be a nonnegative integer")
    if not isinstance(byte_length, int) or byte_length < 0:
        raise ValueError("Micro-point byte_length must be a nonnegative integer")
    if byte_length != len(binary_array):
        raise ValueError("Micro-point binary length does not match metadata")
    if byte_length % MicroPointsTable.dt.itemsize != 0:
        raise ValueError("Micro-point binary length is not a whole number of rows")
    if row_count * MicroPointsTable.dt.itemsize != byte_length:
        raise ValueError("Micro-point row count does not match binary length")

    data = np.frombuffer(binary_array, dtype=MicroPointsTable.dt).copy()
    return MicroPointsTable(data=data, name_id_map=name_id_map)


def deserialize_grid(binary_array:bytes, custom_grid_length: int, topography_length: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Deserialize binary grid data into two numpy arrays.

    This function takes a binary array representing a grid and splits it into two separate
    numpy arrays: one for the custom grid and one for the topography. The binary array is
    segmented based on the provided lengths for the custom grid and topography.

    Args:
        binary_array: The binary data representing the combined custom grid and topography data.
        custom_grid_length: The length of the custom grid data segment in bytes.
        topography_length: The length of the topography data segment in bytes.

    Returns:
        A tuple where the first element is a numpy array representing the custom grid, and
        the second element is a numpy array representing the topography data.

    Raises:
        ValueError: If input lengths do not match the specified boundaries or binary data.
    """

    

    custom_grid_binary = binary_array[:custom_grid_length]
    topography_binary = binary_array[custom_grid_length:custom_grid_length + topography_length]
    custom_grid = np.frombuffer(custom_grid_binary, dtype=np.float64)
    topography = np.frombuffer(topography_binary)
    
    
    return custom_grid, topography
