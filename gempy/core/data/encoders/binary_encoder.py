import numpy as np

from ..surface_points import SurfacePointsTable
from ..orientations import OrientationsTable


def deserialize_input_data_tables(binary_array: bytes, name_id_map: dict, sp_binary_length_: int) -> tuple[OrientationsTable, SurfacePointsTable]:
    sp_binary = binary_array[:sp_binary_length_]
    ori_binary = binary_array[sp_binary_length_:]
    # Reconstruct arrays
    sp_data: np.ndarray = np.frombuffer(sp_binary, dtype=SurfacePointsTable.dt)
    ori_data: np.ndarray = np.frombuffer(ori_binary, dtype=OrientationsTable.dt)
    surface_points_table = SurfacePointsTable(data=sp_data, name_id_map=name_id_map)
    orientations_table = OrientationsTable(data=ori_data, name_id_map=name_id_map)
    return orientations_table, surface_points_table
