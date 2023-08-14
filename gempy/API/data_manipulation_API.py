"""These functions are meant to be used for bulk changes in the model. Otherwise just change the data directly"""
from typing import Sequence, Optional, Union

import numpy as np

from gempy.core.data import GeoModel, SurfacePointsTable, StructuralElement, StructuralFrame
from gempy.core.data.surface_points import DEFAULT_SP_NUGGET


def add_surface_points(geo_model: GeoModel, x: Sequence[float], y: Sequence[float], z: Sequence[float],
                       elements_names: Sequence[str], nugget: Optional[Sequence[float]] = None) -> StructuralFrame:
    # Ensure all provided Sequences have the same length
    lengths = {len(x), len(y), len(z), len(elements_names)}
    if len(lengths) != 1:
        raise ValueError("All input Sequences must have the same length.")

    # If nugget is not provided, create an Sequence filled with the default value
    if nugget is None:
        nugget = [DEFAULT_SP_NUGGET] * len(x)

    # Ensure nugget also has the same length as the other Sequences
    if len(nugget) != len(x):
        raise ValueError("The length of the nugget Sequence does not match the lengths of other input Sequences.")

    # * Split the sequences according to the elements_names
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    elements_names = np.array(elements_names)
    nugget = np.array(nugget)

    unique_names = np.unique(elements_names)
    grouped_data = {}

    for name in unique_names:
        mask = (elements_names == name)
        grouped_data[name] = {
            'x'     : x[mask],
            'y'     : y[mask],
            'z'     : z[mask],
            'nugget': nugget[mask]
        }

    # * Loop per element_name
    for element_name, data in grouped_data.items():
        formatted_data: np.ndarray = SurfacePointsTable.data_from_arrays(
            x=data['x'],
            y=data['y'],
            z=data['z'],
            names=[element_name] * len(data['x']),
            nugget=data['nugget'],
            name_id_map=geo_model.surface_points.name_id_map
        )

        element: StructuralElement = geo_model.structural_frame.get_element_by_name(element_name)
        element.surface_points.data = np.concatenate([
            element.surface_points.data,
            formatted_data
        ])
    
    return geo_model.structural_frame


def delete_surface_points():
    raise NotImplementedError


def add_orientations():
    raise NotImplementedError


def delete_orientations():
    raise NotImplementedError
