from typing import Sequence, Optional, Union

import numpy as np

from gempy.core.data import GeoModel, StructuralFrame, SurfacePointsTable, StructuralElement, OrientationsTable
from gempy.core.data.orientations import DEFAULT_ORI_NUGGET
from gempy.core.data.surface_points import DEFAULT_SP_NUGGET


def add_surface_points(geo_model: GeoModel, x: Sequence[float], y: Sequence[float], z: Sequence[float],
                       elements_names: Sequence[str], nugget: Optional[Sequence[float]] = None) -> StructuralFrame:
    elements_names = _validate_args(elements_names, x, y, z)

    # If nugget is not provided, create a Sequence filled with the default value
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
        formatted_data, _ = SurfacePointsTable.data_from_arrays(
            x=data['x'],
            y=data['y'],
            z=data['z'],
            names=[element_name] * len(data['x']),
            nugget=data['nugget'],
            name_id_map=None
        )

        element: StructuralElement = geo_model.structural_frame.get_element_by_name(element_name)
        element.surface_points.data = np.concatenate([
            element.surface_points.data,
            formatted_data
        ])

    return geo_model.structural_frame


def delete_surface_points():
    raise NotImplementedError


def add_orientations(geo_model: GeoModel, x: Sequence[float], y: Sequence[float], z: Sequence[float],
                     elements_names: Sequence[str], pole_vector: Optional[Sequence[np.ndarray]] = None,
                     orientation: Optional[Sequence[np.ndarray]] = None,
                     nugget: Optional[Sequence[float]] = None) -> StructuralFrame:
    if not pole_vector and not orientation:
        raise ValueError("Either pole_vector or orientation must be provided.")

    if orientation:  # Convert orientation to pole_vector (or gradient)
        pole_vector = convert_orientation_to_pole_vector(
            azimuth=orientation[:, 0],
            dip=orientation[:, 1],
            polarity=orientation[:, 2]
        )

    elements_names = _validate_args(elements_names, x, y, z, pole_vector)

    if nugget is None:  # If nugget is not provided, create a Sequence filled with the default value
        nugget = [DEFAULT_ORI_NUGGET] * len(x)

    # Ensure nugget also has the same length as the other Sequences
    if len(nugget) != len(x):
        raise ValueError("The length of the nugget Sequence does not match the lengths of other input Sequences.")

    # * Split the sequences according to the elements_names
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    elements_names = np.array(elements_names)
    pole_vector = np.array(pole_vector)
    nugget = np.array(nugget)

    unique_names = np.unique(elements_names)
    grouped_data = {}

    for name in unique_names:
        mask = (elements_names == name)
        grouped_data[name] = {
            'x'          : x[mask],
            'y'          : y[mask],
            'z'          : z[mask],
            'pole_vector': pole_vector[mask],
            'nugget'     : nugget[mask]
        }

    # * Loop per element_name
    for element_name, data in grouped_data.items():
        formatted_data, _ = OrientationsTable.data_from_arrays(
            x=data['x'],
            y=data['y'],
            z=data['z'],
            G_x=data['pole_vector'][..., 0],
            G_y=data['pole_vector'][..., 1],
            G_z=data['pole_vector'][..., 2],
            names=[element_name] * len(data['x']),
            nugget=data['nugget'],
        )

        element: StructuralElement = geo_model.structural_frame.get_element_by_name(element_name)
        element.orientations.data = np.concatenate([
            element.orientations.data,
            formatted_data
        ])

    return geo_model.structural_frame


def modify_all_orientations(geo_model: GeoModel, **orientation_field: Union[float, np.ndarray]) -> StructuralFrame:
    """
    Modifies all orientations in the structural frame. The keys of the orientation_field dictionary must match the
    names of the structural elements.
    
    Args:
        geo_model: GeoModel to modify

    Keyword Args:
            * X: Sequence[float] - X coordinates of the orientations
            * Y: Sequence[float] - Y coordinates of the orientations
            * Z: Sequence[float] - Z coordinates of the orientations
            * G_x: Sequence[float] - X component of the gradient vector
            * G_y: Sequence[float] - Y component of the gradient vector
            * G_z: Sequence[float] - Z component of the gradient vector
            * nugget: Sequence[float] - nugget value of the orientations
            
    Returns:
        The modified structural frame
    """
    orientations = geo_model.structural_frame.orientations
    for key, value in orientation_field.items():
        orientations.data[key] = value

    geo_model.orientations = orientations

    return geo_model.structural_frame


def modify_all_surface_points(geo_model: GeoModel, **surface_points_field: Union[float, np.ndarray]) -> StructuralFrame:
    """
    Modifies specified fields of all surface points in the structural frame. The keys of the surface_points_field 
    dictionary should match the field names in the surface points (e.g., "X", "Y", "Z", "nugget").
    
    Args:
        geo_model (GeoModel): The GeoModel instance to modify.
        
    Keyword Args:
        X (Union[float, np.ndarray]): X coordinates of the surface points.
        Y (Union[float, np.ndarray]): Y coordinates of the surface points.
        Z (Union[float, np.ndarray]): Z coordinates of the surface points.
        nugget (Union[float, np.ndarray]): Nugget value of the surface points.
            
    Returns:
        StructuralFrame: The modified structural frame.
    """
    surface_points = geo_model.structural_frame.surface_points
    for key, value in surface_points_field.items():
        surface_points.data[key] = value

    geo_model.surface_points = surface_points
    return geo_model.structural_frame


def delete_orientations():
    raise NotImplementedError


def convert_orientation_to_pole_vector(azimuth: Sequence[float], dip: Sequence[float], polarity: Sequence[float]) -> Sequence[np.ndarray]:
    # Convert sequences to numpy arrays for vectorized operations
    azimuth = np.array(azimuth)
    dip = np.array(dip)
    polarity = np.array(polarity)

    # Calculate gradient components
    G_x = np.sin(np.deg2rad(dip)) * np.sin(np.deg2rad(azimuth)) * polarity
    G_y = np.sin(np.deg2rad(dip)) * np.cos(np.deg2rad(azimuth)) * polarity
    G_z = np.cos(np.deg2rad(dip)) * polarity

    # Combine gradient components into an array of vectors
    gradients = np.vstack([G_x, G_y, G_z]).T

    return gradients


def _validate_args(elements_names, *args):
    if isinstance(elements_names, str):
        elements_names = np.array([elements_names] * len(args[0]))
    elif isinstance(elements_names, Sequence) or isinstance(elements_names, np.ndarray):
        pass
    else:
        raise TypeError(f"Names should be a string or a NumPy array, not {type(elements_names)}")
    # Ensure all provided Sequences have the same length
    lengths = {len(arg) for arg in args}
    if len(lengths) != 1:
        raise ValueError("All input Sequences must have the same length.")
    return elements_names
