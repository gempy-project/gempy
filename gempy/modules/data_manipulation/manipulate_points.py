from typing import Sequence, Optional, Union

import numpy as np

from ...core.data import GeoModel, StructuralFrame, SurfacePointsTable, StructuralElement, OrientationsTable
from ...core.data.orientations import DEFAULT_ORI_NUGGET
from ...core.data.surface_points import DEFAULT_SP_NUGGET


def add_surface_points(
        geo_model: GeoModel,
        x: Sequence[float],
        y: Sequence[float],
        z: Sequence[float],
        elements_names: Sequence[str],
        nugget: Optional[Sequence[float]] = None
) -> StructuralFrame:
    """Add surface points to the geological model.

    This function adds surface points to the specified geological elements in the model.
    The points are grouped by element names, and optional nugget values can be specified
    for each point.

    Args:
        geo_model (GeoModel): The geological model to which the surface points will be added.
        x (Sequence[float]): Sequence of x-coordinates for the surface points.
        y (Sequence[float]): Sequence of y-coordinates for the surface points.
        z (Sequence[float]): Sequence of z-coordinates for the surface points.
        elements_names (Sequence[str]): Sequence of element names corresponding to each surface point.
        nugget (Optional[Sequence[float]]): Sequence of nugget values for each surface point. If not provided,
            a default value will be used for all points.

    Returns:
        StructuralFrame: The updated structural frame of the geological model.

    Raises:
        ValueError: If the length of the nugget sequence does not match the lengths of the other input sequences.
    """
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
        formatted_data, _ = SurfacePointsTable._data_from_arrays(
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


def add_orientations(geo_model: GeoModel,
        x: Sequence[float],
        y: Sequence[float],
        z: Sequence[float],
        elements_names: Sequence[str],
        pole_vector: Optional[Union[Sequence[np.ndarray], np.ndarray]] = None,
        orientation: Optional[Union[Sequence[np.ndarray], np.ndarray]] = None,
        nugget: Optional[Sequence[float]] = None,
        name_id_map: Optional[dict[str, int]] = None  #: A mapping between orientation names and ids.
) -> StructuralFrame:
    """Add orientation data to the geological model.

    This function adds orientation data to the specified geological elements in the model.
    The orientation can be provided directly as pole vectors or as orientation angles (azimuth, dip, polarity).
    Optional nugget values can also be specified for each orientation point.

    Args:
        geo_model (GeoModel): The geological model to which the orientations will be added.
        x (Sequence[float]): Sequence of x-coordinates for the orientation points.
        y (Sequence[float]): Sequence of y-coordinates for the orientation points.
        z (Sequence[float]): Sequence of z-coordinates for the orientation points.
        elements_names (Sequence[str]): Sequence of element names corresponding to each orientation point.
        pole_vector (Optional[Union[Sequence[np.ndarray], np.ndarray]]): Sequence of pole vectors for each orientation point. If is np.ndarray, it should have shape (n, 3).
        orientation (Optional[Union[Sequence[np.ndarray], np.ndarray]]): Sequence of orientation angles for each orientation point. If is np.ndarray, it should have shape (n, 3).
        nugget (Optional[Sequence[float]]): Sequence of nugget values for each orientation point. If not provided,
            a default value will be used for all points.

    Returns:
        StructuralFrame: The updated structural frame of the geological model.

    Raises:
        ValueError: If neither pole_vector nor orientation is provided, or if the length of the nugget sequence
            does not match the lengths of the other input sequences.
    """
    if pole_vector is None and orientation is None:
        raise ValueError("Either pole_vector or orientation must be provided.")

    if orientation is not None:
        orientation = np.array(orientation, ndmin=2)
        pole_vector = convert_orientation_to_pole_vector(
            azimuth=orientation[:, 0],
            dip=orientation[:, 1],
            polarity=orientation[:, 2]
        )
    else:
        pole_vector = np.array(pole_vector, ndmin=2)

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
        formatted_data, _ = OrientationsTable._data_from_arrays(
            x=data['x'],
            y=data['y'],
            z=data['z'],
            G_x=data['pole_vector'][..., 0],
            G_y=data['pole_vector'][..., 1],
            G_z=data['pole_vector'][..., 2],
            names=[element_name] * len(data['x']),
            nugget=data['nugget'],
            name_id_map=name_id_map
        )

        element: StructuralElement = geo_model.structural_frame.get_element_by_name(element_name)
        element.orientations.data = np.concatenate([
                element.orientations.data,
                formatted_data
        ])

    return geo_model.structural_frame


def modify_orientations(geo_model: GeoModel, slice: Optional[Union[int, slice]] = None,
                        **orientation_field: Union[float, np.ndarray]) -> StructuralFrame:
    """
    Modifies specified fields of all orientations in the structural frame. The keys of the orientation_field 
    dictionary should match the field names in the orientations (e.g., "X", "Y", "Z", "G_x", "G_y", "G_z", "nugget").
    
    Args:
        geo_model (GeoModel): The GeoModel instance to modify.
        slice (Optional[Union[int, slice]]): The slice of orientations to modify. If None, all orientations will be modified.
        
    Keyword Args:
        X (Union[float, np.ndarray]): X coordinates of the orientations.
        Y (Union[float, np.ndarray]): Y coordinates of the orientations.
        Z (Union[float, np.ndarray]): Z coordinates of the orientations.
        azimuth (Union[float, np.ndarray]): Azimuth angles of the orientations.
        dip (Union[float, np.ndarray]): Dip angles of the orientations.
        polarity (Union[float, np.ndarray]): Polarity values of the orientations.
        G_x (Union[float, np.ndarray]): X component of the gradient vector.
        G_y (Union[float, np.ndarray]): Y component of the gradient vector.
        G_z (Union[float, np.ndarray]): Z component of the gradient vector.
        nugget (Union[float, np.ndarray]): Nugget value of the orientations.
            
    Returns:
        StructuralFrame: The modified structural frame.
    """

    orientations = geo_model.structural_frame.orientations_copy

    # If no slice is provided, target all rows; else, target specified slice
    target_rows = slice if slice is not None else np.s_[:]

    # Extract provided orientation fields without modifying the dictionary
    azimuth = orientation_field.pop('azimuth', None)
    dip = orientation_field.pop('dip', None)
    polarity = orientation_field.pop('polarity', None)

    # Update all the other fields
    for key, value in orientation_field.items():
        if isinstance(value, np.ndarray) and len(value) != len(orientations.data[target_rows]):
            raise ValueError(f"Length mismatch: Expected size {len(orientations.data[target_rows])} for field {key}, but got {len(value)}.")
        orientations.data[key][target_rows] = value

    # Check if azimuth, dip, or polarity are provided
    any_polar_fields = azimuth is not None or dip is not None or polarity is not None
    all_polar_fields = azimuth is not None and dip is not None and polarity is not None

    match (any_polar_fields, all_polar_fields):
        case (True, True):
            # All polar fields provided, convert to gradients
            gx, gy, gz = convert_orientation_to_pole_vector(np.asarray(azimuth), np.asarray(dip), np.asarray(polarity))
            orientations.data['G_x'][target_rows] = gx
            orientations.data['G_y'][target_rows] = gy
            orientations.data['G_z'][target_rows] = gz

        case (True, False):
            # Some polar fields missing, compute missing fields from gradients
            prev_azimuth, prev_dip, prev_polarity = compute_adp_from_gradients(
                orientations.data['G_x'],
                orientations.data['G_y'],
                orientations.data['G_z']
            )
            azimuth = np.asarray(azimuth) if azimuth is not None else prev_azimuth
            dip = np.asarray(dip) if dip is not None else prev_dip
            polarity = np.asarray(polarity) if polarity is not None else prev_polarity

            gradients = convert_orientation_to_pole_vector(azimuth, dip, polarity)
            orientations.data['G_x'][target_rows] = gradients[:, 0]
            orientations.data['G_y'][target_rows] = gradients[:, 1]
            orientations.data['G_z'][target_rows] = gradients[:, 2]

        case (_, _):
            pass

    geo_model.orientations = orientations
    return geo_model.structural_frame


def modify_surface_points(geo_model: GeoModel,
                          slice: Optional[Union[int, slice]] = None,
                          elements_names: Optional[Sequence[str]] = None,
                          **surface_points_field: Union[float, np.ndarray]) -> StructuralFrame:
    """
    Modifies specified fields of all surface points in the structural frame. The keys of the surface_points_field 
    dictionary should match the field names in the surface points (e.g., "X", "Y", "Z", "nugget").
    
    Args:
        geo_model (GeoModel): The GeoModel instance to modify.
        slice (Optional[Union[int, slice]]): The slice of surface points to modify. If None, all surface points will be modified.
        
    Keyword Args:
        X (Union[float, np.ndarray]): X coordinates of the surface points.
        Y (Union[float, np.ndarray]): Y coordinates of the surface points.
        Z (Union[float, np.ndarray]): Z coordinates of the surface points.
        nugget (Union[float, np.ndarray]): Nugget value of the surface points.
            
    Returns:
        StructuralFrame: The modified structural frame.
    """
    if elements_names is not None and slice is not None:
        raise ValueError("Cannot provide both elements_names and slice.")

    surface_points = geo_model.structural_frame.surface_points_copy

    if elements_names is not None:
        ids = [surface_points.name_id_map[element] for element in elements_names]
        slice = np.s_[np.isin(surface_points.data['id'], ids)]

    # If no slice is provided, target all rows; else, target specified slice
    target_rows = slice if slice is not None else np.s_[:]

    for key, value in surface_points_field.items():
        if isinstance(value, np.ndarray) and len(value) != len(surface_points.data[target_rows]):
            raise ValueError(f"Length mismatch: Expected size {len(surface_points.data[target_rows])} for field {key}, but got {len(value)}.")

        surface_points.data[key][target_rows] = value

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


def compute_adp_from_gradients(G_x: np.ndarray, G_y: np.ndarray, G_z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Calculate polarity (here assumed to be 1 for all, but you can adapt if needed)
    polarity = np.ones_like(G_x)

    # Calculate dip
    dip = np.rad2deg(np.nan_to_num(np.arccos(G_z / polarity)))

    # Calculate azimuth
    azimuth = np.rad2deg(np.nan_to_num(np.arctan2(G_x / polarity, G_y / polarity)))

    # Shift values from [-pi, 0] to [pi,2*pi]
    azimuth[azimuth < 0] += 360

    # Adjust azimuth where dip is nearly zero, because if dip is zero, azimuth is undefined
    azimuth[dip < 0.001] = 0

    return azimuth, dip, polarity


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
