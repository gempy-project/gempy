import warnings
from typing import Union, Hashable

import numpy as np
from numpy import ndarray

from gempy.API.io_API import read_surface_points, read_orientations
from gempy_engine.core.data import InterpolationOptions
from ..core.data.grid_modules import RegularGrid
from ..optional_dependencies import require_subsurface
from ..core.data import StructuralElement
from ..core.data.geo_model import GeoModel
from ..core.data.grid import Grid
from ..core.data.importer_helper import ImporterHelper
from ..core.data.orientations import OrientationsTable
from ..core.data.structural_frame import StructuralFrame
from ..core.data.surface_points import SurfacePointsTable
from ..optional_dependencies import require_pooch


def create_geomodel(
        *,
        project_name: str = 'default_project',
        extent: Union[list, ndarray] = None,
        resolution: Union[list, ndarray] = None,
        refinement: int = 1,
        structural_frame: StructuralFrame = None,
        importer_helper: ImporterHelper = None,
) -> GeoModel:  # ? Do I need to pass pandas read kwargs?
    """
    Initializes and returns a GeoModel instance with specified parameters.

    Args:
        project_name (str, optional): The name of the project. Defaults to 'default_project'.
        extent (Union[List, np.ndarray], optional): The 3D extent of the grid. Must be provided if resolution is specified. Defaults to None.
        resolution (Union[List, np.ndarray], optional): The resolution of the grid. If None, an octree grid will be initialized. Defaults to None.
        refinement (int, optional): The level of refinement for the octree grid. Defaults to 1.
        structural_frame (StructuralFrame, optional): The structural frame of the GeoModel. Either this or importer_helper must be provided. Defaults to None.
        importer_helper (ImporterHelper, optional): Helper object for importing structural elements. Either this or structural_frame must be provided. Defaults to None.

    Returns:
        GeoModel: The initialized GeoModel object.

    Raises:
        ValueError: If neither structural_frame nor importer_helper is provided.
    """
    
    # init resolutions well
    if resolution is None:
        grid: Grid = Grid.init_octree_grid(
            extent=extent,
            octree_levels=refinement
        )
    else:
        grid: Grid = Grid.init_dense_grid(
            extent=extent,
            resolution=resolution
        )

    interpolation_options: InterpolationOptions = InterpolationOptions(
        range=1.7,
        c_o=10,
        mesh_extraction=True,
        number_octree_levels=refinement,
    )

    match (structural_frame, importer_helper):
        case (None, None):
            # ? For now my gut feeling is that is better to pass the structural frame explicitly
            raise ValueError("Either structural_frame or importer_helper must be provided. You can use StructuralFrame.initialize_default_structure() to create a default structural frame.")
            structural_frame = StructuralFrame.initialize_default_structure()
        case (None, _):
            structural_frame = _initialize_structural_frame(importer_helper)
        case _:
            pass

    geo_model: GeoModel = GeoModel(
        name=project_name,
        structural_frame=structural_frame,
        grid=grid,
        interpolation_options=interpolation_options
    )

    return geo_model


def structural_elements_from_borehole_set(
        borehole_set: "subsurface.core.geological_formats.BoreholeSet",
        elements_dict: dict) -> list[StructuralElement]:
    """Creates a list of StructuralElements from a BoreholeSet.

    Args:
        borehole_set (subsurface.core.geological_formats.BoreholeSet): The BoreholeSet object containing the boreholes.
        elements_dict (dict): A dictionary containing the properties of the structural elements to be created.

    Returns:
        list[StructuralElement]: A list of StructuralElement objects created from the borehole set.

    Raises:
        ValueError: If a top lithology ID specified in `elements_dict` is not found in the borehole set.

    """

    ss = require_subsurface()
    borehole_set: ss.core.geological_formats.BoreholeSet

    elements = []
    component_lith: dict[Hashable, np.ndarray] = borehole_set.get_bottom_coords_for_each_lith()

    for name, properties in elements_dict.items():
        top_coordinates = component_lith.get(properties['id'])
        if top_coordinates is None:
            raise ValueError(f"Top lithology {properties['id']} not found in borehole set.")

        element = StructuralElement(
            name=name,
            id=properties['id'],
            color=properties['color'],
            surface_points=SurfacePointsTable.from_arrays(
                x=top_coordinates[:, 0],
                y=top_coordinates[:, 1],
                z=top_coordinates[:, 2],
                names=[name],
                name_id_map={name: properties['id']}
            ),
            orientations=OrientationsTable(np.zeros(0, dtype=OrientationsTable.dt))
        )
        elements.append(element)
    # Reverse the list to have the oldest rocks at the bottom

    return elements


def create_data_legacy(
        *,
        project_name: str = 'default_project',
        extent: Union[list, ndarray] = None,
        resolution: Union[list, ndarray] = None,
        path_i: str = None,
        path_o: str = None) -> GeoModel:  # ? Do I need to pass pandas read kwargs?

    warnings.warn("This method is deprecated. Use create_geomodel instead.", DeprecationWarning)
    return create_geomodel(
        project_name=project_name,
        extent=extent,
        resolution=resolution,
        importer_helper=ImporterHelper(
            path_to_surface_points=path_i,
            path_to_orientations=path_o
        )
    )


def _initialize_structural_frame(importer_helper: ImporterHelper) -> StructuralFrame:
    surface_points, orientations = _read_input_points(importer_helper)

    return StructuralFrame.from_data_tables(surface_points, orientations)


def _read_input_points(importer_helper: ImporterHelper) -> (SurfacePointsTable, OrientationsTable):
    orientations_file, surface_points_file = _fetch_data_with_pooch(
        orientations_hash=importer_helper.hash_orientations,
        orientations_path=importer_helper.path_to_orientations,
        surface_points_hash=importer_helper.hash_surface_points,
        surface_points_path=importer_helper.path_to_surface_points
    )

    surface_points: SurfacePointsTable = read_surface_points(
        path=surface_points_file,
        coord_x_name=importer_helper.coord_x_name,
        coord_y_name=importer_helper.coord_y_name,
        coord_z_name=importer_helper.coord_z_name,
        surface_name=importer_helper.surface_name,
        pandas_kwargs=importer_helper.pandas_reader_kwargs
    )

    orientations: OrientationsTable = read_orientations(
        path=orientations_file,
        coord_x_name=importer_helper.coord_x_name,
        coord_y_name=importer_helper.coord_y_name,
        coord_z_name=importer_helper.coord_z_name,
        surface_name=importer_helper.surface_name,
        gx_name=importer_helper.gx_name,
        gy_name=importer_helper.gy_name,
        gz_name=importer_helper.gz_name,
        pandas_kwargs=importer_helper.pandas_reader_kwargs,
        name_id_map=surface_points.name_id_map
    )

    return surface_points, orientations


def _fetch_data_with_pooch(orientations_hash, orientations_path, surface_points_hash, surface_points_path):
    def is_url(url):
        from urllib.parse import urlparse
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    pooch = require_pooch() if is_url(surface_points_path) or is_url(orientations_path) else None
    # * Fetch or define path for surface points
    if is_url(surface_points_path):
        surface_points_file = pooch.retrieve(
            url=surface_points_path,
            known_hash=surface_points_hash
        )
        print("Surface points hash: ", pooch.file_hash(surface_points_file))
    else:
        surface_points_file = surface_points_path
    # * Fetch or define path for orientations
    if is_url(orientations_path):
        orientations_file = pooch.retrieve(
            url=orientations_path,
            known_hash=orientations_hash
        )
        print("Orientations hash: ", pooch.file_hash(orientations_file))
    else:
        orientations_file = orientations_path
    return orientations_file, surface_points_file
