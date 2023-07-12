from typing import Union

import numpy as np
from numpy import ndarray

from gempy_engine.core.data import InterpolationOptions

from .io_API import read_surface_points, read_orientations
from .. import GeoModel, Grid
from ..core.data.orientations import OrientationsTable
from ..core.data.structural_element import StructuralElement
from ..core.data.structural_frame import StructuralFrame
from ..core.data.structural_group import Stack
from ..core.data.surface_points import SurfacePointsTable
from ..optional_dependencies import require_pooch


def create_data(
        *,
        project_name: str = 'default_project',
        extent: Union[list, ndarray] = None,
        resolution: Union[list, ndarray] = None,
        path_i: str = None,
        path_o: str = None) -> GeoModel:  # ? Do I need to pass pandas read kwargs?

    grid: Grid = Grid(
        extent=extent,
        resolution=resolution
    )

    default_range = np.sqrt(
        (extent[0] - extent[1]) ** 2 +
        (extent[2] - extent[3]) ** 2 +
        (extent[4] - extent[5]) ** 2)
    
    interpolation_options: InterpolationOptions = InterpolationOptions(
        range=default_range,
        c_o=(default_range ** 2) / 14 / 3,
    )

    geo_model: GeoModel = GeoModel(
        name=project_name,
        structural_frame=_initialize_structural_frame(path_i, path_o),  # * Structural elements
        grid=grid,
        interpolation_options=interpolation_options
    )
    
    return geo_model


def _initialize_structural_frame(surface_points_path: str, orientations_path: str) -> StructuralFrame:
    surface_points, orientations = _read_input_points(
        surface_points_path=surface_points_path,
        orientations_path=orientations_path
    )

    surface_points_groups = surface_points.get_surface_points_by_id_groups()
    orientations_groups = orientations.get_orientations_by_id_groups()
    structural_elements = []
    for i in range(len(surface_points_groups)):
        # TODO: Split surface points and orientations by id
        structural_element: StructuralElement = StructuralElement(
            name="layer1",
            surface_points=surface_points_groups[i],
            orientations=orientations_groups[i],
            color=next(StructuralFrame.color_gen),
        )

        structural_elements.append(structural_element)

    # * Structural groups definitions
    default_formation: Stack = Stack(
        name="default_formation",
        elements=structural_elements
    )
    # ? Should I move this to the constructor?
    structural_frame: StructuralFrame = StructuralFrame(
        structural_groups=[default_formation],
        structural_elements=structural_elements
    )
    return structural_frame


def _read_input_points(surface_points_path: str, orientations_path: str) -> (SurfacePointsTable, OrientationsTable):
    from urllib.parse import urlparse

    def is_url(url):
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
            known_hash=None,
        )
        print(pooch.file_hash(surface_points_file))
    else:
        surface_points_file = surface_points_path

    # * Fetch or define path for orientations
    if is_url(orientations_path):
        orientations_file = pooch.retrieve(
            url=orientations_path,
            known_hash=None,
        )
        print(pooch.file_hash(orientations_file))
    else:
        orientations_file = orientations_path

    surface_points: SurfacePointsTable = read_surface_points(
        path=surface_points_file,
    )

    orientations: OrientationsTable = read_orientations(
        path=orientations_file
    )

    return surface_points, orientations


