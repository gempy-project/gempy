import warnings
from typing import Union

from numpy import ndarray

from gempy_engine.core.data.stack_relation_type import StackRelationType
from gempy_engine.core.data import InterpolationOptions

from gempy.API.io_API import read_surface_points, read_orientations
from ..core.data.geo_model import GeoModel
from ..core.data.grid import Grid
from ..core.color_generator import ColorsGenerator
from ..core.data.importer_helper import ImporterHelper
from ..core.data.orientations import OrientationsTable
from ..core.data.structural_element import StructuralElement
from ..core.data.structural_frame import StructuralFrame
from ..core.data.structural_group import Stack
from ..core.data.surface_points import SurfacePointsTable
from ..optional_dependencies import require_pooch


def create_geomodel(
        *,
        project_name: str = 'default_project',
        extent: Union[list, ndarray] = None,
        resolution: Union[list, ndarray] = None,
        number_octree_levels: int = 1,
        importer_helper: ImporterHelper = None,
) -> GeoModel:  # ? Do I need to pass pandas read kwargs?

    grid: Grid = Grid(
        extent=extent,
        resolution=resolution
    )

    interpolation_options: InterpolationOptions = InterpolationOptions(
        range=1.73205,
        c_o=10,
        dual_contouring=True,
        number_octree_levels=number_octree_levels,
    )

    geo_model: GeoModel = GeoModel(
        name=project_name,
        structural_frame=_initialize_structural_frame(importer_helper),  # * Structural elements
        grid=grid,
        interpolation_options=interpolation_options
    )

    return geo_model


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

    surface_points_groups: list[SurfacePointsTable] = surface_points.get_surface_points_by_id_groups()
    orientations_groups: list[OrientationsTable] = orientations.get_orientations_by_id_groups()

    orientations_groups = OrientationsTable.fill_missing_orientations_groups(orientations_groups, surface_points_groups)

    colors_generator = ColorsGenerator()
    structural_elements = []
    
    for i in range(len(surface_points_groups)):
        structural_element: StructuralElement = StructuralElement(
            name=surface_points.id_to_name(i),
            surface_points=surface_points_groups[i],
            orientations=orientations_groups[i],
            color=next(colors_generator)
        )

        structural_elements.append(structural_element)

    # * Structural groups definitions
    default_formation: Stack = Stack(
        name="default_formation",
        elements=structural_elements,
        structural_relation=StackRelationType.ERODE
    )

    # ? Should I move this to the constructor?
    structural_frame: StructuralFrame = StructuralFrame(
        structural_groups=[default_formation],
        color_gen=colors_generator
    )
    return structural_frame


def _read_input_points(importer_helper: ImporterHelper) -> (SurfacePointsTable, OrientationsTable):
    orientations_file, surface_points_file = _fetch_data_with_pooch(
        orientations_hash=importer_helper.hash_orientations,
        orientations_path=importer_helper.path_to_orientations,
        surface_points_hash=importer_helper.hash_surface_points,
        surface_points_path=importer_helper.path_to_surface_points
    )

    surface_points: SurfacePointsTable = read_surface_points(
        path=surface_points_file,
    )

    orientations: OrientationsTable = read_orientations(
        path=orientations_file
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
        print(pooch.file_hash(surface_points_file))
    else:
        surface_points_file = surface_points_path
    # * Fetch or define path for orientations
    if is_url(orientations_path):
        orientations_file = pooch.retrieve(
            url=orientations_path,
            known_hash=orientations_hash
        )
        print(pooch.file_hash(orientations_file))
    else:
        orientations_file = orientations_path
    return orientations_file, surface_points_file
