import pooch
import gempy as gp
import gempy_engine.core.data.solutions
import gempy_viewer
from gempy.core.color_generator import ColorsGenerator
from gempy_engine.core.data import InterpolationOptions
from gempy.core.data import GeoModel
from gempy.API.io_API import read_orientations, read_surface_points
from gempy.core.data.orientations import OrientationsTable
from gempy.core.data.structural_element import StructuralElement
from gempy.core.data.structural_frame import StructuralFrame
from gempy.core.data.structural_group import Stack
from gempy.core.data.surface_points import SurfacePointsTable
from gempy.optional_dependencies import require_gempy_viewer

"""
- [x] create data
- [x] map series
- [ ] plot data
- [ ] compute
- [ ] plot results

"""

def test_read_input_points():
    _, _ = _read_data()


def _read_data() -> tuple[SurfacePointsTable, OrientationsTable]:
    data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
    surface_points_file = pooch.retrieve(
        url=data_path + "data/input_data/jan_models/model1_surface_points.csv",
        known_hash="6f1a39ed77e87a4057f03629c946b1876b87e24409cadfe0e1cf7ab1488f69e4"
    )
    orientations_file = pooch.retrieve(
        url=data_path + "data/input_data/jan_models/model1_orientations.csv",
        known_hash="04c307ae23f70252fe54144a2fb95ca7d96584a2d497ea539ed32dfd23e7cd5d"
    )
    print(pooch.file_hash(surface_points_file))
    print(pooch.file_hash(orientations_file))
    surface_points: SurfacePointsTable = read_surface_points(
        path=surface_points_file,
    )
    orientations: OrientationsTable = read_orientations(
        path=orientations_file
    )
    return surface_points, orientations


def test_create_grid():
    _generate_grid()


def _generate_grid():
    grid: gp.data.Grid = gp.data.Grid(
        extent=[0, 1000, 0, 1000, 0, 500],
        resolution=[50, 5, 50]
    )
    return grid


def test_create_structural_frame():
    # * Structural elements
    _create_structural_frame()


def _create_structural_frame():
    surface_points, orientations = _read_data()
    surface_points_groups = surface_points.get_surface_points_by_id_groups()
    orientations_groups = orientations.get_orientations_by_id_groups()
    structural_elements = []
    color_gen = ColorsGenerator()
    for i in range(len(surface_points_groups)):
        # TODO: Split surface points and orientations by id
        structural_element: StructuralElement = StructuralElement(
            name="layer1",
            surface_points=surface_points_groups[i],
            orientations=orientations_groups[i],
            color=next(color_gen)
        )

        structural_elements.append(structural_element)
    # * Structural groups definitions
    default_formation: Stack = Stack(
        name="default_formation",
        elements=structural_elements,
        structural_relation=gp.data.StackRelationType.ERODE
    )
    # ? Should I move this to the constructor?
    structural_frame: StructuralFrame = StructuralFrame(
        structural_groups=[default_formation],
        color_gen=color_gen
    )
    return structural_frame


def test_create_interpolation_options():
    interpolation_options = _generate_interpolation_options()


def _generate_interpolation_options():
    range_ = 1000.0
    interpolation_options: InterpolationOptions = InterpolationOptions.from_args(
        range=range_,
        c_o=(range_ ** 2) / 14 / 3,
    )
    return interpolation_options


def _create_geomodel() -> GeoModel:
    geo_model: GeoModel = GeoModel.from_args(
        name="horizontal",
        structural_frame=_create_structural_frame(),
        grid=_generate_grid(),
        interpolation_options=_generate_interpolation_options()
    )

    return geo_model


def test_structural_frame_surface_points():
    structural_frame: StructuralFrame = _create_structural_frame()
    print(structural_frame.surface_points_copy)
    pass


def _interpolate_numpy() -> GeoModel:
    geo_model: GeoModel = _create_geomodel()

    solutions: gempy_engine.core.data.solutions.Solutions = gempy_engine.compute_model(
        interpolation_input=geo_model.interpolation_input_copy,
        options=geo_model.interpolation_options,
        data_descriptor=geo_model.input_data_descriptor
    )
    print(solutions)
    geo_model.solutions = solutions
    # TODO: Use gempy API
    if PLOT := True:
        from gempy_engine.core.data.output.blocks_value_type import ValueType
        from gempy_engine.plugins.plotting.helper_functions import plot_block_and_input_2d
        plot_block_and_input_2d(
            stack_number=0,
            interpolation_input=geo_model.interpolation_input_copy,
            outputs=geo_model.solutions.octrees_output,
            structure=geo_model.structural_frame.input_data_descriptor.stack_structure,
            value_type=ValueType.ids
        )
        
        gempy_viewer.plot_3d(geo_model, show_boundaries=True, image=True)
        
    return geo_model


def test_interpolate_aesara():
    geo_model: GeoModel = _create_geomodel()


def test_plot_input():
    geo_model: GeoModel = _create_geomodel()
    gp_viewer: gempy_viewer = require_gempy_viewer()
    # TODO: Add all the plot data in a plot options class

    # TODO: Make options required
    gp_viewer.plot_2d(
        geo_model,
        direction=['y'],
        show_results=False
    )


def test_plot_results():
    solved_geo_model: GeoModel = _interpolate_numpy()
    gp_viewer: gempy_viewer = require_gempy_viewer()

    gp_viewer.plot_2d(
        solved_geo_model,
        direction=['y'],
        show_boundaries=False,  # TODO: Fix boundaries
        show_results=True
    )
