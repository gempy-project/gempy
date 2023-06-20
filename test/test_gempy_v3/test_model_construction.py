import pooch
import gempy as gp
from gempy.API.io_API import read_orientations, read_surface_points
from gempy.core.data.orientations import Orientations
from gempy.core.data.structural_element import StructuralElement
from gempy.core.data.structural_frame import StructuralFrame
from gempy.core.data.surface_points import SurfacePoints


def test_read_input_points():
    data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
    surface_points_file = pooch.retrieve(
        url=data_path + "data/input_data/jan_models/model1_surface_points.csv",
        known_hash=None
    )
    
    orientations_file = pooch.retrieve(
        url=data_path + "data/input_data/jan_models/model1_orientations.csv",
        known_hash=None
    )

    print(pooch.file_hash(surface_points_file))
    print(pooch.file_hash(orientations_file))

    surface_points: SurfacePoints = read_surface_points(
        path=surface_points_file,
    )

    orientations: Orientations = read_orientations(
        path=orientations_file
    )

    return surface_points, orientations


def test_create_grid():
    grid: gp.Grid = gp.Grid(
        extent=[0, 1000, 0, 1000, 0, 1000],
        resolution=[50, 50, 50]
    )


def test_create_structural_frame():
    surface_points, orientations = test_read_input_points()
    
    # TODO: Split surface points and orientations by id
    structural_element: StructuralElement = StructuralElement(
        name="layer1"
    )

    structural_frame: StructuralFrame = StructuralFrame()


def test_create_geo_model():
    geo_data: gp.GeoModel = gp.GeoModel(name='horizontal')
