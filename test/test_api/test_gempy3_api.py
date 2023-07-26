from pprint import pprint

import gempy as gp
import gempy_viewer
from gempy import GeoModel
from gempy.optional_dependencies import require_gempy_legacy
from gempy.API.gp2_gp3_compatibility.gp3_to_gp2_input import gempy3_to_gempy2

from gempy_engine.core.data.solutions import Solutions


def test_api_create_data():
    geo_data = _create_data()

    pprint(geo_data)
    return geo_data


def _create_data():
    data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
    geo_data: GeoModel = gp.create_data_legacy(
        project_name='horizontal',
        extent=[0, 1000, 0, 1000, 0, 1000],
        resolution=[50, 5, 50],
        path_o=data_path + "/data/input_data/jan_models/model1_orientations.csv",
        path_i=data_path + "/data/input_data/jan_models/model1_surface_points.csv"
    )
    return geo_data


def test_map_stack_to_surfaces():
    geo_data = _create_data()
    
    gp.map_stack_to_surfaces(
        gempy_model=geo_data,
        mapping_object={"Strat_Series": ('rock2', 'rock1')}
    )
    
    pprint(geo_data.structural_frame)


def test_api_compute_model():
    geo_data = _create_data()
    
    gp.map_stack_to_surfaces(
        gempy_model=geo_data,
        mapping_object={"Strat_Series": ('rock2', 'rock1')}
    )
    
    gempy_viewer.plot_2d(geo_data, direction=['y'])
    
    sol: Solutions = gp.compute_model(geo_data)
    
    gempy_viewer.plot_2d(geo_data,  direction=['y'], show_data=True, show_boundaries=True, show_lith=False)
    gempy_viewer.plot_2d(geo_data,  direction=['y'], show_data=False, show_scalar=True)
    


def test_compare_numpy_with_legacy():
    geo_data = _create_data()

    gp.map_stack_to_surfaces(
        gempy_model=geo_data,
        mapping_object={"Strat_Series": ('rock2', 'rock1')}
    )

    sol: Solutions = gp.compute_model(geo_data)

    gempy_viewer.plot_2d(geo_data, direction=['y'], show_data=True, show_boundaries=False)

    legacy_model = gempy3_to_gempy2(geo_data)

    gl = require_gempy_legacy()
    gl.set_interpolator(legacy_model)
    gl.compute_model(legacy_model)
    gl.plot_2d(legacy_model, direction=['y'])
