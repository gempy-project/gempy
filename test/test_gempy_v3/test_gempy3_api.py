from pprint import pprint

import gempy as gp
from gempy import GeoModel


def test_api_create_data():
    data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
    geo_data: GeoModel = gp.create_data(
        project_name='horizontal',
        extent=[0, 1000, 0, 1000, 0, 1000],
        resolution=[50, 50, 50],
        path_o=data_path + "/data/input_data/jan_models/model1_orientations.csv",
        path_i=data_path + "/data/input_data/jan_models/model1_surface_points.csv"
    )
    
    pprint(geo_data)
    return geo_data


def test_map_stack_to_surfaces():
    geo_data = test_api_create_data()
    
    gp.map_stack_to_surfaces(
        gempy_model=geo_data,
        mapping_object={"Strat_Series": ('rock2', 'rock1')}
    )
    
    pprint(geo_data)

