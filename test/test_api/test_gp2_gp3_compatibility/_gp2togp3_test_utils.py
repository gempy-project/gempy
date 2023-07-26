import os

import gempy as gp
from gempy import Project


input_path = os.path.dirname(__file__) + '/../../../test/input_data'

def create_interpolator():
    m = gp.create_model('JustInterpolator')
    return gp.set_interpolator(m, theano_optimizer='fast_compile')


def load_model():
    geo_model = gp.create_model('Model_Tuto1-1')
    # Importing the data from CSV-files and setting extent and resolution
    gp.init_data(geo_model, [0, 2000., 0, 2000., 0, 2000.], [50, 50, 50],
                 path_o=input_path + "/simple_fault_model_orientations.csv",
                 path_i=input_path + "/simple_fault_model_points.csv", default_values=True)
    return geo_model


def map_sequential_pile(geo_model: Project) -> Project:
    gp.map_stack_to_surfaces(geo_model=geo_model,
                             mapping_object={
                                 "Fault_Series": 'Main_Fault',
                                 "Strat_Series": ('Sandstone_2', 'Siltstone',
                                                  'Shale', 'Sandstone_1', 'basement')
                             },
                             remove_unused_series=True)

    geo_model.set_is_fault(['Fault_Series'])
    return geo_model
