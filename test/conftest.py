# import pytest
# import gempy.gempy_api as gempy
# import gempy as gp
import enum
import os

input_path = os.path.dirname(__file__) + '/input_data'
input_path2 = os.path.dirname(__file__) + '/../examples/data/input_data/'
import numpy as np

np.random.seed(1234)


class TestSpeed(enum.Enum):
    MILLISECONDS = 0
    SECONDS = 1
    MINUTES = 2
    HOURS = 3


TEST_SPEED = TestSpeed.SECONDS  # * Use seconds for compile errors, minutes before pushing and hours before release

# 
# @pytest.fixture(scope='session')
# def data_path():
#     return os.path.abspath(os.path.dirname(__file__) + '/input_data')
# 
# 
# @pytest.fixture(scope='session')
# def interpolator():
#     m = gp.create_model('JustInterpolator')
#     return gp.set_interpolator(m, aesara_optimizer='fast_run', verbose=["nsle", "mask_matrix_loop"])
# 
# 
# @pytest.fixture(scope='session')
# def interpolator_gravity():
#     m = gp.create_model('InterpolatorGravity')
#     return gp.set_interpolator(m, aesara_optimizer='fast_run', output=['gravity'],
#                                gradient=False)
# 
# 
# @pytest.fixture(scope='session')
# def interpolator_magnetics():
#     m = gp.create_model('InterpolatorMagnetics')
#     return gp.set_interpolator(m, aesara_optimizer='fast_run', output=['magnetics'],
#                                gradient=False)
# 
# 
# @pytest.fixture(scope='session')
# def one_fault_model_no_interp():
#     """This only makes sense for running small test fast"""
#     model = gp.create_data('one_fault_model', [0, 2000, 0, 2000, 0, 2000],
#                            [50, 50, 50],
#                            path_o=input_path2 + 'tut_chapter1/simple_fault_model_orientations.csv',
#                            path_i=input_path2 + 'tut_chapter1/simple_fault_model_points.csv')
# 
#     # Assigning series to surface as well as their order (timewise)
#     gp.map_stack_to_surfaces(model, {"Fault_Series": 'Main_Fault',
#                                      "Strat_Series": ('Sandstone_2', 'Siltstone',
#                                                       'Shale', 'Sandstone_1')},
#                              )
#     model.set_is_fault(['Fault_Series'])
#     return model
# 
# 
# @pytest.fixture(scope='session')
# def one_fault_model(interpolator):
#     """This only makes sense for running small test fast"""
#     model = gp.create_data('one_fault_model', [0, 2000, 0, 2000, 0, 2000],
#                            [50, 50, 50],
#                            path_o=input_path2 + 'tut_chapter1/simple_fault_model_orientations.csv',
#                            path_i=input_path2 + 'tut_chapter1/simple_fault_model_points.csv')
# 
#     # Assigning series to surface as well as their order (timewise)
#     gp.map_stack_to_surfaces(model, {"Fault_Series": 'Main_Fault',
#                                      "Strat_Series": ('Sandstone_2', 'Siltstone',
#                                                       'Shale', 'Sandstone_1')},
#                              )
#     model.set_is_fault(['Fault_Series'])
# 
#     model.set_aesara_function(interpolator)
# 
#     return model
# 
# 
# @pytest.fixture(scope='session')
# def one_fault_model_solution(one_fault_model):
#     gp.compute_model(one_fault_model)
#     return one_fault_model
# 
# 
# @pytest.fixture(scope='session')
# def one_fault_model_topo_solution(one_fault_model):
#     one_fault_model.update_additional_data()
#     one_fault_model.update_to_interpolator()
#     one_fault_model.set_topography(d_z=(800, 1800))
#     gp.compute_model(one_fault_model)
#     return one_fault_model
# 
# 
# @pytest.fixture(scope='session')
# def unconformity_model(interpolator):
#     geo_model = gp.create_data('unconformity_model',
#                                [0, 1000, 0, 1000, 0, 1000],
#                                resolution=[50, 42, 33],
#                                path_o=input_path2 + "jan_models/model6_orientations.csv",
#                                path_i=input_path2 + "jan_models/model6_surface_points.csv"
#                                )
#     gp.map_stack_to_surfaces(
#         geo_model,
#         {"Strat_Series1": ('rock3'),
#          "Strat_Series2": ('rock2', 'rock1'),
#          "Basement_Series": ('basement')}
#     )
# 
#     # with open("input_data/geomodel_jan_sol.p", "rb") as f:
#     # geo_model.solutions = load(f)
#     geo_model.set_aesara_function(interpolator)
#     gp.compute_model(geo_model)
#     return geo_model
# 
# 
# @pytest.fixture(scope='session')
# def unconformity_model_topo(interpolator):
#     geo_model = gp.create_data('unconformity_model',
#                                [0, 1000, 0, 1000, 0, 1000],
#                                resolution=[50, 42, 38],
#                                path_o=input_path2 + "jan_models/model6_orientations.csv",
#                                path_i=input_path2 + "jan_models/model6_surface_points.csv"
#                                )
# 
#     geo_model.set_topography('random', d_z=(200, 920))
#     gp.map_stack_to_surfaces(
#         geo_model,
#         {"Strat_Series1": ('rock3'),
#          "Strat_Series2": ('rock2', 'rock1'),
#          "Basement_Series": ('basement')}
#     )
# 
#     # with open("input_data/geomodel_jan_sol.p", "rb") as f:
#     # geo_model.solutions = load(f)
#     geo_model.set_aesara_function(interpolator)
#     gp.compute_model(geo_model)
#     return geo_model
# 
# 
# @pytest.fixture(scope='session')
# def model_horizontal_two_layers(interpolator):
#     geo_model = gp.create_model('interpolator')
# 
#     # Importing the data from csv files and settign extent and resolution
#     gp.init_data(geo_model, [0, 10, 0, 10, -10, 0], [50, 50, 50],
#                  path_o=input_path + "/GeoModeller/test_a/test_a_Foliations.csv",
#                  path_i=input_path + "/GeoModeller/test_a/test_a_Points.csv")
# 
#     geo_model.set_aesara_function(interpolator)
#     gp.compute_model(geo_model)
#     return geo_model
# 
# 
# @pytest.fixture(scope='session')
# def model_complex(interpolator):
#     model = gempy.create_data(extent=[0, 2500, 0, 1000, 0, 1000],
#                               resolution=[50, 20, 20],
#                               path_o=input_path2 + "jan_models/fixture_model_orientations.csv",
#                               path_i=input_path2 + "jan_models/fixture_model_surfaces.csv")
# 
#     # Assigning series to surface as well as their order (timewise)
#     gp.map_stack_to_surfaces(model,
#                              {"Fault_Series": ('fault'), "Strat_Series1": ('rock3'),
#                               "Strat_Series2": ('rock2', 'rock1'),
#                               "Basement_Series": ('basement')})
# 
#     model.set_is_fault(['Fault_Series'])
#     model.set_aesara_function(interpolator)
# 
#     return model
