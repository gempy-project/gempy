import pytest
import gempy.core.gempy_api as gempy
import gempy as gp

import sys, os

input_path = os.path.dirname(__file__)+'/input_data'
input_path2 = os.path.dirname(__file__)+'/../../notebooks/data'


@pytest.fixture(scope='session')
def interpolator():

    geo_model = gp.create_model('interpolator')

    # Importing the data from csv files and settign extent and resolution
    gp.init_data(geo_model, #[0, 10, 0, 10, -10, 0], [50, 50, 50],
                                 path_o=input_path + "/GeoModeller/test_a/test_a_Foliations.csv",
                                 path_i=input_path + "/GeoModeller/test_a/test_a_Points.csv")

    gempy.set_interpolation_data(geo_model,  grid=None, compile_theano=True)
    return geo_model.interpolator


@pytest.fixture(scope='session')
def interpolator_islith_isfault():

    geo_model = gp.create_model('interpolator_islith_isfault')

    # Importing the data from CSV-files and setting extent and resolution
    gp.init_data(geo_model,
                 path_o=input_path + "/simple_fault_model_orientations.csv",
                 path_i=input_path + "/simple_fault_model_points.csv", default_values=True)

    gp.map_series_to_surfaces(geo_model, {"Fault_Series": 'Main_Fault',
                                            "Strat_Series": ('Sandstone_2', 'Siltstone',
                                                             'Shale', 'Sandstone_1', 'basement')},
                                remove_unused_series=True)

    geo_model.set_is_fault(['Fault_Series'])

    gp.set_interpolation_data(geo_model, grid=None,
                              compile_theano=True,
                              theano_optimizer='fast_compile',
                              verbose=[])

    return geo_model.interpolator


@pytest.fixture(scope='session')
def interpolator_islith_nofault():

    geo_model = gp.create_model('interpolator_islith_isfault')

    # Importing the data from csv files and settign extent and resolution
    gp.init_data(geo_model, #[0, 10, 0, 10, -10, 0], [50, 50, 50],
                                 path_o=input_path + "/GeoModeller/test_a/test_a_Foliations.csv",
                                 path_i=input_path + "/GeoModeller/test_a/test_a_Points.csv")

    interpolator = gempy.set_interpolation_data(geo_model, grid=None, compile_theano=True)
    return interpolator


@pytest.fixture(scope='session')
def theano_f():
    # Importing the data from csv files and settign extent and resolution
    geo_data = gempy.create_data([0, 10, 0, 10, -10, 0], [50, 50, 50],
                                 path_o=input_path + "/GeoModeller/test_a/test_a_Foliations.csv",
                                 path_i=input_path + "/GeoModeller/test_a/test_a_Points.csv")

    interp_data = gempy.InterpolatorData(geo_data, dtype='float64', compile_theano=True,
                                         verbose=[])
    return interp_data


@pytest.fixture(scope='session')
def theano_f_1f():
    # Importing the data from csv files and settign extent and resolution
    geo_data = gempy.create_data([0, 10, 0, 10, -10, 0], [50, 50, 50],
                                 path_o=input_path+"/GeoModeller/test_d/test_d_Foliations.csv",
                                 path_i=input_path+"/GeoModeller/test_d/test_d_Points.csv")

    gempy.set_series(geo_data, {'series': ('A', 'B'),
                                'fault1': 'f1'}, order_series=['fault1', 'series'])

    interp_data = gempy.InterpolatorData(geo_data, dtype='float64', compile_theano=True, verbose=[])
    return interp_data


@pytest.fixture(scope='session')
def theano_f_grav():
    # Importing the data from csv files and settign extent and resolution
    geo_data = gempy.create_data([0, 10, 0, 10, -10, 0], [50, 50, 50],
                                 path_o=input_path + "/GeoModeller/test_a/test_a_Foliations.csv",
                                 path_i=input_path + "/GeoModeller/test_a/test_a_Points.csv")

    gempy.set_series(geo_data, {'series': ('A', 'B'),
                                'fault1': 'f1'}, order_series=['fault1', 'series'])

    interp_data = gempy.InterpolatorData(geo_data, dtype='float64', compile_theano=True, output='gravity', verbose=[])
    return interp_data