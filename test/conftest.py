
import pytest
import gempy
import sys, os

input_path = os.path.dirname(__file__)+'/input_data'
input_path2 = os.path.dirname(__file__)+'/../../notebooks'


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