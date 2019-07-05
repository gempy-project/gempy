import pytest
import os
import gempy as gp
import numpy as np
from gempy.utils import stochastic_surface as ss

input_path = os.path.dirname(__file__)+'/../input_data'


@pytest.fixture(scope="module")
def load_model():
    geo_model = gp.create_model('Model_Tuto1-1')

    # Importing the data from CSV-files and setting extent and resolution
    gp.init_data(geo_model, [0, 2000., 0, 2000., 0, 2000.], [50 ,50 ,50],
          path_o = input_path+"/simple_fault_model_orientations.csv",
          path_i = input_path+"/simple_fault_model_points.csv",
                 default_values=True)

    gp.get_data(geo_model, 'surface_points').head()
    return geo_model


@pytest.fixture(scope="module")
def stochsurf_scipy(load_model):
    return ss.StochasticSurfaceScipy(load_model, "Shale")


@pytest.mark.usefixtures("stochsurf_scipy")
class TestScipy(object):

    def test_single_len(self, stochsurf_scipy):
        stochsurf_scipy.parametrize_surfpts_single(1)
        assert all(stochsurf_scipy.parametrization[0][0] == np.arange(16))

    def test_single_axis(self, stochsurf_scipy):
        stochsurf_scipy.parametrize_surfpts_single(1, direction="X")
        assert stochsurf_scipy.parametrization[0][1] == "X"

    def test_single_dist(self, stochsurf_scipy):
        stochsurf_scipy.parametrize_surfpts_single(1)
        assert str(type(stochsurf_scipy.parametrization[0][2])) == "<class 'scipy.stats._distn_infrastructure.rv_frozen'>"