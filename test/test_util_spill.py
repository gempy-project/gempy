import pytest
import numpy as np
import sys, os
sys.path.append("../..")
import gempy as gp
from scipy.spatial import distance
import gempy.utils.spill_analysis as sp

input_path = os.path.dirname(__file__)+'/../notebooks'

@pytest.fixture
def spill_geodata():
    # initialize geo_data object

    sys.path.insert(0, input_path)

    geo_data = gp.create_data([0,1000,0,1000,0,1000],[50,50,50],
                         path_o = input_path+"/input_data/simple_dome_orientations1.csv",
                         path_i = input_path+"/input_data/simple_dome_points1.csv")

    gp.set_series(geo_data, {"Dome": ('Shale')},
                  order_series=["Dome", ], order_formations=['Shale', ])

    return geo_data

@pytest.fixture
def spill_interp(spill_geodata):
    interp_data = gp.InterpolatorData(spill_geodata,u_grade=[3],
                                  output='gradients', dtype='float64', compile_theano=True)
    return interp_data

@pytest.fixture
def spill_model(spill_interp):
    lith_sol = gp.compute_model(spill_interp)[0]
    return lith_sol

@pytest.fixture
def spill_surface(spill_geodata, spill_interp, spill_model):
    seal_n = spill_geodata.formations.loc['Shale', 'formation_number']
    seal_surf_v, seal_surf_s = gp.get_surfaces(spill_interp, spill_model[1], potential_fault=None, n_formation=seal_n)
    return seal_surf_v

#lith_block = lith_sol[0] # lithology block
#pot_field  = lith_sol[1] # potential field
#GX         = lith_sol[2] # gradient field in X-direction
#GY         = lith_sol[3] # gradient field in Y-direction
#GZ         = lith_sol[4] # gradient field in Z-direction

def test_get_surface_extrema(spill_geodata, spill_surface, spill_model):
    sp.get_surface_extrema(spill_geodata, spill_surface, spill_model[2], spill_model[3])

def test_plot_surface_extrema(spill_geodata, spill_surface, spill_model):
    sp.plot_surface_extrema(spill_geodata, spill_surface, spill_model[2], spill_model[3])

def test_get_highest_max(spill_geodata, spill_surface, spill_model):
    h_max_test = np.array([520.,480.,700.92750549])
    highest_max = sp.get_highest_max(spill_geodata, spill_surface, spill_model[2], spill_model[3])
    assert distance.euclidean(h_max_test, highest_max) < 50, "Significant mismatch in determination of highest surface maximum."

def test_get_highest_saddle(spill_geodata, spill_surface, spill_model):
    h_saddle_test = np.array([660.,340.,585.688591])
    highest_saddle = sp.get_highest_saddle_point(spill_geodata, spill_surface, spill_model[2], spill_model[3])
    assert distance.euclidean(h_saddle_test, highest_saddle) < 50, "Significant mismatch in determination of highest surface saddle point."

def test_get_gradient_minima(spill_geodata, spill_model):
    lith_sol = spill_model
    sp.get_gradient_minima(spill_geodata, lith_sol[2],lith_sol[3],lith_sol[4], direction='z', ref='x')

def test_get_gradmin_intersect(spill_geodata, spill_surface, spill_model):
    lith_sol = spill_model
    gradmin = sp.get_gradient_minima(spill_geodata, lith_sol[2], lith_sol[3], lith_sol[4], direction='z', ref='x')
    sp.get_gradmin_intersect(spill_geodata, spill_surface, gradmin)
