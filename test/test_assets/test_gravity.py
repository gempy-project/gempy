# These two lines are necessary only if GemPy is not installed
import sys, os
sys.path.append("../..")

# Importing GemPy
import gempy as gp
from gempy.assets.geophysics import MagneticsPreprocessing

# Importing auxiliary libraries
import numpy as np
import matplotlib.pyplot as plt
import pytest


def test_gravity():
    geo_model = gp.create_model('2-layers')
    gp.init_data(geo_model, extent=[0, 12, -2, 2, 0, 4], resolution=[500, 1, 500])
    geo_model.add_surfaces('surface 1')
    geo_model.add_surfaces('surface 2')
    geo_model.add_surfaces('basement')
    dz = geo_model.grid.regular_grid.dz
    geo_model.surfaces.add_surfaces_values([dz, 0, 0], ['dz'])
    geo_model.surfaces.add_surfaces_values([2.6, 2.4, 3.2], ['density'])
    geo_model.add_surface_points(3, 0, 3.05, 'surface 1')
    geo_model.add_surface_points(9, 0, 3.05, 'surface 1')

    geo_model.add_surface_points(3, 0, 1.02, 'surface 2')
    geo_model.add_surface_points(9, 0, 1.02, 'surface 2')

    geo_model.add_orientations(6, 0, 4, 'surface 1', [0, 0, 1])
    device_loc = np.array([[6, 0, 4]])

    geo_model.set_centered_grid(device_loc, resolution=[10, 10, 100], radius=16000)
    gp.set_interpolator(geo_model, output=['gravity'], pos_density=2, gradient=True,
                        theano_optimizer='fast_compile')

    gp.compute_model(geo_model, set_solutions=True, compute_mesh=False)
    print(geo_model.solutions.fw_gravity)
    np.testing.assert_almost_equal(geo_model.solutions.fw_gravity,
                                   np.array([-9291.8003]), decimal=4)
