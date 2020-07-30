# Importing GemPy
import gempy as gp
from gempy.assets.geophysics import MagneticsPreprocessing

# Importing auxiliary libraries
import numpy as np
import pytest


def test_magnetics_api(interpolator_magnetics):
    # TODO add the check

    geo_model = gp.create_model('test_center_grid_slicing')
    geo_model.set_default_surfaces()
    geo_model.add_surface_points(X=-1, Y=0, Z=0, surface='surface1')
    geo_model.add_surface_points(X=1, Y=0, Z=0, surface='surface1')
    geo_model.add_orientations(X=0, Y=0, Z=0, surface='surface1', pole_vector=(0, 0, 1))
    geo_model._surfaces.add_surfaces_values([0.037289, 0.0297], ['susceptibility'])

    # needed constants
    mu_0 = 4.0 * np.pi * 10e-7  # magnetic permeability in free space [N/A^2]
    cm = mu_0 / (4.0 * np.pi)  # constant for SI unit
    incl = 77.0653  # NOAA
    decl = 6.8116
    B_ext = 52819.8506939139e-9  # T

    geo_model.set_regular_grid(extent=[-5, 5, -5, 5, -5, 5], resolution=[5, 5, 5])
    geo_model.set_centered_grid(np.array([[0, 0, 0]]), resolution=[10, 10, 15], radius=5000)
    geo_model.set_theano_function(interpolator_magnetics)
    geo_model._interpolator.set_theano_shared_magnetics(V='auto', pos_magnetics=1,
                                                        incl= incl, decl=decl, B_ext=B_ext)

    gp.compute_model(geo_model)
    print(geo_model._interpolator.theano_graph.lg0.get_value())
    print(geo_model.solutions.fw_magnetics)
    np.testing.assert_almost_equal(geo_model.solutions.fw_magnetics,
                                   np.array([473.7836]), decimal=4)
    return geo_model


@pytest.fixture(scope="module")
def test_magnetics_no_regular_grid(interpolator_magnetics):
    # TODO add the check

    geo_model = gp.create_model('test_center_grid_slicing')
    geo_model.set_default_surfaces()
    geo_model.add_surface_points(X=-1, Y=0, Z=0, surface='surface1')
    geo_model.add_surface_points(X=1, Y=0, Z=0, surface='surface1')
    geo_model.add_orientations(X=0, Y=0, Z=0, surface='surface1', pole_vector=(0, 0, 1))
    geo_model._surfaces.add_surfaces_values([0.037289, 0.0297], ['susceptibility'])

    # needed constants
    mu_0 = 4.0 * np.pi * 10e-7  # magnetic permeability in free space [N/A^2]
    cm = mu_0 / (4.0 * np.pi)  # constant for SI unit
    incl = 77.0653  # NOAA
    decl = 6.8116
    B_ext = 52819.8506939139e-9  # T

    geo_model.set_centered_grid(np.array([0, 0, 0]), resolution=[10, 10, 15], radius=5000)

    Vmodel = MagneticsPreprocessing(geo_model._grid.centered_grid).set_Vs_kernel()
    # gp.set_interpolator(geo_model, output=['magnetics'])
    geo_model.set_theano_function(interpolator_magnetics)
    geo_model._interpolator.set_theano_shared_magnetics(V= Vmodel, pos_magnetics=1,
                                                        incl= incl, decl=decl, B_ext=B_ext)

    # geo_model.interpolator.theano_graph.V.set_value(Vmodel)
    # geo_model.interpolator.theano_graph.incl.set_value(incl)
    # geo_model.interpolator.theano_graph.decl.set_value(decl)
    # geo_model.interpolator.theano_graph.B_ext.set_value(B_ext)

    gp.compute_model(geo_model)
    np.testing.assert_almost_equal(geo_model.solutions.fw_magnetics,
                                   np.array([473.7836]), decimal=4)
    gp.compute_model(geo_model)
    return geo_model


def test_center_grid_slicing(test_magnetics_no_regular_grid):
    geo_model = test_magnetics_no_regular_grid

    geo_model.set_centered_grid(np.array([[0, 0, 0],
                                [1, 1, 1]]), resolution=[10, 10, 15], radius=5000)

    gp.compute_model(geo_model)
    print(geo_model._interpolator.theano_graph.lg0.get_value())
