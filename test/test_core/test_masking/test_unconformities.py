

# These two lines are necessary only if GemPy is not installed
import sys, os
sys.path.append("../..")
input_path = os.path.dirname(__file__)+'/../../input_data'

# Importing GemPy
import gempy as gp


# Importing auxiliary libraries
import numpy as np
import pandas as pn
import matplotlib.pyplot as plt
import pytest

save = False


@pytest.fixture(scope="module")
def geo_model():
    geo_model = gp.create_model('Test_uncomformities')

    # Importing the data from CSV-files and setting extent and resolution
    gp.init_data(geo_model, [0, 10., 0, 2., 0, 5.], [100, 3, 100],
                 path_o=input_path + '/05_toy_fold_unconformity_orientations.csv',
                 path_i=input_path + '/05_toy_fold_unconformity_interfaces.csv', default_values=True);

    gp.map_series_to_surfaces(geo_model,
                              {"Flat_Series": 'Flat',
                               "Inclined_Series": 'Inclined',
                               "Fold_Series": ('Basefold', 'Topfold', 'basement')})

    # Create the theano model
    gp.set_interpolator(geo_model, theano_optimizer='fast_compile')

    return geo_model


def test_all_erosion(geo_model):
    sol = gp.compute_model(geo_model, compute_mesh=False)
    gp.plot.plot_section(geo_model, cell_number=2)

    if save:
        np.save(os.path.dirname(__file__)+'/all_ero', sol.lith_block)

    check = np.load(os.path.dirname(__file__)+'/all_ero.npy')
    np.testing.assert_allclose(sol.lith_block, check)
    plt.savefig(os.path.dirname(__file__)+'/all_ero')

    print(sol)


def test_one_onlap(geo_model):
    geo_model.set_bottom_relation('Inclined_Series', bottom_relation='Onlap')
    geo_model.set_bottom_relation('Flat_Series', bottom_relation='Erosion')
    sol = gp.compute_model(geo_model, compute_mesh=False)
    gp.plot.plot_section(geo_model, cell_number=2)

    if save:
        np.save(os.path.dirname(__file__)+'/one_onlap', sol.lith_block)

    check = np.load(os.path.dirname(__file__)+'/one_onlap.npy')
    np.testing.assert_allclose(sol.lith_block, check)

    plt.savefig(os.path.dirname(__file__)+'/one_onlap')

    print(sol)


def test_two_onlap(geo_model):
    geo_model.set_bottom_relation(['Flat_Series', 'Inclined_Series'], bottom_relation='Onlap')
    sol = gp.compute_model(geo_model, compute_mesh=False)
    gp.plot.plot_section(geo_model, cell_number=2)

    if save:
        np.save(os.path.dirname(__file__)+'/two_onlap', sol.lith_block)

    check = np.load(os.path.dirname(__file__)+'/two_onlap.npy')
    np.testing.assert_allclose(sol.lith_block, check)

    plt.savefig(os.path.dirname(__file__)+'/two_onlap')
    print(sol)

