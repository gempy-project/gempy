

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
def geo_model(interpolator):
    geo_model = gp.create_model('Test_uncomformities')

    # Importing the data from CSV-files and setting extent and resolution
    gp.init_data(geo_model, [0, 10., 0, 2., 0, 5.], [100, 3, 100],
                 path_o=input_path + '/05_toy_fold_unconformity_orientations.csv',
                 path_i=input_path + '/05_toy_fold_unconformity_interfaces.csv', default_values=True);

    gp.map_stack_to_surfaces(geo_model,
                             {"Flat_Series": 'Flat',
                               "Inclined_Series": 'Inclined',
                               "Fold_Series": ('Basefold', 'Topfold', 'basement')})

    # Create the theano model
    geo_model.set_theano_function(interpolator)

    return geo_model


def test_all_erosion(geo_model):
    sol = gp.compute_model(geo_model, compute_mesh=True)
    gp.plot.plot_2d(geo_model, cell_number=2)

    gp.plot_2d(geo_model, cell_number=[2],
               regular_grid=geo_model.solutions.mask_matrix_pad[0],
               show_data=True, kwargs_regular_grid={'cmap': 'gray', 'norm': None})

    gp.plot_2d(geo_model, cell_number=[2],
               regular_grid=geo_model.solutions.mask_matrix_pad[1],
               show_data=True, kwargs_regular_grid={'cmap': 'gray', 'norm': None})

    gp.plot_2d(geo_model, cell_number=[2],
               regular_grid=geo_model.solutions.mask_matrix_pad[2],
               show_data=True, kwargs_regular_grid={'cmap': 'gray', 'norm': None})

    if save:
        np.save(os.path.dirname(__file__)+'/all_ero', sol.lith_block)

    check = np.load(os.path.dirname(__file__)+'/all_ero.npy')
    np.testing.assert_allclose(sol.lith_block, check)
    # plt.savefig(os.path.dirname(__file__)+'/all_ero')

    p3d = gp.plot_3d(geo_model, show_surfaces=True, show_data=True,
                     image=True,
                     kwargs_plot_structured_grid={'opacity': .2})

    print(sol)


def test_one_onlap(geo_model):
    geo_model.set_bottom_relation('Inclined_Series', bottom_relation='Onlap')
    geo_model.set_bottom_relation('Flat_Series', bottom_relation='Erosion')
    sol = gp.compute_model(geo_model, compute_mesh=True)
    gp.plot.plot_2d(geo_model, cell_number=2)

    gp.plot_2d(geo_model, cell_number=[2],
               regular_grid=geo_model.solutions.mask_matrix_pad[0],
               show_data=True, kwargs_regular_grid={'cmap': 'gray', 'norm': None})

    gp.plot_2d(geo_model, cell_number=[2],
               regular_grid=geo_model.solutions.mask_matrix_pad[1],
               show_data=True, kwargs_regular_grid={'cmap': 'gray', 'norm': None})

    gp.plot_2d(geo_model, cell_number=[2],
               regular_grid=geo_model.solutions.mask_matrix_pad[2],
               show_data=True, kwargs_regular_grid={'cmap': 'gray', 'norm': None})

    p3d = gp.plot_3d(geo_model, show_surfaces=True, show_data=True,
                     image=True,
                     kwargs_plot_structured_grid={'opacity': .2})

    if save:
        np.save(os.path.dirname(__file__)+'/one_onlap', sol.lith_block)

    check = np.load(os.path.dirname(__file__)+'/one_onlap.npy')
    np.testing.assert_allclose(sol.lith_block, check)

    print(sol)


def test_two_onlap(geo_model):
    geo_model.set_bottom_relation(['Flat_Series', 'Inclined_Series'], bottom_relation='Onlap')
    sol = gp.compute_model(geo_model, compute_mesh=True)
    gp.plot.plot_2d(geo_model, cell_number=2)

    gp.plot_2d(geo_model, cell_number=[2],
               regular_grid=geo_model.solutions.mask_matrix_pad[0],
               show_data=True, kwargs_regular_grid={'cmap': 'gray', 'norm': None})

    gp.plot_2d(geo_model, cell_number=[2],
               regular_grid=geo_model.solutions.mask_matrix_pad[1],
               show_data=True, kwargs_regular_grid={'cmap': 'gray', 'norm': None})

    gp.plot_2d(geo_model, cell_number=[2],
               regular_grid=geo_model.solutions.mask_matrix_pad[2],
               show_data=True, kwargs_regular_grid={'cmap': 'gray', 'norm': None})

    p3d = gp.plot_3d(geo_model, show_surfaces=True, show_data=True,
                     image=True,
                     kwargs_plot_structured_grid={'opacity': .2})

    if save:
        np.save(os.path.dirname(__file__)+'/two_onlap', sol.lith_block)

    check = np.load(os.path.dirname(__file__)+'/two_onlap.npy')
    np.testing.assert_allclose(sol.lith_block, check)

    plt.savefig(os.path.dirname(__file__)+'/two_onlap')
    print(sol)


def test_masked_marching_cubes():
    cwd = os.path.dirname(__file__)
    data_path = cwd + '/../../../examples/'
    geo_model = gp.load_model(r'Tutorial_ch1-8_Onlap_relations',
                              path=data_path + 'data/gempy_models/Tutorial_ch1-8_Onlap_relations',
                              recompile=True)

    geo_model.set_regular_grid([-200, 1000, -500, 500, -1000, 0], [50, 50, 50])
   # geo_model.set_topography(d_z=np.array([-600, -100]))

    s = gp.compute_model(geo_model, compute_mesh=True, debug=False)

    gp.plot.plot_2d(geo_model, cell_number=2)

    gp.plot_2d(geo_model, cell_number=[2],
               regular_grid=geo_model.solutions.mask_matrix_pad[0],
               show_data=True, kwargs_regular_grid={'cmap': 'gray', 'norm': None})

    gp.plot_2d(geo_model, cell_number=[2],
               regular_grid=geo_model.solutions.mask_matrix_pad[1],
               show_data=True, kwargs_regular_grid={'cmap': 'gray', 'norm': None})

    gp.plot_2d(geo_model, cell_number=[2],
               regular_grid=geo_model.solutions.mask_matrix_pad[2],
               show_data=True, kwargs_regular_grid={'cmap': 'gray', 'norm': None})

    gp.plot_2d(geo_model, cell_number=[2],
               regular_grid=geo_model.solutions.mask_matrix_pad[3],
               show_data=True, kwargs_regular_grid={'cmap': 'gray', 'norm': None})

    p3d = gp.plot_3d(geo_model, show_surfaces=True, show_data=True,
                     image=True,
                     kwargs_plot_structured_grid={'opacity': .2})



