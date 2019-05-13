import gempy as gp
import matplotlib.pyplot as plt
import pandas as pn
import numpy as np
import os
import pytest
input_path = os.path.dirname(__file__)+'/../input_data'

# ## Preparing the Python environment
#
# For modeling with GemPy, we first need to import it. We should also import any other packages we want to utilize in our Python environment.Typically, we will also require `NumPy` and `Matplotlib` when working with GemPy. At this point, we can further customize some settings as desired, e.g. the size of figures or, as we do here, the way that `Matplotlib` figures are displayed in our notebook (`%matplotlib inline`).


# These two lines are necessary only if GemPy is not installed
import sys, os
sys.path.append("../..")


@pytest.fixture(scope="module")
def load_model():
    geo_model = gp.create_model('Model_Tuto1-1')

    # Importing the data from CSV-files and setting extent and resolution
    gp.init_data(geo_model, [0, 2000., 0, 2000., 0, 2000.], [50 ,50 ,50],
          path_o = input_path+"/simple_fault_model_orientations.csv",
          path_i = input_path+"/simple_fault_model_points.csv", default_values=True)

    gp.get_data(geo_model, 'surface_points').head()
    return geo_model


@pytest.fixture(scope='module')
def map_sequential_pile(load_model):
    geo_model = load_model

    # TODO decide what I do with the layer order

    gp.map_series_to_surfaces(geo_model, {"Fault_Series": 'Main_Fault',
                                          "Strat_Series": ('Sandstone_2', 'Siltstone',
                                                           'Shale', 'Sandstone_1', 'basement')},
                                            remove_unused_series=True)

    geo_model.set_is_fault(['Fault_Series'])
    return geo_model


def test_get_data(load_model):
    geo_model = load_model
    gp.get_data(geo_model, 'orientations').head()


def test_define_sequential_pile(map_sequential_pile):
    print(map_sequential_pile.surfaces)


def test_kriging_parameters(map_sequential_pile):
    # TODO Update
    geo_model = map_sequential_pile
    #gp.get_kriging_parameters(geo_model)


def test_compute_model(interpolator_islith_isfault, map_sequential_pile):
    geo_model = map_sequential_pile
    geo_model.set_theano_graph(interpolator_islith_isfault)

    gp.compute_model(geo_model, compute_mesh=False)

    test_values = [45, 150, 2500]
    if False:
        np.save(input_path+'/test_integration_lith_block.npy', geo_model.solutions.lith_block[test_values])

    # Load model
    real_sol = np.load(input_path + '/test_integration_lith_block.npy')

    # We only compare the block because the absolute pot field I changed it
    np.testing.assert_array_almost_equal(np.round(geo_model.solutions.lith_block[test_values]), real_sol, decimal=0)

    gp.plot.plot_section(geo_model, cell_number=25,
                         direction='y', show_data=True)
    plt.savefig(os.path.dirname(__file__)+'/../figs/test_integration_lith_block')

    gp.plot.plot_scalar_field(geo_model, cell_number=25, series=1, N=15,
                              direction='y', show_data=True)

    plt.savefig(os.path.dirname(__file__)+'/../figs/test_integration_scalar')



