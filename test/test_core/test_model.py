import gempy as gp
import matplotlib.pyplot as plt
import pandas as pn
import numpy as np
import os
import pytest
input_path = os.path.dirname(__file__)+'/../input_data'

# ## Preparing the Python environment
#
# For modeling with GemPy, we first need to import it. We should also import any other packages we want to
# utilize in our Python environment.Typically, we will also require `NumPy` and `Matplotlib` when working
# with GemPy. At this point, we can further customize some settings as desired, e.g. the size of figures or,
# as we do here, the way that `Matplotlib` figures are displayed in our notebook (`%matplotlib inline`).


# These two lines are necessary only if GemPy is not installed
import sys, os
sys.path.append("../..")


@pytest.fixture(scope="module")
def load_model():
    verbose = False
    geo_model = gp.create_model('Model_Tuto1-1')

    # Importing the data from CSV-files and setting extent and resolution
    gp.init_data(geo_model, [0, 2000., 0, 2000., 0, 2000.], [50 ,50 ,50],
                 path_o=input_path+"/simple_fault_model_orientations.csv",
                 path_i=input_path+"/simple_fault_model_points.csv", default_values=True)

    df_cmp_i = gp.get_data(geo_model, 'surface_points')
    df_cmp_o = gp.get_data(geo_model, 'orientations')

    df_o = pn.read_csv(input_path + "/simple_fault_model_orientations.csv")
    df_i = pn.read_csv(input_path + "/simple_fault_model_points.csv")

    assert not df_cmp_i.empty, 'data was not set to dataframe'
    assert not df_cmp_o.empty, 'data was not set to dataframe'
    assert df_cmp_i.shape[0] == df_i.shape[0], 'data was not set to dataframe'
    assert df_cmp_o.shape[0] == df_o.shape[0], 'data was not set to dataframe'

    if verbose:
        gp.get_data(geo_model, 'surface_points').head()

    return geo_model


def test_load_model_df():

    verbose = True
    df_i = pn.DataFrame(np.random.randn(6,3), columns='X Y Z'.split())
    df_i['formation'] = ['surface_1' for _ in range(3)] + ['surface_2' for _ in range(3)]

    df_o = pn.DataFrame(np.random.randn(6,6), columns='X Y Z azimuth dip polarity'.split())
    df_o['formation'] = ['surface_1' for _ in range(3)] + ['surface_2' for _ in range(3)]

    geo_model = gp.create_model('test')
    # Importing the data directly from the dataframes
    gp.init_data(geo_model, [0, 2000., 0, 2000., 0, 2000.], [50, 50, 50],
                 surface_points_df=df_i, orientations_df=df_o, default_values=True)

    df_cmp_i = gp.get_data(geo_model, 'surface_points')
    df_cmp_o = gp.get_data(geo_model, 'orientations')

    if verbose:
        print(df_cmp_i.head())
        print(df_cmp_o.head())

    assert not df_cmp_i.empty, 'data was not set to dataframe'
    assert not df_cmp_o.empty, 'data was not set to dataframe'
    assert df_cmp_i.shape[0] == 6, 'data was not set to dataframe'
    assert df_cmp_o.shape[0] == 6, 'data was not set to dataframe'

    # try without the default_values command

    geo_model = gp.create_model('test')
    # Importing the data directly from the dataframes
    gp.init_data(geo_model, [0, 2000., 0, 2000., 0, 2000.], [50 ,50 ,50],
                 surface_points_df=df_i, orientations_df=df_o)

    df_cmp_i2 = gp.get_data(geo_model, 'surface_points')
    df_cmp_o2 = gp.get_data(geo_model, 'orientations')

    if verbose:
        print(df_cmp_i2.head())
        print(df_cmp_o2.head())

    assert not df_cmp_i2.empty, 'data was not set to dataframe'
    assert not df_cmp_o2.empty, 'data was not set to dataframe'
    assert df_cmp_i2.shape[0] == 6, 'data was not set to dataframe'
    assert df_cmp_o2.shape[0] == 6, 'data was not set to dataframe'

    return geo_model


@pytest.fixture(scope='module')
def map_sequential_pile(load_model):
    geo_model = load_model

    # TODO decide what I do with the layer order

    gp.map_stack_to_surfaces(geo_model, {"Fault_Series": 'Main_Fault',
                                          "Strat_Series": ('Sandstone_2', 'Siltstone',
                                                           'Shale', 'Sandstone_1', 'basement')},
                             remove_unused_series=True)

    geo_model.set_is_fault(['Fault_Series'])
    return geo_model


def test_get_data(load_model):
    geo_model = load_model
    return gp.get_data(geo_model, 'orientations').head()


def test_define_sequential_pile(map_sequential_pile):
    print(map_sequential_pile._surfaces)


def test_compute_model(interpolator, map_sequential_pile):
    geo_model = map_sequential_pile
    geo_model.set_theano_graph(interpolator)

    gp.compute_model(geo_model, compute_mesh=False)

    test_values = [45, 150, 2500]
    if False:
        np.save(input_path+'/test_integration_lith_block.npy', geo_model.solutions.lith_block[test_values])

    # Load model
    real_sol = np.load(input_path + '/test_integration_lith_block.npy')

    # We only compare the block because the absolute pot field I changed it
    np.testing.assert_array_almost_equal(np.round(geo_model.solutions.lith_block[test_values]), real_sol, decimal=0)

    gp.plot.plot_2d(geo_model, cell_number=25,
                    direction='y', show_data=True)
    plt.savefig(os.path.dirname(__file__)+'/../figs/test_integration_lith_block')

    gp.plot.plot_2d(geo_model, cell_number=25, series_n=1, N=15, show_scalar=True,
                    direction='y', show_data=True)

    plt.savefig(os.path.dirname(__file__)+'/../figs/test_integration_scalar')


def test_kriging_mutation(interpolator, map_sequential_pile):
    geo_model = map_sequential_pile
    geo_model.set_theano_graph(interpolator)

    gp.compute_model(geo_model, compute_mesh=False)
    gp.plot.plot_2d(geo_model, cell_number=25, show_scalar=True, series_n=1, N=15,
                    direction='y', show_data=True)
    print(geo_model.solutions.lith_block, geo_model._additional_data)
    #plt.savefig(os.path.dirname(__file__)+'/figs/test_kriging_mutation')

    geo_model.modify_kriging_parameters('range', 1)
    geo_model.modify_kriging_parameters('drift equations', [0, 3])

    print(geo_model.solutions.lith_block, geo_model._additional_data)
    # copy dataframe before interpolator is calculated
    pre = geo_model._additional_data.kriging_data.df.copy()

    gp.set_interpolator(geo_model, compile_theano=True,
                        theano_optimizer='fast_compile', update_kriging=False)
    gp.compute_model(geo_model, compute_mesh=False)

    gp.plot.plot_2d(geo_model, cell_number=25, series_n=1, N=15, show_boundaries=False,
                    direction='y', show_data=True, show_lith=True)

    print(geo_model.solutions.lith_block, geo_model._additional_data)
    plt.savefig(os.path.dirname(__file__)+'/../figs/test_kriging_mutation2')
    assert geo_model._additional_data.kriging_data.df['range'][0] == pre['range'][0]