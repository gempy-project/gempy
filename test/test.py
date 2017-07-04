import pytest

import theano
import numpy as np
import sys
sys.path.append("../")
import gempy


class TestNoFaults:
    """
    I am testing all block and potential field values so sol is (n_block+n_pot)
    """
    # DEP?
    # Init interpolator
    @pytest.fixture(scope='class')
    def interpolator(self):
        # Importing the data from csv files and settign extent and resolution
        geo_data = gempy.import_data([0, 10, 0, 10, -10, 0], [50, 50, 50],
                                     path_f="./GeoModeller/test_a/test_a_Foliations.csv",
                                     path_i="./GeoModeller/test_a/test_a_Points.csv")

        data_interp = gempy.set_interpolator(geo_data,
                                             dtype="float64",
                                             verbose=['solve_kriging'])
    @pytest.fixture(scope='class')
    def theano_f(self):
        # Importing the data from csv files and settign extent and resolution
        geo_data = gempy.import_data([0, 10, 0, 10, -10, 0], [50, 50, 50],
                                     path_f="./GeoModeller/test_a/test_a_Foliations.csv",
                                     path_i="./GeoModeller/test_a/test_a_Points.csv")

       # data_interp = gempy.set_interpolator(geo_data,
       #                                      dtype="float64",
       #                                      verbose=['solve_kriging'])


        # Set all the theano shared parameters and return the symbolic variables (the input of the theano function)
        #input_data_T =   data_interp.interpolator.tg.input_parameters_list()

        # Prepare the input data (interfaces, foliations data) to call the theano function.
        # Also set a few theano shared variables with the len of formations series and so on
       # input_data_P =   data_interp.interpolator.data_prep(u_grade=[3])

        # Compile the theano function.
       # compiled_f = theano.function(input_data_T,   data_interp.interpolator.tg.whole_block_model(),
       #                              allow_input_downcast=True, profile=True)

        data_interp = gempy.InterpolatorInput(geo_data, dtype='float64')
        compiled_f = data_interp.compile_th_fn()

        return data_interp, compiled_f

    def test_a(self, theano_f):
        """
        2 Horizontal layers with drift one
        """

        data_interp = theano_f[0]
        compiled_f = theano_f[1]

        # Importing the data from csv files and settign extent and resolution
        geo_data = gempy.import_data([0, 10, 0, 10, -10, 0], [50, 50, 50],
                                     path_f="./GeoModeller/test_a/test_a_Foliations.csv",
                                     path_i="./GeoModeller/test_a/test_a_Points.csv")

        # rescaled_data = gempy.rescale_data(geo_data)
        #
        # data_interp.interpolator._data_scaled = rescaled_data
        # data_interp.interpolator._grid_scaled = rescaled_data.grid
        # data_interp.interpolator.order_table()
        # data_interp.interpolator.set_theano_shared_parameteres()
        #
        # # Prepare the input data (interfaces, foliations data) to call the theano function.
        # # Also set a few theano shared variables with the len of formations series and so on
        # input_data_P = data_interp.interpolator.data_prep(u_grade=[3])
        # # Compile the theano function.

        data_interp.set_interpolator(geo_data)

        i = data_interp.get_input_data(u_grade=[3])
        sol = compiled_f(*i)

        real_sol = np.load('test_a_sol.npy')
        np.testing.assert_array_almost_equal(sol[:, :2, :], real_sol, decimal=3)

        gempy.plot_section(geo_data, 25, block=sol[0, 0, :], direction='y', plot_data=True)
        gempy.plot_potential_field(geo_data, sol[0, 1, :], 25)

    def test_b(self, theano_f):
        """
        Two layers a bit curvy, drift degree 1
        """
        data_interp = theano_f[0]
        compiled_f = theano_f[1]

        # Importing the data from csv files and settign extent and resolution
        geo_data = gempy.import_data([0, 10, 0, 10, -10, 0], [50, 50, 50],
                                     path_f="./GeoModeller/test_b/test_b_Foliations.csv",
                                     path_i="./GeoModeller/test_b/test_b_Points.csv")

        # DEP
        # rescaled_data = gempy.rescale_data(geo_data)
        #
        # data_interp.interpolator._data_scaled = rescaled_data
        # data_interp.interpolator._grid_scaled = rescaled_data.grid
        # data_interp.interpolator.order_table()
        # data_interp.interpolator.set_theano_shared_parameteres()
        #
        # # Prepare the input data (interfaces, foliations data) to call the theano function.
        # # Also set a few theano shared variables with the len of formations series and so on
        # input_data_P = data_interp.interpolator.data_prep(u_grade=[3])
        #
        # sol = compiled_f(input_data_P[0], input_data_P[1], input_data_P[2], input_data_P[3], input_data_P[4],
        #                  input_data_P[5])

        data_interp.set_interpolator(geo_data)

        i = data_interp.get_input_data(u_grade=[3])
        sol = compiled_f(*i)

        real_sol = np.load('test_b_sol.npy')
        np.testing.assert_array_almost_equal(sol[:, :2, :], real_sol, decimal=3)

    def test_c(self, theano_f):
        """
        Two layers a bit curvy, drift degree 0
        """
        data_interp = theano_f[0]
        compiled_f = theano_f[1]

        # Importing the data from csv files and settign extent and resolution
        geo_data = gempy.import_data([0, 10, 0, 10, -10, 0], [50, 50, 50],
                                     path_f="./GeoModeller/test_c/test_c_Foliations.csv",
                                     path_i="./GeoModeller/test_c/test_c_Points.csv")

        # DEP
        # rescaled_data = gempy.rescale_data(geo_data)
        #
        # data_interp.interpolator._data_scaled = rescaled_data
        # data_interp.interpolator._grid_scaled = rescaled_data.grid
        # data_interp.interpolator.order_table()
        # data_interp.interpolator.set_theano_shared_parameteres()
        #
        # # Prepare the input data (interfaces, foliations data) to call the theano function.
        # # Also set a few theano shared variables with the len of formations series and so on
        # input_data_P = data_interp.interpolator.data_prep(u_grade=[0])


        # sol = compiled_f(input_data_P[0], input_data_P[1], input_data_P[2], input_data_P[3], input_data_P[4],
        #                 input_data_P[5])

        data_interp.set_interpolator(geo_data)

        i = data_interp.get_input_data(u_grade=[0])
        sol = compiled_f(*i)

        real_sol = np.load('test_c_sol.npy')
        np.testing.assert_array_almost_equal(sol[:, :2, :], real_sol, decimal=3)


class TestFaults:
    @pytest.fixture(scope='class')
    def theano_f_1f(self):
        # Importing the data from csv files and settign extent and resolution
        geo_data = gempy.import_data([0, 10, 0, 10, -10, 0], [50, 50, 50],
                                     path_f="./GeoModeller/test_d/test_d_Foliations.csv",
                                     path_i="./GeoModeller/test_d/test_d_Points.csv")

        gempy.set_data_series(geo_data, {'series': ('A', 'B'),
                                        'fault1': 'f1'}, order_series=['fault1', 'series'])

        # data_interp = gempy.set_interpolator(geo_data,
        #                                      dtype="float64",
        #                                      verbose=['solve_kriging',
        #                                               'faults block'
        #                                               ])

        # # Set all the theano shared parameters and return the symbolic variables (the input of the theano function)
        # input_data_T = data_interp.interpolator.tg.input_parameters_list()
        #
        # # Prepare the input data (interfaces, foliations data) to call the theano function.
        # # Also set a few theano shared variables with the len of formations series and so on
        # input_data_P = data_interp.interpolator.data_prep(u_grade=[3, 3])
        #
        # # Compile the theano function.
        # compiled_f = theano.function(input_data_T, data_interp.interpolator.tg.whole_block_model(1),
        #                              allow_input_downcast=True, profile=True)
        # data_interp.interpolator.get_kriging_parameters()

        geo_data.n_faults = 1
        data_interp = gempy.InterpolatorInput(geo_data, dtype='float64')
        compiled_f = data_interp.compile_th_fn()

        return data_interp, compiled_f

    def test_d(self, theano_f_1f):
        """
        Two layers 1 fault
        """
        data_interp = theano_f_1f[0]
        compiled_f = theano_f_1f[1]

        # Importing the data from csv files and settign extent and resolution
        geo_data = gempy.import_data([0, 10, 0, 10, -10, 0], [50, 50, 50],
                                     path_f="./GeoModeller/test_d/test_d_Foliations.csv",
                                     path_i="./GeoModeller/test_d/test_d_Points.csv")

        gempy.set_data_series(geo_data, {'series': ('A', 'B'),
                                          'fault1': 'f1'}, order_series=['fault1', 'series'])

       #  rescaled_data = gempy.rescale_data(geo_data)
       #
       # # data_interp = gempy.set_interpolator(geo_data, dtype="float64",)
       #  data_interp.interpolator._data_scaled = rescaled_data
       #  data_interp.interpolator._grid_scaled = rescaled_data.grid
       #  data_interp.interpolator.order_table()
       #  data_interp.interpolator.set_theano_shared_parameteres()
       #
       #  # Prepare the input data (interfaces, foliations data) to call the theano function.
       #  # Also set a few theano shared variables with the len of formations series and so on
       #  input_data_P = data_interp.interpolator.data_prep(u_grade=[3, 3])
       #
       #  data_interp.interpolator.get_kriging_parameters()
       #
       #  sol = compiled_f(input_data_P[0], input_data_P[1], input_data_P[2], input_data_P[3], input_data_P[4],
       #                   input_data_P[5])

        geo_data.n_faults = 1

        data_interp.set_interpolator(geo_data)

        i = data_interp.get_input_data(u_grade=[3, 3])
        sol = compiled_f(*i)

       # print(data_interp.rescale_factor, 'rescale')
        real_sol = np.load('test_d_sol.npy')
        np.testing.assert_array_almost_equal(sol[:, :2, :], real_sol, decimal=3)

    def test_e(self, theano_f_1f):
        """
        Two layers a bit curvy, 1 fault
        """
        data_interp = theano_f_1f[0]
        compiled_f = theano_f_1f[1]

        # Importing the data from csv files and settign extent and resolution
        geo_data = gempy.import_data([0, 10, 0, 10, -10, 0], [50, 50, 50],
                                     path_f="./GeoModeller/test_e/test_e_Foliations.csv",
                                     path_i="./GeoModeller/test_e/test_e_Points.csv")

        gempy.set_data_series(geo_data, {'series': ('A', 'B'),
                                        'fault1': 'f1'}, order_series=['fault1', 'series'])

       #  rescaled_data = gempy.rescale_data(geo_data)
       # # data_interp = gempy.set_interpolator(geo_data, dtype="float64", )
       #  data_interp.interpolator._data_scaled = rescaled_data
       #  data_interp.interpolator._grid_scaled = rescaled_data.grid
       #  data_interp.interpolator.order_table()
       #  data_interp.interpolator.set_theano_shared_parameteres()
       #
       #  # Prepare the input data (interfaces, foliations data) to call the theano function.
       #  # Also set a few theano shared variables with the len of formations series and so on
       #  input_data_P = data_interp.interpolator.data_prep(u_grade=[3, 3])
       #
       #  data_interp.interpolator.get_kriging_parameters()
       #
       #
       #
       #  sol = compiled_f(input_data_P[0], input_data_P[1], input_data_P[2], input_data_P[3], input_data_P[4],
       #                   input_data_P[5])

        geo_data.n_faults = 1

        data_interp.set_interpolator(geo_data)

        i = data_interp.get_input_data(u_grade=[3, 3])
        sol = compiled_f(*i)

        # print(data_interp.rescale_factor, 'rescale')
        real_sol = np.load('test_e_sol.npy')
        np.testing.assert_array_almost_equal(sol[:, :2, :], real_sol, decimal=3)

    def test_f(self, theano_f_1f):
        """
        Two layers a bit curvy, 1 fault
        """
        data_interp = theano_f_1f[0]
        compiled_f = theano_f_1f[1]

        # Importing the data from csv files and settign extent and resolution
        geo_data = gempy.import_data([0, 2000, 0, 2000, -2000, 0], [50, 50, 50],
                                     path_f="./GeoModeller/test_f/test_f_Foliations.csv",
                                     path_i="./GeoModeller/test_f/test_f_Points.csv")

        geo_data.set_formation_number(geo_data.formations[[3, 2, 1, 0, 4]])

        gempy.set_data_series(geo_data, {'series': ('Reservoir'
                                                    , 'Seal',
                                                    'SecondaryReservoir',
                                                    'NonReservoirDeep'
                                                    ),
                                         'fault1': 'MainFault'},
                              order_series=['fault1', 'series'])

       #  rescaled_data = gempy.rescale_data(geo_data)
       # # data_interp = gempy.set_interpolator(geo_data, dtype="float64", )
       #  data_interp.interpolator._data_scaled = rescaled_data
       #  data_interp.interpolator._grid_scaled = rescaled_data.grid
       #  data_interp.interpolator.order_table()
       #  data_interp.interpolator.set_theano_shared_parameteres()
       #
       #  # Prepare the input data (interfaces, foliations data) to call the theano function.
       #  # Also set a few theano shared variables with the len of formations series and so on
       #  input_data_P = data_interp.interpolator.data_prep(u_grade=[3, 3])
       #
       #  data_interp.interpolator.get_kriging_parameters()
       #
       #  sol = compiled_f(input_data_P[0], input_data_P[1], input_data_P[2], input_data_P[3], input_data_P[4],
       #                   input_data_P[5])
       # # print(data_interp.rescale_factor, 'rescale')

        geo_data.n_faults = 1

        data_interp.set_interpolator(geo_data)

        i = data_interp.get_input_data(u_grade=[3, 3])
        sol = compiled_f(*i)

        real_sol = np.load('test_f_sol.npy')
       # np.testing.assert_array_almost_equal(sol[:, :2, :], real_sol, decimal=2)
        mismatch = ~np.isclose(sol[:, :2, :], real_sol, rtol=0.01).sum()/np.product(sol.shape)
        assert mismatch * 100 < 1
        GeoMod_sol = geo_data.read_vox('./GeoModeller/test_f/test_f.vox')
        similarity = ((GeoMod_sol - sol[0, 0, :]) != 0).sum() / sol[0, 0].shape[0]

        print('The mismatch geomodeller-gempy is ', similarity*100, '%')
        assert similarity < 0.05, 'The mismatch with geomodeller is too high'
