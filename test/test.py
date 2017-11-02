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
    # # DEP?
    # # Init interpolator
    # @pytest.fixture(scope='class')
    # def interpolator(self):
    #     # Importing the data from csv files and settign extent and resolution
    #     geo_data = gempy.import_data([0, 10, 0, 10, -10, 0], [50, 50, 50],
    #                                  path_f="./GeoModeller/test_a/test_a_Foliations.csv",
    #                                  path_i="./GeoModeller/test_a/test_a_Points.csv")
    #
    #     data_interp = gempy.InterpolatorInput(geo_data,
    #                                          dtype="float64",
    #                                          verbose=['solve_kriging'])
    @pytest.fixture(scope='class')
    def theano_f(self):
        # Importing the data from csv files and settign extent and resolution
        geo_data = gempy.create_data([0, 10, 0, 10, -10, 0], [50, 50, 50],
                                     path_f="./GeoModeller/test_a/test_a_Foliations.csv",
                                     path_i="./GeoModeller/test_a/test_a_Points.csv")

        interp_data = gempy.InterpolatorInput(geo_data, dtype='float64')

        return interp_data

    def test_a(self, theano_f):
        """
        2 Horizontal layers with drift one
        """

        # Importing the data from csv files and settign extent and resolution
        geo_data = gempy.create_data([0, 10, 0, 10, -10, 0], [50, 50, 50],
                                     path_f="./GeoModeller/test_a/test_a_Foliations.csv",
                                     path_i="./GeoModeller/test_a/test_a_Points.csv")

        interp_data = theano_f

        # Create new interp data without compiling theano
        new_interp_data = gempy.InterpolatorInput(geo_data, dtype='float64', compile_theano=False)
        # Updating the interp data which has theano compiled
        interp_data.update_interpolator(new_interp_data.geo_data_res)

        # Compute model
        sol = gempy.compute_model(interp_data, u_grade=[3])

        # Load model
        real_sol = np.load('test_a_sol.npy')

        # We only compare the block because the absolute pot field I changed it
        np.testing.assert_array_almost_equal(sol[0][0, :], real_sol[0][0, :], decimal=3)

        # Checking that the plots do not rise errors
        gempy.plot_section(geo_data, sol[0][0, :], 25, direction='y', plot_data=True)
        gempy.plot_potential_field(geo_data, sol[0][1, :], 25)

    def test_b(self, theano_f):
        """
        Two layers a bit curvy, drift degree 1
        """

        # Importing the data from csv files and settign extent and resolution
        geo_data = gempy.create_data([0, 10, 0, 10, -10, 0], [50, 50, 50],
                                     path_f="./GeoModeller/test_b/test_b_Foliations.csv",
                                     path_i="./GeoModeller/test_b/test_b_Points.csv")

        interp_data = theano_f

        # Create new interp data without compiling theano
        new_interp_data = gempy.InterpolatorInput(geo_data, dtype='float64', compile_theano=False)
        # Updating the interp data which has theano compiled
        interp_data.update_interpolator(new_interp_data.geo_data_res)

        # Compute model
        sol = gempy.compute_model(interp_data, u_grade=[3])

        # Load model
        real_sol = np.load('test_b_sol.npy')

        # We only compare the block because the absolute pot field I changed it
        np.testing.assert_array_almost_equal(sol[0][0, :], real_sol[0][0, :], decimal=3)

        # Checking that the plots do not rise errors
        gempy.plot_section(geo_data, sol[0][0, :], 25, direction='y', plot_data=True)
        gempy.plot_potential_field(geo_data, sol[0][1, :], 25)


    def test_c(self, theano_f):
        """
        Two layers a bit curvy, drift degree 0
        """

        # Importing the data from csv files and settign extent and resolution
        geo_data = gempy.create_data([0, 10, 0, 10, -10, 0], [50, 50, 50],
                                     path_f="./GeoModeller/test_c/test_c_Foliations.csv",
                                     path_i="./GeoModeller/test_c/test_c_Points.csv")

        interp_data = theano_f

        # Create new interp data without compiling theano
        new_interp_data = gempy.InterpolatorInput(geo_data, dtype='float64', compile_theano=False)
        # Updating the interp data which has theano compiled
        interp_data.update_interpolator(new_interp_data.geo_data_res)

        # Compute model
        sol = gempy.compute_model(interp_data, u_grade=[0])

        # Load model
        real_sol = np.load('test_c_sol.npy')

        # We only compare the block because the absolute pot field I changed it
        np.testing.assert_array_almost_equal(sol[0][0, :], real_sol[0][0, :], decimal=3)

        # Checking that the plots do not rise errors
        gempy.plot_section(geo_data, sol[0][0, :], 25, direction='y', plot_data=True)
        gempy.plot_potential_field(geo_data, sol[0][1, :], 25)


class TestFaults:
    @pytest.fixture(scope='class')
    def theano_f_1f(self):
        # Importing the data from csv files and settign extent and resolution
        geo_data = gempy.create_data([0, 10, 0, 10, -10, 0], [50, 50, 50],
                                     path_f="./GeoModeller/test_d/test_d_Foliations.csv",
                                     path_i="./GeoModeller/test_d/test_d_Points.csv")

        gempy.set_series(geo_data, {'series': ('A', 'B'),
                                        'fault1': 'f1'}, order_series=['fault1', 'series'])

        # interp_data = gempy.set_interpolator(geo_data,
        #                                      dtype="float64",
        #                                      verbose=['solve_kriging',
        #                                               'faults block'
        #                                               ])

        # # Set all the theano shared parameters and return the symbolic variables (the input of the theano function)
        # input_data_T = interp_data.interpolator.tg.input_parameters_list()
        #
        # # Prepare the input data (interfaces, foliations data) to call the theano function.
        # # Also set a few theano shared variables with the len of formations series and so on
        # input_data_P = interp_data.interpolator.data_prep(u_grade=[3, 3])
        #
        # # Compile the theano function.
        # compiled_f = theano.function(input_data_T, interp_data.interpolator.tg.whole_block_model(1),
        #                              allow_input_downcast=True, profile=True)
        # interp_data.interpolator.get_kriging_parameters()

       # geo_data.n_faults = 1
       #  interp_data = gempy.InterpolatorInput(geo_data, dtype='float64')
       #  compiled_f = interp_data.compile_th_fn()
        interp_data = gempy.InterpolatorInput(geo_data, dtype='float64')
        return interp_data#, compiled_f

    def test_d(self, theano_f_1f):
        """
        Two layers 1 fault
        """
        # interp_data = theano_f_1f[0]
        # compiled_f = theano_f_1f[1]

        # Importing the data from csv files and settign extent and resolution
        geo_data = gempy.create_data([0, 10, 0, 10, -10, 0], [50, 50, 50],
                                     path_f="./GeoModeller/test_d/test_d_Foliations.csv",
                                     path_i="./GeoModeller/test_d/test_d_Points.csv")

        gempy.set_series(geo_data, {'series': ('A', 'B'),
                                          'fault1': 'f1'}, order_series=['fault1', 'series'],
                         verbose=0)

       #  rescaled_data = gempy.rescale_data(geo_data)
       #
       # # interp_data = gempy.set_interpolator(geo_data, dtype="float64",)
       #  interp_data.interpolator._data_scaled = rescaled_data
       #  interp_data.interpolator._grid_scaled = rescaled_data.grid
       #  interp_data.interpolator.order_table()
       #  interp_data.interpolator.set_theano_shared_parameteres()
       #
       #  # Prepare the input data (interfaces, foliations data) to call the theano function.
       #  # Also set a few theano shared variables with the len of formations series and so on
       #  input_data_P = interp_data.interpolator.data_prep(u_grade=[3, 3])
       #
       #  interp_data.interpolator.get_kriging_parameters()
       #
       #  sol = compiled_f(input_data_P[0], input_data_P[1], input_data_P[2], input_data_P[3], input_data_P[4],
       #                   input_data_P[5])

      #  geo_data.n_faults = 1

        interp_data = theano_f_1f

        # Create new interp data without compiling theano
        new_interp_data = gempy.InterpolatorInput(geo_data, dtype='float64', compile_theano=False)
        # Updating the interp data which has theano compiled
        interp_data.update_interpolator(new_interp_data.geo_data_res)

        # Compute model
        sol = gempy.compute_model(interp_data, u_grade=[3, 3])

        # Load model
        real_sol = np.load('test_d_sol.npy')

        # We only compare the block because the absolute pot field I changed it
        np.testing.assert_array_almost_equal(sol[0][0, :], real_sol[0][0, :], decimal=3)

        # # Checking that the plots do not rise errors
        # gempy.plot_section(geo_data, sol[0][0, :], 25, direction='y', plot_data=True)
        # gempy.plot_potential_field(geo_data, sol[0][1, :], 25)
        #

       #  interp_data.set_interpolator(geo_data)
       #
       #  i = interp_data.get_input_data(u_grade=[3, 3])
       #  sol = compiled_f(*i)
       #
       # # print(interp_data.rescale_factor, 'rescale')
       #  real_sol = np.load('test_d_sol.npy')
       #  np.testing.assert_array_almost_equal(sol[:, :2, :], real_sol, decimal=3)

    def test_e(self, theano_f_1f):
        """
        Two layers a bit curvy, 1 fault
        """
        # interp_data = theano_f_1f[0]
        # compiled_f = theano_f_1f[1]

        # Importing the data from csv files and settign extent and resolution
        geo_data = gempy.create_data([0, 10, 0, 10, -10, 0], [50, 50, 50],
                                     path_f="./GeoModeller/test_e/test_e_Foliations.csv",
                                     path_i="./GeoModeller/test_e/test_e_Points.csv")

        gempy.set_series(geo_data, {'series': ('A', 'B'),
                                        'fault1': 'f1'}, order_series=['fault1', 'series'],
                         verbose=0)

        # geo_data.n_faults = 1
        #
        # interp_data.set_interpolator(geo_data)
        #
        # i = interp_data.get_input_data(u_grade=[3, 3])
        # sol = compiled_f(*i)

        interp_data = theano_f_1f

        # Create new interp data without compiling theano
        new_interp_data = gempy.InterpolatorInput(geo_data, dtype='float64', compile_theano=False)
        # Updating the interp data which has theano compiled
        interp_data.update_interpolator(new_interp_data.geo_data_res)

        # Compute model
        sol = gempy.compute_model(interp_data, u_grade=[3, 3])

        # Load model
        real_sol = np.load('test_e_sol.npy')

        # We only compare the block because the absolute pot field I changed it
        np.testing.assert_array_almost_equal(sol[0][0, :], real_sol[0][0, :], decimal=3)


        # # print(interp_data.rescale_factor, 'rescale')
        # real_sol = np.load('test_e_sol.npy')
        # np.testing.assert_array_almost_equal(sol[:, :2, :], real_sol, decimal=3)

    def test_f(self, theano_f_1f):
        """
        Two layers a bit curvy, 1 fault
        """
        # interp_data = theano_f_1f[0]
        # compiled_f = theano_f_1f[1]

        # Importing the data from csv files and settign extent and resolution
        geo_data = gempy.create_data([0, 2000, 0, 2000, -2000, 0], [50, 50, 50],
                                     path_f="./GeoModeller/test_f/test_f_Foliations.csv",
                                     path_i="./GeoModeller/test_f/test_f_Points.csv")

       # geo_data.set_formation_number(geo_data.formations[[3, 2, 1, 0, 4]])

        gempy.set_series(geo_data, {'series': ('Reservoir'
                                                    , 'Seal',
                                                    'SecondaryReservoir',
                                                    'NonReservoirDeep'
                                               ),
                                         'fault1': 'MainFault'},
                         order_series=['fault1', 'series'],
                         verbose=0)

        interp_data = theano_f_1f

        # Create new interp data without compiling theano
        new_interp_data = gempy.InterpolatorInput(geo_data, dtype='float64', compile_theano=False)
        # Updating the interp data which has theano compiled
        interp_data.update_interpolator(new_interp_data.geo_data_res)

        # Compute model
        sol = gempy.compute_model(interp_data, u_grade=[3, 3])

        # Load model
        # real_sol = np.load('test_d_sol.npy')
        #
        # # We only compare the block because the absolute pot field I changed it
        # np.testing.assert_array_almost_equal(sol[0][0, :], real_sol[0][0, :], decimal=3)

        # geo_data.n_faults = 1
        #
        # interp_data.set_interpolator(geo_data)
        #
        # i = interp_data.get_input_data(u_grade=[3, 3])
        # sol = compiled_f(*i)

        real_sol = np.load('test_f_sol.npy')
       # np.testing.assert_array_almost_equal(sol[:, :2, :], real_sol, decimal=2)
        mismatch = ~np.isclose(sol[0][0, :], real_sol[0][0, :], rtol=0.01).sum()/sol[0][0].shape[0]
        assert mismatch * 100 < 1
        GeoMod_sol = geo_data._read_vox('./GeoModeller/test_f/test_f.vox')
        op1 = (GeoMod_sol - sol[0][0, :])
        op2 = (op1 != 0).sum()
        similarity = op2/sol[0][0].shape[0]
        #similarity = ((GeoMod_sol - sol[0][0, :]) != 0).sum() / sol[0][0].shape[0]

        print('The mismatch geomodeller-gempy is ', similarity*100, '%')
        assert similarity < 0.05, 'The mismatch with geomodeller is too high'
