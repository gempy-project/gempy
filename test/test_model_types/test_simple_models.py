import pytest

import theano
import numpy as np
import sys, os
sys.path.append("../..")
import gempy
import matplotlib.pyplot as plt
import pdb


input_path = os.path.dirname(__file__)+'/../input_data'


class TestNoFaults:
    """
    I am testing all block and potential field values so sol is (n_block+n_pot)
    """

    def test_a(self, interpolator_islith_nofault):
        """
        2 Horizontal layers with drift 0
        """
        # Importing the data from csv files and settign extent and resolution
        geo_data = gempy.create_data([0, 10, 0, 10, -10, 0], [50, 50, 50],
                                     path_o=input_path+"/GeoModeller/test_a/test_a_Foliations.csv",
                                     path_i=input_path+"/GeoModeller/test_a/test_a_Points.csv")

        geo_data.set_theano_function(interpolator_islith_nofault)

        # Compute model
        sol = gempy.compute_model(geo_data)

        if False:
            np.save(input_path+'/test_a_sol.npy', sol)

        # Load model
        real_sol = np.load(input_path + '/test_a_sol.npy')

        # Checking that the plots do not rise errors
        gempy.plot.plot_section(geo_data, 25, direction='y', plot_data=True)
        plt.savefig(os.path.dirname(__file__)+'/../figs/test_a.png', dpi=100)

        gempy.plot.plot_scalar_field(geo_data, 25)

        # We only compare the block because the absolute pot field I changed it
        np.testing.assert_array_almost_equal(np.round(sol.lith_block), real_sol[0][0, :], decimal=0)

    def test_b(self, interpolator_islith_nofault):
        """
        Two layers a bit curvy, drift degree 1
        """

        # Importing the data from csv files and settign extent and resolution
        geo_data = gempy.create_data([0, 10, 0, 10, -10, 0], [50, 50, 50],
                                     path_o=input_path+"/GeoModeller/test_b/test_b_Foliations.csv",
                                     path_i=input_path+"/GeoModeller/test_b/test_b_Points.csv")

        geo_data.set_theano_function(interpolator_islith_nofault)

        print(gempy.get_kriging_parameters(geo_data))

        # Compute model
        sol = gempy.compute_model(geo_data)

        gempy.plot.plot_section(geo_data, 25, direction='y', plot_data=True)
        plt.savefig(os.path.dirname(__file__)+'/../figs/test_b.png', dpi=200)

        if False:
            np.save(input_path + '/test_b_sol.npy', sol)

        # Load model
        real_sol = np.load(input_path + '/test_b_sol.npy')

        # Checking that the plots do not rise errors
        gempy.plot.plot_scalar_field(geo_data, 25)

        # We only compare the block because the absolute pot field I changed it
        np.testing.assert_array_almost_equal(np.round(sol.lith_block), real_sol[0][0, :], decimal=0)

    def test_c(self, interpolator_islith_nofault):
        """
        Two layers a bit curvy, drift degree 0
        """

        # Importing the data from csv files and settign extent and resolution
        geo_data = gempy.create_data([0, 10, 0, 10, -10, 0], [50, 50, 50],
                                     path_o=input_path+"/GeoModeller/test_c/test_c_Foliations.csv",
                                     path_i=input_path+"/GeoModeller/test_c/test_c_Points.csv")

        geo_data.set_theano_function(interpolator_islith_nofault)


        # Compute model
        sol = gempy.compute_model(geo_data)

        gempy.plot.plot_section(geo_data, 25, direction='y', plot_data=True)
        plt.savefig(os.path.dirname(__file__)+'/../figs/test_c.png', dpi=200)

        if False:
            np.save(input_path + '/test_c_sol.npy', sol)

        # Load model
        real_sol = np.load(input_path + '/test_c_sol.npy')

        # Checking that the plots do not rise errors
        gempy.plot.plot_section(geo_data, 25, direction='y', plot_data=True)
        gempy.plot.plot_scalar_field(geo_data, 25)

        # We only compare the block because the absolute pot field I changed it
        np.testing.assert_array_almost_equal(np.round(sol.lith_block), real_sol[0][0, :], decimal=0)


class TestFaults:

    def test_d(self, interpolator_islith_isfault):
        """
        Two layers 1 fault
        """

        # Importing the data from csv files and settign extent and resolution
        geo_data = gempy.create_data([0, 10, 0, 10, -10, 0], [50, 50, 50],
                                     path_o=input_path+"/GeoModeller/test_d/test_d_Foliations.csv",
                                     path_i=input_path+"/GeoModeller/test_d/test_d_Points.csv")

        gempy.set_series(geo_data, {'series': ('A', 'B'),
                                    'fault1': 'f1'}, order_series=['fault1', 'series'],
                                                     order_formations=['f1', 'A', 'B'],
                         verbose=0)

        geo_data.set_theano_function(interpolator_islith_isfault)

        # Compute model
        sol = gempy.compute_model(geo_data)

        gempy.plot.plot_section(geo_data, 25, direction='y', plot_data=True)
        plt.savefig(os.path.dirname(__file__)+'/../figs/test_d.png', dpi=200)

        if False:
            np.save(input_path + '/test_d_sol.npy', sol)

        # Load model
        real_sol = np.load(input_path + '/test_d_sol.npy')

        # We only compare the block because the absolute pot field I changed it
        np.testing.assert_array_almost_equal(np.round(sol.lith_block), real_sol[0][0, :], decimal=0)

    def test_e(self, interpolator_islith_isfault):
        """
        Two layers a bit curvy, 1 fault
        """
        # Importing the data from csv files and settign extent and resolution
        geo_data = gempy.create_data([0, 10, 0, 10, -10, 0], [50, 50, 50],
                                     path_o=input_path+"/GeoModeller/test_e/test_e_Foliations.csv",
                                     path_i=input_path+"/GeoModeller/test_e/test_e_Points.csv")

        gempy.set_series(geo_data, {'series': ('A', 'B'),
                                        'fault1': 'f1'}, order_series=['fault1', 'series'],
                                                         order_formations=['f1','A','B'],
                         verbose=0)

        geo_data.set_theano_function(interpolator_islith_isfault)

        # Compute model
        sol = gempy.compute_model(geo_data)

        if False:
            np.save(input_path + '/test_e_sol.npy', sol)

        gempy.plot.plot_section(geo_data, 25, direction='y', plot_data=True)
        plt.savefig(os.path.dirname(__file__)+'/../figs/test_e.png', dpi=200)

        # Load model
        real_sol = np.load(input_path + '/test_e_sol.npy')

        # We only compare the block because the absolute pot field I changed it
        np.testing.assert_array_almost_equal(np.round(sol.lith_block), real_sol[0][0, :], decimal=0)

    def test_f(self, interpolator_islith_isfault):
        """
        Two layers a bit curvy, 1 fault. Checked with geomodeller
        """

        # Importing the data from csv files and settign extent and resolution
        geo_data = gempy.create_data([0, 2000, 0, 2000, -2000, 0], [50, 50, 50],
                                     path_o=input_path+"/GeoModeller/test_f/test_f_Foliations.csv",
                                     path_i=input_path+"/GeoModeller/test_f/test_f_Points.csv")

        gempy.set_series(geo_data, {'series': ('Reservoir',
                                               'Seal',
                                               'SecondaryReservoir',
                                               'NonReservoirDeep'
                                               ),
                                    'fault1': 'MainFault'},
                         order_series=['fault1', 'series'],
                         order_formations=['MainFault', 'SecondaryReservoir', 'Seal', 'Reservoir', 'NonReservoirDeep'],
                         verbose=0)

        geo_data.set_theano_function(interpolator_islith_isfault)

        # Compute model
        sol = gempy.compute_model(geo_data)

        if False:
            np.save(input_path + '/test_f_sol.npy', sol)

        real_sol = np.load(input_path + '/test_f_sol.npy')

        gempy.plot.plot_section(geo_data, 25, direction='y', plot_data=True)

        plt.savefig(os.path.dirname(__file__)+'/../figs/test_f.png', dpi=200)

        # We only compare the block because the absolute pot field I changed it
        np.testing.assert_array_almost_equal(np.round(sol.lith_block), real_sol[0][0, :], decimal=0)

        ver, sim = gempy.get_surfaces(geo_data)
        print(ver, sim)
        # gempy.plot.plot_surfaces_3D(geo_model, ver, sim)
