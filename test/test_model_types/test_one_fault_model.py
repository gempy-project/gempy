import gempy as gp
import pandas as pn
import numpy as np
import os
import pytest
import matplotlib.pyplot as plt
input_path = os.path.dirname(__file__)+'/../../notebooks/data'


class TestFabianModel:

    # @pytest.fixture(scope='class')
    # def one_fault_model(self):
    #     model = gp.create_data([0, 2000, 0, 2000, 0, 2000], [50, 50, 50],
    #                            path_o=input_path + '/input_data/tut_chapter1/simple_fault_model_orientations.csv',
    #                            path_i=input_path + '/input_data/tut_chapter1/simple_fault_model_points.csv')
    #
    #     # Assigning series to surface as well as their order (timewise)
    #     gp.set_series(model, {"Fault_Series": 'Main_Fault',
    #                           "Strat_Series": ('Sandstone_2', 'Siltstone',
    #                                            'Shale', 'Sandstone_1')},
    #                 )
    #     return model
    #
    # @pytest.fixture(scope='session')
    # def fabian_model(self, interpolator, one_fault_model):
    #     one_fault_model.set_theano_function(interpolator)
    #     sol = gp.compute_model(one_fault_model)
    #     print(interpolator)
    #     print(sol)
    #     return sol
    #
    def test_init_model(self, one_fault_model):
        print(one_fault_model)

    def test_get_data(self, one_fault_model):
        print(gp.get_data(one_fault_model))
        print(gp.get_data(one_fault_model, itype='additional_data'))

    def test_plotting_data(self, one_fault_model):
        gp.plot.plot_data(one_fault_model)

    def test_compute_model(self, one_fault_model_solution, ):
        sol = one_fault_model_solution.solutions
        print(sol)
        return sol

    def test_plot_section(self, one_fault_model):

        gp.plot.plot_section(one_fault_model, cell_number=25,
                             direction='y', show_data=True)

       # plt.savefig(os.pardir+'/../figs/example1.png')

