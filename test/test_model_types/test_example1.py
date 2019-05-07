import gempy as gp
import pandas as pn
import numpy as np
import os
import pytest
import matplotlib.pyplot as plt
input_path = os.path.dirname(__file__)+'/../../notebooks/data'


class TestFabianModel:

    @pytest.fixture(scope='class')
    def geo_model(self):
        model = gp.create_data([0, 2000, 0, 2000, 0, 2000], [50, 50, 50],
                               path_o=input_path + '/input_data/tut_chapter1/simple_fault_model_orientations.csv',
                               path_i=input_path + '/input_data/tut_chapter1/simple_fault_model_points.csv')

        # Assigning series to surface as well as their order (timewise)
        gp.set_series(model, {"Fault_Series": 'Main_Fault',
                              "Strat_Series": ('Sandstone_2', 'Siltstone',
                                               'Shale', 'Sandstone_1')},
                    )
        return model

    def test_init_model(self, geo_model):
        print(geo_model)

    def test_get_data(self, geo_model):
        print(gp.get_data(geo_model))
        print(gp.get_data(geo_model, itype='additional_data'))

    def test_plotting_data(self, geo_model):
        gp.plot.plot_data(geo_model)

    @pytest.fixture(scope='class')
    def compute_model(self, interpolator_islith_isfault, geo_model):
        geo_model.set_theano_function(interpolator_islith_isfault)
        sol = gp.compute_model(geo_model)
        print(interpolator_islith_isfault)
        print(sol)
        return sol

    def test_compute_model(self, interpolator_islith_isfault, geo_model):
        geo_model.set_theano_function(interpolator_islith_isfault)
        sol = gp.compute_model(geo_model)

        print(interpolator_islith_isfault)
        print(sol)
        return sol

    def test_plot_section(self, geo_model, compute_model):

        gp.plot.plot_section(geo_model, cell_number=25,
                             direction='y', show_data=True)

       # plt.savefig(os.pardir+'/../figs/example1.png')

