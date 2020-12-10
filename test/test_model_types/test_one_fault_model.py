import gempy as gp
import pandas as pn
import numpy as np
import os
import pytest
import matplotlib.pyplot as plt
input_path = os.path.dirname(__file__)+'/../../notebooks/data'


class TestFabianModel:

    def test_init_model(self, one_fault_model):
        print(one_fault_model)

    def test_get_data(self, one_fault_model):
        print(gp.get_data(one_fault_model))
        print(gp.get_data(one_fault_model, itype='additional_data'))

    def test_plotting_data(self, one_fault_model):
        gp.plot.plot_2d(one_fault_model, show_data=True, show_results=False)

    def test_compute_model(self, one_fault_model_solution):
        sol = one_fault_model_solution.solutions
        print(sol)
        return sol

    def test_plot_section(self, one_fault_model):

        gp.plot.plot_2d(one_fault_model, cell_number=25,
                        direction='y', show_data=True)

       # plt.savefig(os.pardir+'/../figs/example1.png')

