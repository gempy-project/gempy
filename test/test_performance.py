import pytest

import theano
import numpy as np
import sys, os
sys.path.append("../..")
import gempy
import matplotlib.pyplot as plt
import pdb


class TestPerfomance:
    """
    I am testing all block and potential field values so sol is (n_block+n_pot)
    """

    # Importing the data from csv files and settign extent and resolution
    geo_data = gempy.create_data([0, 10, 0, 10, -10, 0], [50, 50, 50],
                                 path_o=os.path.dirname(__file__) + "/GeoModeller/test_a/test_a_Foliations.csv",
                                 path_i=os.path.dirname(__file__) + "/GeoModeller/test_a/test_a_Points.csv")

    interp_data = gempy.InterpolatorData(geo_data, dtype='float64', compile_theano=False)

    n_faults = 0
    interp_data.interpolator.tg.fault_matrix = theano.tensor.zeros((n_faults * 2,
                                                                    interp_data.interpolator.tg.grid_val_T.shape[0] +
                                                                    2 * interp_data.interpolator.tg.len_points))

    th_f = theano.function(interp_data.interpolator.tg.input_parameters_list(),
                           interp_data.interpolator.tg.compute_a_series(
                               *interp_data.interpolator.tg.len_series_i[n_faults:],
                               *interp_data.interpolator.tg.len_series_f[n_faults:],
                               *interp_data.interpolator.tg.n_formations_per_serie[n_faults:],
                               *interp_data.interpolator.tg.n_universal_eq_T[n_faults:],
                               *interp_data.interpolator.tg.lith_block_init,
                               *interp_data.interpolator.tg.fault_matrix
                           ),
                           on_unused_input='ignore')
