import pytest

import theano
import numpy as np
import sys, os
sys.path.append("../..")
import gempy
import matplotlib.pyplot as plt
import pdb


class TestPerfomance:

    def test_high_res(self):
        pass
        # TODO Update!!!
        #
        # # Importing the data from csv files and settign extent and resolution
        # geo_model = gempy.read_pickle(os.path.dirname(__file__)+"/input_data/geo_model.pickle")
        # geo_model.surface_points['formation'] = geo_model.surface_points['formation'].astype('category')
        # geo_model.orientations['formation'] = geo_model.orientations['formation'].astype('category')
        # geo_model.add_orientation(X=-2.88043478e+04, Y=6.21413043e+06, Z=-1.17648965e+02, dip=0, azimuth=0, polarity=1,
        #                          formation='basement')
        # geo_model.faults_relations_df = None
        # new_grid = gempy.GridClass()
        # res = 100
        #
        # # Create a new grid object with the new resolution
        # new_grid.set_regular_grid(geo_model.extent, [res, res, res])
        #
        # # Setting the new grid to the geodata
        # gempy.set_grid(geo_model, new_grid)
        #
        # n_faults = 0
        #
        # interp_data = gempy.InterpolatorData(geo_model, dtype='float32', compile_theano=False, verbose=[],
        #                                      theano_optimizer='fast_run')
        #
        # interp_data.interpolator.tg.fault_matrix = theano.tensor.zeros((n_faults * 2,
        #                                                                 interp_data.interpolator.tg.grid_val_T.shape[0] +
        #                                                                 2 * interp_data.interpolator.tg.len_points))
        # theano.config.optimizer = 'fast_run'
        # th_f = theano.function(interp_data.interpolator.tg.input_parameters_list(),
        #                        interp_data.interpolator.tg.compute_a_series(
        #                            *interp_data.interpolator.tg.len_series_i.get_value()[n_faults:].astype(int),
        #                            *interp_data.interpolator.tg.len_series_f.get_value()[n_faults:].astype(int),
        #                            *interp_data.interpolator.tg.n_formations_per_series.get_value()[n_faults:],
        #                            *interp_data.interpolator.tg.n_universal_eq_T.get_value()[n_faults:],
        #                            interp_data.interpolator.tg.lith_block_init,
        #                            interp_data.interpolator.tg.final_scalar_field_at_formations_op,
        #                           # interp_data.interpolator.tg.fault_matrix
        #                        ),
        #                        on_unused_input='ignore',
        #                        allow_input_downcast=True,
        #                        profile=False)
        #
        # sol = th_f(*interp_data.get_input_data())
        #
        # assert sol
