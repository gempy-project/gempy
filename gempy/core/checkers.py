# TODO
# - Check the basement layer is not in in SurfacePoints and Orientations

import pandas as pn
import numpy as np

def check_for_nans(array_like):
    if ~(pn.notnull(np.atleast_1d(array_like))).any():
        array_like = None
    return array_like


def check_for_nans_input_theano():
    pass


def check_kriging():
    # TODO check range and covariance are not 0
    pass

def check_faults_are_set():
    pass

def check_fault_relations(self):
    # Method to check that only older df offset newer ones?
    #
    # try:
    #     # Check if there is already a categories_df
    #     self.df
    #
    #     try:
    #         if any(self.df.columns != series.columns):
    #             series_fault = self.count_faults()
    #             self.df = pn.DataFrame(index=series.columns, columns=['isFault'])
    #             self.df['isFault'] = self.df.index.isin(series_fault)
    #     except ValueError:
    #         series_fault = self.count_faults()
    #         self.df = pn.DataFrame(index=series.columns, columns=['isFault'])
    #         self.df['isFault'] = self.df.index.isin(series_fault)
    #
    #     if series_fault:
    #         self.df['isFault'] = self.df.index.isin(series_fault)
    #
    # except AttributeError:
    #
    #     if not series_fault:
    #         series_fault = self.count_faults()
    #         self.df = pn.DataFrame(index=series.columns, columns=['isFault'])
    #         self.df['isFault'] = self.df.index.isin(series_fault)

    # self.surface_points.loc[:, 'isFault'] = self.surface_points['series'].isin(self.df.index[self.df['isFault']])
    # self.orientations.loc[:, 'isFault'] = self.orientations['series'].isin(
    #     self.df.index[self.df['isFault']])
    pass