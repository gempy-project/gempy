import os
import sys
from os import path

# This is for sphenix to find the packages
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

import copy
import numpy as np
import pandas as pn

import warnings

try:
    import qgrid
except ImportError:
    warnings.warn('qgrid package is not installed. No interactive dataframes available.')

pn.options.mode.chained_assignment = None

from .data import *


class Model(object):
    def __init__(self, name_project='default_project'):

        self.meta = MetaData(name_project=name_project)
        self.grid = GridClass()
        self.series = Series()
        self.faults = Faults(self.series)
        self.formations = Formations()
        self.interfaces = Interfaces()
        self.orientations = Orientations()
        self.solutions = Solution()
        self.rescaling = RescaledData(self.interfaces, self.orientations, self.grid)
        self.additional_data = AdditionalData(self.interfaces, self.orientations, self.grid, self.faults,
                                              self.formations, self.rescaling)
        self.interpolator = Interpolator(self.interfaces, self.orientations, self.grid, self.formations,
                                         self.faults, self.additional_data)

    def new_model(self, name_project='default_project'):
        self.__init__(name_project)

    def save_model(self, path=False):
        """
        Short term model storage. Object to a python pickle (serialization of python). Be aware that if the dependencies
        versions used to export and import the pickle differ it may give problems

        Args:
            path (str): path where save the pickle

        Returns:
            None
        """

        if not path:
            # TODO: Update default to meta name
            path = './'+self.meta.project_name
        import pickle
        with open(path+'.pickle', 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def save_model_long_term(self):
        # TODO saving the main attributes in a seriealize way independent on the package i.e. interfaces and
        # TODO orientations df, grid values etc.
        pass

    def get_data(self, itype='all', numeric=False, verbosity=0):
        """
        Method that returns the interfaces and orientations pandas Dataframes. Can return both at the same time or only
        one of the two

        Args:
            itype: input_data data type, either 'orientations', 'interfaces' or 'all' for both.
            numeric(bool): Return only the numerical values of the dataframe. This is much lighter database for storing
                traces
            verbosity (int): Number of properties shown
        Returns:
            pandas.core.frame.DataFrame: Data frame with the raw data

        """
        # dtype = 'object'
        # TODO adapt this
        if verbosity == 0:
            show_par_f = self.orientations._columns_o_1
            show_par_i = self.interfaces._columns_i_1
        else:
            show_par_f = self.orientations.df.columns
            show_par_i = self.interfaces.df.columns

        if numeric:
            show_par_f = self.orientations._columns_o_num
            show_par_i = self.interfaces._columns_i_num
            dtype = 'float'

        if itype == 'orientations':
            raw_data = self.orientations.df[show_par_f]  # .astype(dtype)
            # Be sure that the columns are in order when used for operations
            if numeric:
                raw_data = raw_data[['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', 'dip', 'azimuth', 'polarity']]
        elif itype == 'interfaces':
            raw_data = self.interfaces.df[show_par_i]  # .astype(dtype)
            # Be sure that the columns are in order when used for operations
            if numeric:
                raw_data = raw_data[['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', 'dip', 'azimuth', 'polarity']]
        elif itype == 'all':
            raw_data = pn.concat([self.interfaces.df[show_par_i],  # .astype(dtype),
                                  self.orientations.df[show_par_f]],  # .astype(dtype)],
                                 keys=['interfaces', 'orientations'],
                                 sort=False)
            # Be sure that the columns are in order when used for operations
            if numeric:
                raw_data = raw_data[['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', 'dip', 'azimuth', 'polarity']]

        elif itype is 'formations':
            raw_data = self.formations
        elif itype is 'series':
            raw_data = self.series
        elif itype is 'faults':
            raw_data = self.faults
        elif itype is 'faults_relations':
            raw_data = self.faults.faults_relations
        else:
            raise AttributeError('itype has to be \'all\', \'interfaces\', \'orientations\', \'formations\', \
                                          \'serires\', \'faults\' or \'faults_relations\'')

        # else:
        #    raise AttributeError('itype has to be: \'orientations\', \'interfaces\', or \'all\'')

        return raw_data

    def get_theano_input(self):
        pass

    def update_df(self, series_distribution=None, order=None):
       pass
       #  self.interfaces['formation'] = self.interfaces['formation'].astype('category')
       #  self.orientations['formation'] = self.orientations['formation'].astype('category')
       #
       #  self.set_series(series_distribution=series_distribution, order=order)
       #  self.set_basement()
       #  faults_series = self.count_faults()
       #  self.set_faults(faults_series)
       #
       # # self.reset_indices()
       #
       #  self.set_formations()
       #  self.order_table()
       #  self.set_fault_relation()