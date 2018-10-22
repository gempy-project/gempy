import os
import sys
from os import path

# This is for sphenix to find the packages
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
import pandas as pn
pn.options.mode.chained_assignment = None
from .data import *
from gempy.utils.meta import _setdoc


@_setdoc([MetaData.__doc__, GridClass.__doc__])
class Model(object):
    """
    Container class of all objects that constitute a GemPy model. In addition the class provides the methods that
    act in more than one of this class.
    """
    def __init__(self, project_name='default_project'):

        self.meta = MetaData(project_name=project_name)
        self.grid = GridClass()
        self.series = Series()
        self.formations = Formations()
        self.faults = Faults(self.series)
        self.interfaces = Interfaces()
        self.orientations = Orientations()

        self.rescaling = RescaledData(self.interfaces, self.orientations, self.grid)
        self.additional_data = AdditionalData(self.interfaces, self.orientations, self.grid, self.faults,
                                              self.formations, self.rescaling)
        self.interpolator = Interpolator(self.interfaces, self.orientations, self.grid, self.formations,
                                         self.faults, self.additional_data)
        self.solutions = Solution(self.additional_data, self.formations, self.grid)

    def __str__(self):
        return self.meta.project_name

    def new_model(self, name_project='default_project'):
        self.__init__(name_project)

    def save_model(self, path=False):
        """
        Short term model storage. Object to a python pickle (serialization of python). Be aware that if the dependencies
        versions used to export and import the pickle differ it may give problems

        Args:
            path (str): path where save the pickle

        Returns:
            True
        """
        sys.setrecursionlimit(10000)

        if not path:
            # TODO: Update default to meta name
            path = './'+self.meta.project_name
        import pickle
        with open(path+'.pickle', 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        return True

    @staticmethod
    def load_model(path):
        """
        Read InputData object from python pickle.

        Args:
           path (str): path where save the pickle

        Returns:
            :class:`gempy.core.model.Model`

        """
        import pickle
        with open(path, 'rb') as f:
            # The protocol version used is detected automatically, so we do not
            # have to specify it.
            model = pickle.load(f)
            return model

    def save_model_long_term(self):
        # TODO saving the main attributes in a seriealize way independent on the package i.e. interfaces and
        # TODO orientations df, grid values etc.
        pass

    def get_data(self, itype='data', numeric=False, verbosity=0):
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
        elif itype == 'data':
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
        elif itype is 'additional_data':
            raw_data = self.additional_data
        else:
            raise AttributeError('itype has to be \'data\', \'additional data\', \'interfaces\', \'orientations\','
                                 ' \'formations\',\'series\', \'faults\' or \'faults_relations\'')

        return raw_data

    # def get_theano_input(self):
    #     pass

    def set_grid(self, grid: GridClass, only_model=False):
        self.grid = grid
        if only_model is not True:
            self.additional_data.grid = grid
            self.interpolator.grid = grid
            self.rescaling.grid = grid
            self.solutions.grid = grid

    def set_series(self):
        pass

    def set_formations(self):
        pass

    def set_faults(self):
        pass

    def set_interfaces(self):
        pass

    def set_orientations(self):
        pass

    def set_interpolator(self, interpolator: Interpolator):
        self.interpolator = interpolator

    def set_theano_function(self, interpolator: Interpolator):
        self.interpolator.theano_graph = interpolator.theano_graph
        self.interpolator.theano_function = interpolator.theano_function
        self.interpolator.set_theano_shared_parameters()
