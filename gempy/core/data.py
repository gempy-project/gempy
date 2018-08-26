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


class Model(object):
    def __init__(self):
        self.meta = None
        self.grid = None
        self.faults = None
        self.series = None
        self.formations = None
        self.interfaces = None
        self.orientations = None
        self.structure = None
        self.model = None

    def save_data(self):
        pass

    def data_to_pickle(self, path=False):
        """
        Save InputData object to a python pickle (serialization of python). Be aware that if the dependencies
        versions used to export and import the pickle differ it may give problems

        Args:
            path (str): path where save the pickle

        Returns:
            None
        """

        if not path:
            # TODO: Update default to meta name
            path = './geo_data'
        import pickle
        with open(path+'.pickle', 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

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
            show_par_f = self._columns_o_1
            show_par_i = self._columns_i_1
        else:
            show_par_f = self.orientations.columns
            show_par_i = self.interfaces.columns

        if numeric:
            show_par_f = self._columns_o_num
            show_par_i = self._columns_i_num
            dtype = 'float'

        if itype == 'orientations':
            raw_data = self.orientations[show_par_f]  # .astype(dtype)
            # Be sure that the columns are in order when used for operations
            if numeric:
                raw_data = raw_data[['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', 'dip', 'azimuth', 'polarity']]
        elif itype == 'interfaces':
            raw_data = self.interfaces[show_par_i]  # .astype(dtype)
            # Be sure that the columns are in order when used for operations
            if numeric:
                raw_data = raw_data[['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', 'dip', 'azimuth', 'polarity']]
        elif itype == 'all':
            raw_data = pn.concat([self.interfaces[show_par_i],  # .astype(dtype),
                                  self.orientations[show_par_f]],  # .astype(dtype)],
                                 keys=['interfaces', 'orientations'])
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
            raw_data = self.faults_relations
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




class MetaData(object):
    def __init__(self, name_project='default_project'):
        self.name_project = name_project
        self.date = None


class GridClass(object):
    """
    Class to generate grids to pass later on to a InputData class.
    """

    def __init__(self):

        self.grid_type = 'foo'
        self.resolution = None
        self.extent = None
        self.values = None
        self.values_r = None

    def __str__(self):
        return self.grid_type

    def __repr__(self):
        return np.array_repr(self.values)

    def set_custom_grid(self, custom_grid):
        """
        Give the coordinates of an external generated grid

        Args:
            custom_grid (numpy.ndarray like): XYZ (in columns) of the desired coordinates

        Returns:
              numpy.ndarray: Unraveled 3D numpy array where every row correspond to the xyz coordinates of a regular grid
        """
        assert type(custom_grid) is np.ndarray and custom_grid.shape[1] is 3, 'The shape of new grid must be (n,3)' \
                                                                              ' where n is the number of points of ' \
                                                                              'the grid'
        self.values = custom_grid
        return self.values

    @staticmethod
    def create_regular_grid_3d(extent, resolution):
        """
        Method to create a 3D regular grid where is interpolated

        Args:
            extent (list):  [x_min, x_max, y_min, y_max, z_min, z_max]
            resolution (list): [nx, ny, nz].

        Returns:
            numpy.ndarray: Unraveled 3D numpy array where every row correspond to the xyz coordinates of a regular grid
        """

        dx, dy, dz = (extent[1] - extent[0]) / resolution[0], (extent[3] - extent[2]) / resolution[0],\
                                    (extent[5] - extent[4]) / resolution[0]

        g = np.meshgrid(
            np.linspace(extent[0] + dx / 2, extent[1] - dx / 2, resolution[0], dtype="float32"),
            np.linspace(extent[2] + dy / 2, extent[3] - dy / 2, resolution[1], dtype="float32"),
            np.linspace(extent[4] + dz / 2, extent[5] - dz / 2, resolution[2], dtype="float32"), indexing="ij"
        )

        values = np.vstack(map(np.ravel, g)).T.astype("float32")
        return values

    def set_regular_grid(self, extent, resolution):
        self.extent = extent
        self.resolution = resolution
        self.values = self.create_regular_grid_3d(extent, resolution)


class Series(object):
    def __init__(self):
        self.df = pn.DataFrame({"Default series": [None]}, dtype=str)
        self.sequential_pile = None

    def __repr__(self):
        return self.df.to_string()

    def _repr_html_(self):
        return self.df.to_html()

    def set_series(self, series_distribution, order=None):
        """
        Method to define the different series of the project.

        Args:
            series_distribution (dict): with the name of the serie as key and the name of the formations as values.
            order(Optional[list]): order of the series by default takes the dictionary keys which until python 3.6 are
                random. This is important to set the erosion relations between the different series

        Returns:
            self.series: A pandas DataFrame with the series and formations relations
            self.interfaces: one extra column with the given series
            self.orientations: one extra column with the given series
        """

        if isinstance(series_distribution, Interfaces):
            self.df = pn.DataFrame({"Default series": self.interfaces["formation"].unique().astype(list)},
                                   dtype=str)

        elif type(series_distribution) is dict:
            if order is None:
                order = series_distribution.keys()
            else:
                assert all(
                    np.in1d(order, list(series_distribution.keys()))), 'Order series must contain the same keys as' \
                                                                       'the passed dictionary ' + str(
                    series_distribution.keys())
            self.df = pn.DataFrame(dict([(k, pn.Series(v)) for k, v in series_distribution.items()]),
                                   columns=order)

        elif type(series_distribution) is pn.core.frame.DataFrame:
                self.df = series_distribution

        else:
            raise AttributeError('You must pass either a Interface object (or container with it) or'
                                 ' series_distribution (dict or DataFrame),'
                                 ' see Docstring for more information')

        if 'basement' not in self.df.iloc[:, -1].values:
            self.df.loc[self.df.shape[0], self.df.columns[-1]] = 'basement'

        # # Default series df. We extract the formations from Interfaces
        # if series_distribution is None and interfaces is not None:
        #     if self.df is None:
        #         self.df = pn.DataFrame({"Default series": self.interfaces["formation"].unique().astype(list)},
        #                                    dtype=str)
        # # We pass a df or dictionary with the right shape
        # else:
        #     if type(series_distribution) is dict:
        #         if order is None:
        #             order = series_distribution.keys()
        #         else:
        #             assert all(np.in1d(order, list(series_distribution.keys()))), 'Order series must contain the same keys as' \
        #                                                                'the passed dictionary ' + str(series_distribution.keys())
        #         self.df = pn.DataFrame(dict([(k, pn.Series(v)) for k, v in series_distribution.items()]),
        #                                columns=order)
        #         print (self.df)
        #     elif type(series_distribution) is pn.core.frame.DataFrame:
        #         self.df = series_distribution

      # TODO decide if we allow to update Interfaces and Orientations here
       #  # Now we fill the column series in the interfaces and orientations tables with the correspondant series and
       #  # assigned number to the series
       #  self.interfaces["series"] = [(i == self.series).sum().idxmax() for i in self.interfaces["formation"]]
       #  self.interfaces["series"] = self.interfaces["series"].astype('category')
       #  self.interfaces["order_series"] = [(i == self.series).sum().as_matrix().argmax().astype(int) + 1
       #                                     for i in self.interfaces["formation"]]
       #  self.orientations["series"] = [(i == self.series).sum().idxmax() for i in self.orientations["formation"]]
       #  self.orientations["series"] = self.orientations["series"].astype('category')
       #  self.orientations["order_series"] = [(i == self.series).sum().as_matrix().argmax().astype(int) + 1
       #                                     for i in self.orientations["formation"]]
       #
       #  # We sort the series altough is only important for the computation (we will do it again just before computing)
       # # if series_distribution is not None:
       #  self.interfaces.sort_values(by='order_series', inplace=True)
       #  self.orientations.sort_values(by='order_series', inplace=True)
       #
       #  self.interfaces['series'].cat.set_categories(self.series.columns, inplace=True)
       #  self.orientations['series'].cat.set_categories(self.series.columns, inplace=True)

        return self.df

    def map_formations(self, formations_df):

        # TODO review
        # Addind the formations of the new series to the formations df
        new_formations = self.series.values.reshape(1, -1)
        # Dropping nans
        new_formations = new_formations[~pn.isna(new_formations)]
        self.set_formations(formation_order=new_formations)

    
class Faults(object):
    def __init__(self):
        self.faults_relations = pn.DataFrame(index=['DefaultSeries'], columns=['DefaultSeries'], dtype='bool')
        self.faults = pn.DataFrame(columns=['isFault', 'isFinite'])
        self.n_faults = None

    def __repr__(self):
        return self.faults.to_string()

    def _repr_html_(self):
        return self.faults.to_html()

    def set_faults(self, series, series_fault=None):
        """
        Set a flag to the series that are faults.

        Args:
            series_fault(list or Interfaces): Name of the series which are faults
        """

        #if isinstance(series_fault, Interfaces):
        if series_fault is None:
            series_fault = self.count_faults(series.df.columns)

        self.faults = pn.DataFrame(index=series.df.columns, columns=['isFault'])
        self.faults['isFault'] = self.faults.index.isin(series_fault)
        self.n_faults = self.faults['isFault'].sum()
        return self.faults

        #
        # try:
        #     # Check if there is already a df
        #     self.faults
        #
        #     try:
        #         if any(self.faults.columns != series.columns):
        #             series_fault = self.count_faults()
        #             self.faults = pn.DataFrame(index=series.columns, columns=['isFault'])
        #             self.faults['isFault'] = self.faults.index.isin(series_fault)
        #     except ValueError:
        #         series_fault = self.count_faults()
        #         self.faults = pn.DataFrame(index=series.columns, columns=['isFault'])
        #         self.faults['isFault'] = self.faults.index.isin(series_fault)
        #
        #     if series_fault:
        #         self.faults['isFault'] = self.faults.index.isin(series_fault)
        #
        # except AttributeError:
        #
        #     if not series_fault:
        #         series_fault = self.count_faults()
        #         self.faults = pn.DataFrame(index=series.columns, columns=['isFault'])
        #         self.faults['isFault'] = self.faults.index.isin(series_fault)

        # self.interfaces.loc[:, 'isFault'] = self.interfaces['series'].isin(self.faults.index[self.faults['isFault']])
        # self.orientations.loc[:, 'isFault'] = self.orientations['series'].isin(
        #     self.faults.index[self.faults['isFault']])


    def check_fault_relations(self):
        pass

    def set_fault_relation(self, series, rel_matrix=None):
        """
        Method to set the faults that offset a given sequence and therefore also another fault

        Args:
            rel_matrix (numpy.array): 2D Boolean array with the logic. Rows affect (offset) columns
        """
        # TODO: Change the fault relation automatically every time we add a fault
        if not rel_matrix:
            rel_matrix = np.zeros((series.df.columns.shape[0],
                                   series.df.columns.shape[0]))

        self.faults_relations = pn.DataFrame(rel_matrix, index=series.df.columns,
                                             columns=series.df.columns, dtype='bool')

        return self.faults_relations
        #
        # try:
        #     self.faults_relations
        #
        # except AttributeError:
        #
        #     if rel_matrix is not None:
        #         self.faults_relations = pn.DataFrame(rel_matrix, index=series.df.columns,
        #                                              columns=series.df.columns, dtype='bool')

    @staticmethod
    def count_faults(list_of_names):
        """
        Read the string names of the formations to detect automatically the number of faults.
        """
        faults_series = []
        for i in list_of_names:
            try:
                if ('fault' in i or 'Fault' in i) and 'Default' not in i:
                    faults_series.append(i)
            except TypeError:
                pass
        return faults_series


class Formations(object):
    def __init__(self):
        self.df = pn.DataFrame(columns=['formation', 'id', 'isBasement'])
        self.df['isBasement'] = self.df['isBasement'].astype(bool)
        self.df["formation"] = self.df["formation"].astype('category')

        self.formations_names = np.empty(0)
        self._formation_values_set = False

    def __repr__(self):
        return self.df.to_string()

    def _repr_html_(self):
        return self.df.to_html()

    def set_formation_names(self, list_names):
        if type(list_names) is list or type(list_names) is np.ndarray:
            self.formations_names = np.asarray(list_names)
        elif isinstance(list_names, Interfaces):
            self.formations_names = np.asarray(list_names.df['formation'].cat.categories)
        else:
            raise AttributeError('list_names must be either array_like type or Interfaces')

        self.map_formation_names()

    def map_formation_names(self):
        if self.df['formation'].shape[0] == self.formations_names.shape[0]:
            self.df['formation'] = self.formations_names

        elif self.df['formation'].shape[0] > self.formations_names.shape[0]:
            n_to_append = self.df['formation'].shape[0] - self.formations_names.shape[0]
            for i in range(n_to_append):
                self.formations_names = np.append(self.formations_names, 'default_formation_' + str(i))
            warnings.warn('Length of formation_names does not match number of formations')
            self.df['formation'] = self.formations_names

        elif self.df['formation'].shape[0] < self.formations_names.shape[0]:
            warnings.warn('Length of formation_names does not match number of formations')
            self.df['formation'] = self.formations_names[:self.df.shape[0]]

        self.df['formation'] = self.df['formation'].astype('category')
        # try:
        #     self.df['formation_name'] = self.formations_names
        # except ValueError:
        #     warnings.warn('Length of formation_names does not match number of formations')
        #     self.df['formation_name'] = self.formations_names[:self.df.shape[0]]

    def set_basement(self, basement_formation):
        self.df['isBasement'] = self.df['formation'] == basement_formation

    def sort_formations(self, series):
        """
        Sort formations df regarding series order
        Args:
            series:

        Returns:

        """
        pass

    @staticmethod
    def set_id(df):
        df['id'] = df.index + 1
        return df

    def set_formations(self, values_array, values_names=np.empty(0), formation_names=None):
        self.df = pn.DataFrame(columns=['formation', 'isBasement', 'id'])
        self.df['isBasement'] = self.df['isBasement'].astype(bool)
        self.df["formation"] = self.df["formation"].astype('category')

        if type(values_array) is np.ndarray:
            if values_names.size is 0:
                for i in range(values_array.shape[1]):
                    values_names = np.append(values_names, 'value_' + str(i))
            vals_df = pn.DataFrame(values_array, columns=values_names)

        elif isinstance(values_array, pn.core.frame.DataFrame):
            vals_df = values_array

        else:
            raise AttributeError('values_array must be either numpy array or pandas df')

        if formation_names:
            self.set_formation_names(formation_names)
        #
        # if formation_names is None and self.formations_names.shape[0] <= values_array.shape[0]:
        #     n_to_append = values_array.shape[0] - self.formations_names.shape[0]
        #     for i in range(n_to_append):
        #         self.formations_names = np.append(self.formations_names, 'default_formation_' + str(i))
        #
        # vals_df['formation_name'] = self.formations_names

        f_df = pn.concat([self.df, vals_df], sort=False, axis=1, verify_integrity=True, ignore_index=False)

        #f_df = pn.merge(self.df, vals_df, on='')
        self.df = self.set_id(f_df)
        self.map_formation_names()
        self.set_basement(None)
        return self.df

    #
    # def _set_formations_values(self, formation_values=None, interfaces_df=None, formation_order=None):
    #
    #     #self.orientations['formation'] = self.orientations['formation'].astype('category')
    #
    #     if formation_order is None and interfaces_df is not None:
    #         if self.formations is None:
    #             interfaces_df['formation'] = interfaces_df['formation'].astype('category')
    #             # if self._formation_values_set is False:
    #             formation_order = interfaces_df['formation'].cat.categories
    #         else:
    #             # We check if in the df we are setting there is a new formation. if yes we append it to to the cat
    #             new_cat = interfaces_df['formation'].cat.categories[
    #                 ~np.in1d(interfaces_df['formation'].cat.categories,
    #                          self.formations.index)]
    #             if new_cat.empty:
    #                 formation_order = self.formations.index
    #             else:
    #                 formation_order = np.insert(self.formations.index.get_values(), 0, new_cat)
    #         # try:
    #         #     # Check if there is already a df
    #         #     formation_order = self.formations.index
    #         #
    #         # except AttributeError:
    #         #
    #
    #     if 'basement' not in formation_order:
    #         formation_order = np.append(formation_order, 'basement')
    #
    #     if formation_values is None:
    #         if self._formation_values_set:
    #             # Check if there is already a df
    #             formation_values = self.formations['value'].squeeze()
    #         else:
    #             formation_values = np.arange(1, formation_order.shape[0] + 1)
    #     else:
    #         self._formation_values_set = True
    #
    #     if np.atleast_1d(formation_values).shape[0] < np.atleast_1d(formation_order).shape[0]:
    #         formation_values = np.append(formation_values, formation_values.max() + 1)
    #
    #     self.formations = pn.DataFrame(index=formation_order,
    #                                    columns=['value', 'formation_number'])
    #
    #     self.formations['value'] = formation_values
    #     self.formations['formation_number'] = np.arange(1, self.formations.shape[0] + 1)

        # self.interfaces['formation_number'] = self.interfaces['formation'].map(self.formations.iloc[:, 1])
        # self.orientations['formation_number'] = self.orientations['formation'].map(self.formations.iloc[:, 1])
        #
        # self.interfaces['formation_value'] = self.interfaces['formation'].map(self.formations.iloc[:, 0])
        # self.orientations['formation_value'] = self.orientations['formation'].map(self.formations.iloc[:, 0])
        #
        # self.interfaces['formation'].cat.set_categories(formation_order, inplace=True)
        # self.orientations['formation'].cat.set_categories(formation_order, inplace=True)


class Data(object):
    def __init__(self):
        self.df = pn.DataFrame()

    def __repr__(self):
        return self.df.to_string()

    def _repr_html_(self):
        return self.df.to_html()

    def read_data(self, filepath, **kwargs):

        if 'sep' not in kwargs:
            kwargs['sep'] = ','

        table = pn.read_table(filepath, **kwargs)

        return table


    def sort_table(self):
        """
        First we sort the dataframes by the series age. Then we set a unique number for every formation and resort
        the formations. All inplace
        """

        # # We order the pandas table by series
        # self.df.sort_values(by=['order_series'],
        #                             ascending=True, kind='mergesort',
        #                             inplace=True)

        # We order the pandas table by formation (also by series in case something weird happened)
        self.df.sort_values(by=['order_series', 'id'],
                             ascending=True, kind='mergesort',
                             inplace=True)


        # Pandas dataframe set an index to every row when the dataframe is created. Sorting the table does not reset
        # the index. For some of the methods (pn.drop) we have to apply afterwards we need to reset these indeces
        # self.reset_indices()
        # DEP
        # self.interfaces.reset_index(drop=True, inplace=True)
        # self.orientations.reset_index(drop=True, inplace=True)

        # TODO check if this works
        # Update labels for anotations
        #self.set_annotations()

    def map_series(self, series):
        # Now we fill the column series in the interfaces and orientations tables with the correspondant series and
        # assigned number to the series
        series_df = series.df

        self.df["series"] = [(i == series_df).sum().idxmax() for i in self.df["formation"]]
        self.df["series"] = self.df["series"].astype('category')
        self.df["order_series"] = [(i == series_df).sum().values.argmax().astype(int) + 1
                                   for i in self.df["formation"]]
        # We sort the series altough is only important for the computation (we will do it again just before computing)
        # if series_distribution is not None:
        #   self.df.sort_values(by='order_series', inplace=True)

        self.df['series'].cat.set_categories(series_df.columns, inplace=True)
        return self.df

    def map_formations(self, formations, sort=True, inplace=True):
        i_df = self.df.drop('id', axis=1)
        new_df = pn.merge(i_df, formations.df, on=['formation'], how='left')
        #
        # formations_df = formations.df
        # formation_order = formations.df.index


        # TODO: Substitute by ID
        if inplace:
            self.df = new_df
            self.df['formation_number'] = self.df['id']
            self.set_dypes()
            self.df['formation'].cat.set_categories(formations.df['formation'].cat.categories, inplace=True)
            if sort:
                self.sort_table()

        else:
            return new_df
        # TODO: DEP
        # self.df['formation_value'] = self.df['formation'].map(formations_df['id'])
        # self.df['formation'].cat.set_categories(formation_order, inplace=True)

    def map_faults(self, faults, inplace=True):

        faults_df = faults.faults

        if any(self.df['series'].isna()):
            warnings.warn('Some points do not have series/fault')
        if inplace:

            self.df.loc[:, 'isFault'] = self.df['series'].isin(faults_df.index[faults_df['isFault']])
        else:
            return self.df['series'].isin(faults_df.index[faults_df['isFault']])

    def set_dypes(self):
        # Choose types
        self.df['formation'] = self.df['formation'].astype('category', copy=True)
        self.df['series'] = self.df['series'].astype('category', copy=True)
        self.df['isFault'] = self.df['isFault'].astype('bool')
        try:
            self.df[['formation_number', 'order_series', 'id']] = self.df[
                ['formation_number', 'order_series', 'id']].astype(int, copy=True)
        except ValueError:
            warnings.warn('You may have non-finite values (NA or inf) on the dataframe')

    def rescale_data(self, rescaling_factor, centers):
        self.df[['Xr', 'Yr', 'Zr']] = (self.df[['X', 'Y', 'Z']] - centers) / rescaling_factor + 0.5001




class Interfaces(Data):
    def __init__(self):

        super().__init__()
        self._columns_i_all = ['X', 'Y', 'Z', 'formation', 'series', 'X_std', 'Y_std', 'Z_std',
                               'order_series', 'formation_number']
        self._columns_i_1 = ['X', 'Y', 'Z', 'formation', 'series', 'id', 'formation_number', 'order_series', 'isFault']
        self._columns_i_num = ['X', 'Y', 'Z']
        self.df = pn.DataFrame(columns=self._columns_i_1)

        # Choose types
        self.df[self._columns_i_num] = self.df[self._columns_i_num].astype(float)
        self.set_dypes()
        self.df.itype = 'interfaces'

       # self.set_basement()
       # self.df['isFault'] = self.df['isFault'].astype('bool')

    def __str__(self):
        return 'fooo'

    def read_interfaces(self, filepath, debug=False, inplace=False, append=False, **kwargs):

        if 'sep' not in kwargs:
            kwargs['sep'] = ','

        table = pn.read_table(filepath, **kwargs)
        if debug is True:
            print('Debugging activated. Changes won\'t be saved.')
            return table
        else:
            assert set(['X', 'Y', 'Z', 'formation']).issubset(table.columns), \
                "One or more columns do not match with the expected values " + str(table.columns)

            # c = np.array(self._columns_i_1)
            # interfaces_read = table.assign(**dict.fromkeys(c[~np.in1d(c, table.columns)], False))
            # self.set_interfaces(interfaces_read, append=append)

            if inplace:
                # self.df[table.columns] = table
                c = np.array(self._columns_i_1)
                interfaces_read = table.assign(**dict.fromkeys(c[~np.in1d(c, table.columns)], np.nan))
                self.set_interfaces(interfaces_read, append=append)
                #self.set_interfaces(interfaces_read, append=append)
                self.df['formation'] = self.df['formation'].astype('category')

            else:
                return table

    def set_interfaces(self, interf_Dataframe, append=False, order_table=True):
        """
        Method to change or append a Dataframe to interfaces in place. A equivalent Pandas Dataframe with
        ['X', 'Y', 'Z', 'formation'] has to be passed.

        Args:
            interf_Dataframe: pandas.core.frame.DataFrame with the data
            append: Bool: if you want to append the new data frame or substitute it
        """
        assert set(self._columns_i_1).issubset(interf_Dataframe.columns), \
            "One or more columns do not match with the expected values " + str(self._columns_i_1)

        interf_Dataframe[self._columns_i_num] = interf_Dataframe[self._columns_i_num].astype(float, copy=True)
        try:
            interf_Dataframe[['formation_number', 'order_series']] = interf_Dataframe[
                ['formation_number', 'order_series']].astype(int, copy=True)
        except ValueError:
            print('No formation_number or order_series in the file')
            pass
        interf_Dataframe['formation'] = interf_Dataframe['formation'].astype('category', copy=True)
        interf_Dataframe['series'] = interf_Dataframe['series'].astype('category', copy=True)

        if append:
            self.df = self.df.append(interf_Dataframe)
        else:
            # self.interfaces[self._columns_i_1] = interf_Dataframe[self._columns_i_1]
            self.df = interf_Dataframe[self._columns_i_1]

        self.df = self.df[~self.df[['X', 'Y', 'Z']].isna().any(1)]

        # self.set_annotations()

        # self.set_annotations()
        if not self.df.index.is_unique:
            self.df.reset_index(drop=True, inplace=True)

        # if order_table:
        #     self.set_series()
        #     self.order_table()
        #
        # # # We check if in the df we are setting there is a new formation. if yes we append it to to the cat
        # # new_cat = interf_Dataframe['formation'].cat.categories[~np.in1d(interf_Dataframe['formation'].cat.categories,
        # #                                                                self.formations)]
        # # self.formations.index.insert(0, new_cat)
        # self.set_series()
        # self.set_formations()
        # self.set_faults()
        # self.interfaces.sort_index()


    def set_basement(self):
        pass

        # try:
        #     self.df['formation'].cat.add_categories('basement', inplace=True)
        # except ValueError:
        #     pass
        #
        # try:
        #     n_series = self.df['order_series'].unique().max()
        # except ValueError:
        #     n_series = 0
        #
        # drop_basement = self.df['formation'] == 'basement'
        # original_frame = self.df[~drop_basement]
        #
        # try:
        #     n_formation = original_frame['formation_number'].unique().max() + 1
        # except ValueError:
        #     n_formation = 1
        # l = len(self.df)
        #
        # if not 'basement' in self.df['formation'].values:
        #
        #     try:
        #         columns = {'X': self.extent[0], 'Y': self.extent[2], 'Z': self.extent[4], 'formation': 'basement',
        #                    'order_series': n_series, 'formation_number': n_formation, 'series': self.series.columns[-1]}
        #     except AttributeError:
        #         columns = {'X': self.extent[0], 'Y': self.extent[2], 'Z': self.extent[4], 'formation': 'basement',
        #                    'order_series': n_series, 'formation_number': n_formation, 'series': 'Default series'}
        #
        #     for key in columns:
        #         self.df.at[l, str(key)] = columns[key]
        #
        #     self.order_table()
        # # sef.add_interface(formation='basement', order_series=n_series, formation_number = n_formation)
        # else:
        #     self.modify_interface((drop_basement.index[drop_basement])[0], formation='basement', order_series=n_series,
        #                           formation_number=n_formation)
        #
        # self.order_table()

    # TODO any use here? -> moved to faults
    # def count_faults(self):
    #     """
    #     Read the string names of the formations to detect automatically the number of faults.
    #     """
    #     faults_series = []
    #     for i in self.df['series'].unique():
    #         try:
    #             if ('fault' in i or 'Fault' in i) and 'Default' not in i:
    #                 faults_series.append(i)
    #         except TypeError:
    #             pass
    #     return faults_series

    def set_default_interface(self, extent):
        # TODO this part is to update the formation class!!!!!!!!!!!!!!!!!!
        if self.formations.index[0] is 'basement':
            formation = 'default'
            self.set_formations(formation_order=[formation])
        else:
            formation = self.formations.index[0]
            # self.set_formations(formation_order=[self.formations.index])

        self.set_interfaces(pn.DataFrame({'X': [(extent[1] - extent[0]) / 2],
                                          'Y': [(extent[3] - extent[2]) / 2],
                                          'Z': [(extent[4] - extent[5]) / 2],
                                          'formation': [formation], 'order_series': [0],
                                          'formation_number': [1], 'series': ['Default series'],
                                          'isFault': False}))

        self.set_basement()

    def get_formations(self):
        """
        Returns:
             pandas.core.frame.DataFrame: Returns a list of formations

        """
        return self.df["formation"].unique()

    # def import_data_csv(self, path, **kwargs):
    #     """
    #     Method to import interfaces and orientations from csv. The format is the same as the export 3D model data of
    #     GeoModeller (check in the input_data data folder for an example).
    #
    #     Args:
    #         path_i (str): path to the csv table
    #         path_o (str): path to the csv table
    #         **kwargs: kwargs of :func: `~pn.read_csv`
    #
    #     Attributes:
    #         orientations(pandas.core.frame.DataFrame): Pandas data frame with the orientations data
    #         Interfaces(pandas.core.frame.DataFrame): Pandas data frame with the interfaces data
    #     """
    #
    #     interfaces_read = pn.read_csv(path, **kwargs)
    #     assert set(['X', 'Y', 'Z', 'formation']).issubset(interfaces_read.columns), \
    #         "One or more columns do not match with the expected values " + str(interfaces_read.columns)
    #
    #     c = np.array(self._columns_i_1)
    #     interfaces_read = interfaces_read.assign(**dict.fromkeys(c[~np.in1d(c, interfaces_read.columns)], False))
    #     self.set_interfaces(interfaces_read, append=True)
    #     # self.interfaces[interfaces_read.columns] = interfaces_read[interfaces_read.columns]
    #         # gagag
    #     self.update_df()

    # def map_series(self, series):
    #     # Now we fill the column series in the interfaces and orientations tables with the correspondant series and
    #     # assigned number to the series
    #     series_df = series.df
    #
    #     self.df["series"] = [(i == series_df).sum().idxmax() for i in self.df["formation"]]
    #     self.df["series"] = self.df["series"].astype('category')
    #     self.df["order_series"] = [(i == series_df).sum().values.argmax().astype(int) + 1
    #                                        for i in self.df["formation"]]
    #     # We sort the series altough is only important for the computation (we will do it again just before computing)
    #     # if series_distribution is not None:
    #  #   self.df.sort_values(by='order_series', inplace=True)
    #
    #     self.df['series'].cat.set_categories(series_df.columns, inplace=True)
    #     return self.df
    #
    # def map_formations(self, formations, sort=True, inplace=True):
    #
    #     new_df = pn.merge(self.df, formations.df, on='formation', how='outer')
    #     #
    #     # formations_df = formations.df
    #     # formation_order = formations.df.index
    #
    #     if sort:
    #         self.sort_table(new_df)
    #
    #     # TODO: Substitute by ID
    #     if inplace:
    #         self.df = new_df
    #         self.df['formation_number'] = self.df['id']
    #     else:
    #         return new_df
    #     # TODO: DEP
    #     #self.df['formation_value'] = self.df['formation'].map(formations_df['id'])
    #     #self.df['formation'].cat.set_categories(formation_order, inplace=True)
    #
    # def map_faults(self, faults):
    #
    #     faults_df = faults.faults
    #     self.df.loc[:, 'isFault'] = self.df['series'].isin(faults_df.index[faults_df['isFault']])

    def set_annotations(self):
        """
        Add a column in the Dataframes with latex names for each input_data paramenter.

        Returns:
            None
        """
        point_num = self.df.groupby('id').cumcount()
        point_l = [r'${\bf{x}}_{\alpha \,{\bf{' + str(f) + '}},' + str(p) + '}$'
                   for p, f in zip(point_num, self.df['id'])]

        self.df['annotations'] = point_l


class Orientations(Data):
    def __init__(self):
        super().__init__()
        self._columns_o_all = ['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', 'dip', 'azimuth', 'polarity',
                               'formation', 'series', 'id', 'order_series', 'formation_number']
        self._columns_o_1 = ['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', 'dip', 'azimuth', 'polarity', 'formation',
                             'series', 'id', 'formation_number', 'order_series', 'isFault']
        self._columns_o_num = ['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', 'dip', 'azimuth', 'polarity']
        self.df = pn.DataFrame(columns=self._columns_o_1)
        self.df[self._columns_o_num] = self.df[self._columns_o_num].astype(float)
        self.df.itype = 'orientations'
        
        self.calculate_gradient()
     #   self.df['isFault'] = self.df['isFault'].astype('bool')

    def calculate_gradient(self):
        """
        Calculate the gradient vector of module 1 given dip and azimuth to be able to plot the orientations

         Attributes:
            orientations: extra columns with components xyz of the unity vector.
        """

        self.df['G_x'] = np.sin(np.deg2rad(self.df["dip"].astype('float'))) * \
                                   np.sin(np.deg2rad(self.df["azimuth"].astype('float'))) * \
                                   self.df["polarity"].astype('float') + 1e-12
        self.df['G_y'] = np.sin(np.deg2rad(self.df["dip"].astype('float'))) * \
                                   np.cos(np.deg2rad(self.df["azimuth"].astype('float'))) * \
                                   self.df["polarity"].astype('float') + 1e-12
        self.df['G_z'] = np.cos(np.deg2rad(self.df["dip"].astype('float'))) * \
                                   self.df["polarity"].astype('float') + 1e-12

    def calculate_orientations(self):
        """
        Calculate and update the orientation data (azimuth and dip) from gradients in the data frame.
        """

        self.df["dip"] = np.rad2deg(
            np.nan_to_num(np.arccos(self.df["G_z"] / self.df["polarity"])))
        # TODO if this way to compute azimuth breaks there is in rgeomod=kml_to_plane line 170 a good way to do it
        self.df["azimuth"] = np.rad2deg(
            np.nan_to_num(np.arctan(self.df["G_x"] / self.df["G_y"])))
        self.df['azimuth'][
            (self.df['G_x'] < 0).values * (self.df['G_y'] >= 0).values] += 360
        self.df['azimuth'][(self.df['G_y'] < 0).values] += 180
        self.df['azimuth'][
            (self.df['G_x'] > 0).values * (self.df['G_y'] == 0).values] = 90
        self.df['azimuth'][
            (self.df['G_x'] < 0).values * (self.df['G_y'] == 0).values] = 270

    def create_orientation_from_interface(self, indices):
        selected_points = self.interfaces[['X', 'Y', 'Z']].loc[indices].values.T

        center, normal = self.plane_fit(selected_points)
        orientation = get_orientation(normal)

        return np.array([*center, *orientation, *normal])

    # TODO: Update!!!!!1
    def set_default_orientation(self, extent):
        ori = pn.DataFrame([[(extent[1] - extent[0]) / 2,
                             (extent[3] - extent[2]) / 2,
                             (extent[4] - extent[5]) / 2,
                             0, 0, 1,
                             0, 0, 1,
                             'basement',
                             'Default series',
                             1, 1, False]], columns=self._columns_o_1)

        self.set_orientations(ori)

    def read_orientations(self, filepath, debug=False, inplace=False, append=False, **kwargs):

        if 'sep' not in kwargs:
            kwargs['sep'] = ','

        table = pn.read_table(filepath, **kwargs)
        if debug is True:
            print('Debugging activated. Changes won\'t be saved.')
            return table
        else:
            assert set(['X', 'Y', 'Z', 'dip', 'azimuth', 'polarity', 'formation']).issubset(table.columns), \
                "One or more columns do not match with the expected values " + str(table.columns)

            if inplace:
               # self.df[table.columns] = table
                c = np.array(self._columns_o_1)
                orientations_read = table.assign(**dict.fromkeys(c[~np.in1d(c, table.columns)], np.nan))
                self.set_orientations(orientations_read, append=append)
                #self.set_interfaces(interfaces_read, append=append)
                self.df['formation'] = self.df['formation'].astype('category')

            else:
                return table

    def set_orientations(self, foliat_Dataframe, append=False, order_table=True):
        """
          Method to change or append a Dataframe to orientations in place.  A equivalent Pandas Dataframe with
        ['X', 'Y', 'Z', 'dip', 'azimuth', 'polarity', 'formation'] has to be passed.

          Args:
              interf_Dataframe: pandas.core.frame.DataFrame with the data
              append: Bool: if you want to append the new data frame or substitute it
          """
        assert set(self._columns_o_1).issubset(
            foliat_Dataframe.columns), "One or more columns do not match with the expected values " +\
                                       str(self._columns_o_1)

        foliat_Dataframe[self._columns_o_num] = foliat_Dataframe[self._columns_o_num].astype(float, copy=True)

        if append:
            self.df = self.orientations.append(foliat_Dataframe)
        else:
            #self.orientations[self._columns_o_1] = foliat_Dataframe[self._columns_o_1]
            self.df = foliat_Dataframe[self._columns_o_1]

        # self.calculate_orientations()
        self.calculate_gradient()

       #  self.orientations = self.orientations[~self.orientations[['X', 'Y', 'Z']].isna().any(1)]
       #
       #  self.set_series()
       #  self.set_formations()
       #  self.set_faults()
       # # self.set_annotations()
       #
       #  if order_table:
       #    #  self.set_formation_number()
       #      self.set_series()
       #      self.order_table()
       #
       #  self.orientations.sort_index()

    # def import_data_csv(self, path, **kwargs):
    #     """
    #     Method to import interfaces and orientations from csv. The format is the same as the export 3D model data of
    #     GeoModeller (check in the input_data data folder for an example).
    #
    #     Args:
    #         path_i (str): path to the csv table
    #         path_o (str): path to the csv table
    #         **kwargs: kwargs of :func: `~pn.read_csv`
    #
    #     Attributes:
    #         orientations(pandas.core.frame.DataFrame): Pandas data frame with the orientations data
    #         Interfaces(pandas.core.frame.DataFrame): Pandas data frame with the interfaces data
    #     """
    #
    #     orientations_read = pn.read_csv(path, **kwargs)
    #
    #     assert set(['X', 'Y', 'Z', 'dip', 'azimuth', 'polarity', 'formation']).issubset(orientations_read.columns), \
    #         "One or more columns do not match with the expected values " + str(orientations_read.columns)
    #
    #     self.orientations[orientations_read.columns] = orientations_read[orientations_read.columns]
    #
    #     self.update_df()
    #
    # def map_series(self, series_df):
    #     # Now we fill the column series in the interfaces and orientations tables with the correspondant series and
    #     # assigned number to the series
    #     self.df["series"] = [(i == series_df).sum().idxmax() for i in self.df["formation"]]
    #     self.df["series"] = self.df["series"].astype('category')
    #     self.df["order_series"] = [(i == series_df).sum().values.argmax().astype(int) + 1
    #                                          for i in self.df["formation"]]
    #
    #     # We sort the series altough is only important for the computation (we will do it again just before computing)
    #     # if series_distribution is not None:
    #     self.df.sort_values(by='order_series', inplace=True)
    #
    #     self.df['series'].cat.set_categories(series_df.columns, inplace=True)
    #
    # def map_formations(self, formations):
    #     formations_df = formations.df
    #     formation_order = formations.df.index
    #
    #     self.df['formation_number'] = self.df['formation'].map(formations_df.iloc[:, 1])
    #
    #     self.df['formation_value'] = self.df['formation'].map(formations_df.iloc[:, 0])
    #
    #     self.df['formation'].cat.set_categories(formation_order, inplace=True)
    #
    # def map_faults(self, faults):
    #
    #     faults_df = faults.df
    #     self.df.loc[:, 'isFault'] = self.df['series'].isin(
    #         faults_df.index[faults_df['isFault']])

    def set_annotations(self):
        """
        Add a column in the Dataframes with latex names for each input_data paramenter.

        Returns:
            None
        """

        orientation_num = self.df.groupby('formation_number').cumcount()
        foli_l = [r'${\bf{x}}_{\beta \,{\bf{' + str(f) + '}},' + str(p) + '}$'
                   for p, f in zip(orientation_num, self.df['formation_number'])]

        self.df['annotations'] = foli_l

def get_orientation(normal):
    """Get orientation (dip, azimuth, polarity ) for points in all point set"""
    #    if "normal" not in dir(self):
    #        self.plane_fit()

    # calculate dip
    dip = np.arccos(normal[2]) / np.pi * 180.

    # calculate dip direction
    # +/+
    if normal[0] >= 0 and normal[1] > 0:
        dip_direction = np.arctan(normal[0] / normal[1]) / np.pi * 180.
    # border cases where arctan not defined:
    elif normal[0] > 0 and normal[1] == 0:
        dip_direction = 90
    elif normal[0] < 0 and normal[1] == 0:
        dip_direction = 270
    # +-/-
    elif normal[1] < 0:
        dip_direction = 180 + np.arctan(normal[0] / normal[1]) / np.pi * 180.
    # -/-
    elif normal[0] < 0 and normal[1] >= 0:
        dip_direction = 360 + np.arctan(normal[0] / normal[1]) / np.pi * 180.
    # if dip is just straight up vertical
    elif normal[0] == 0 and normal[1] == 0:
        dip_direction = 0

    if -90 < dip < 90:
        polarity = 1
    else:
        polarity = -1

    return dip, dip_direction, polarity


def plane_fit(point_list):
    """
    Fit plane to points in PointSet
    Fit an d-dimensional plane to the points in a point set.
    adjusted from: http://stackoverflow.com/questions/12299540/plane-fitting-to-4-or-more-xyz-points

    Args:
        point_list (array_like): array of points XYZ

    Returns:
        Return a point, p, on the plane (the point-cloud centroid),
        and the normal, n.
    """

    import numpy as np

    #     points = np.empty((3, len(point_list)))
    #     for i, point in enumerate(point_list):
    #         points[0, i] = point.x
    #         points[1, i] = point.y
    #         points[2, i] = point.z
    points = point_list

    from numpy.linalg import svd
    points = np.reshape(points, (np.shape(points)[0], -1))  # Collapse trialing dimensions
    assert points.shape[0] <= points.shape[1], "There are only {} points in {} dimensions.".format(points.shape[1],
                                                                                                   points.shape[0])
    ctr = points.mean(axis=1)
    x = points - ctr[:, np.newaxis]
    M = np.dot(x, x.T)  # Could also use np.cov(x) here.

    # ctr = Point(x=ctr[0], y=ctr[1], z=ctr[2], type='utm', zone=self.points[0].zone)
    normal = svd(M)[0][:, -1]
    # return ctr, svd(M)[0][:, -1]
    if normal[2] < 0:
        normal = - normal

    return ctr, normal


class RescaledData(object):
    def __init__(self, interfaces: Interfaces, orientations: Orientations, grid: GridClass,
                 rescaling_factor=None, centers=None):

        self.interfaces = interfaces
        self.orientations = orientations
        self.grid = grid

        max_coord, min_coord = self.max_min_coord(interfaces, orientations)
        if not rescaling_factor:
            self.rescaling_factor = self.compute_rescaling_factor(interfaces, orientations,
                                                                  max_coord, min_coord)
        else:
            self.rescaling_factor = rescaling_factor
        if not centers:
            self.centers = self.compute_data_center(interfaces, orientations,
                                                    max_coord, min_coord)
        else:
            self.centers = centers

        self.set_rescaled_interfaces()
        self.set_rescaled_orientations()
        self.set_rescaled_grid()

    def get_rescaled_data(self):
        pass

    @staticmethod
    def max_min_coord(interfaces=None, orientations=None):
        if interfaces is None:
            #df = orientations
            if orientations is None:
                raise AttributeError('You must pass at least one Data object')
            else:
                df = orientations.df
        else:
            if orientations is None:
                df = interfaces.df
            else:
                df = pn.concat([orientations.df, interfaces.df], sort=False)

        max_coord = df.max()[['X', 'Y', 'Z']]
        min_coord = df.min()[['X', 'Y', 'Z']]
        return max_coord, min_coord

    def compute_data_center(self, interfaces=None, orientations=None,
                    max_coord=None, min_coord=None):

        if max_coord is None or min_coord is None:
            max_coord, min_coord = self.max_min_coord(interfaces, orientations)

        # Get the centers of every axis
        centers = ((max_coord + min_coord) / 2).astype(float)
        return centers

    def compute_rescaling_factor(self, interfaces=None, orientations=None,
                                 max_coord=None, min_coord=None):
        if max_coord is None or min_coord is None:
            max_coord, min_coord = self.max_min_coord(interfaces, orientations)
        rescaling_factor_val = (2 * np.max(max_coord - min_coord))
        return rescaling_factor_val

    @staticmethod
    def rescale_interfaces(interfaces, rescaling_factor, centers):
        # Change the coordinates of interfaces
        new_coord_interfaces = (interfaces.df[['X', 'Y', 'Z']] -
                                centers) / rescaling_factor + 0.5001
        return new_coord_interfaces

    def set_rescaled_interfaces(self):

        self.interfaces.df[['X_r', 'Y_r', 'Z_r']] = self.rescale_interfaces(self.interfaces, self.rescaling_factor,
                                                                            self.centers)

        return self.interfaces

    @staticmethod
    def rescale_orientations(orientations, rescaling_factor, centers):
        # Change the coordinates of orientations
        new_coord_orientations = (orientations.df[['X', 'Y', 'Z']] -
                                  centers) / rescaling_factor + 0.5001
        return new_coord_orientations

    def set_rescaled_orientations(self):

        self.orientations.df[['X_r', 'Y_r', 'Z_r']] = self.rescale_orientations(self.orientations, self.rescaling_factor,
                                                                                self.centers)

        return self.orientations

    @staticmethod
    def rescale_grid(grid, rescaling_factor, centers):
        new_grid_extent = (grid.extent - np.repeat(centers, 2)) / rescaling_factor + 0.5001
        new_grid_values = (grid.values - centers.values) / rescaling_factor + 0.5001
        return new_grid_extent, new_grid_values

    def set_rescaled_grid(self):
        self.grid.extent_r, self.grid.values_r = self.rescale_grid(self.grid, self.rescaling_factor, self.centers)


class Structure(object):
    def __init__(self, interfaces, orientations):

        self.len_formations_i = self.set_length_formations_i(interfaces)
        self.len_series_i = self.set_length_series_i(interfaces)
        self.len_series_o = self.set_length_series_o(orientations)
        self.ref_position = self.set_ref_position()
        self.nfs = self.set_number_of_formations_per_series(interfaces)

    def set_length_formations_i(self, interfaces):
        # ==================
        # Extracting lengths
        # ==================
        # Array containing the size of every formation. Interfaces
        self.len_formations_i = interfaces.df['id'].value_counts(sort=False).values
        return self.len_formations_i

    def set_length_series_i(self, interfaces):

        # Array containing the size of every series. Interfaces.
        len_series_i = interfaces.df['order_series'].value_counts(sort=False).values

        if len_series_i.shape[0] is 0:
            len_series_i = np.insert(len_series_i, 0, 0)

        self.len_series_i = len_series_i
        return self.len_series_i

    def set_length_series_o(self, orientations):
        # Array containing the size of every series. orientations.
        self.len_series_o = orientations.df['id'].value_counts(sort=False).values
        return self.len_series_o

    def set_ref_position(self):
        self.ref_position = np.insert(self.len_formations_i[:-1], 0, 0).cumsum()
        return self.ref_position

    def set_number_of_formations_per_series(self, interfaces):
        self.nfs = interfaces.df.groupby('order_series').formation.nunique().values.cumsum()
        return self.nfs


class AdditionalData(Structure, RescaledData):
    def __init__(self, interfaces, orientations, grid, faults, formations, rescaling):
        # TODO: probably not all the attributes need to be present until I do a check before computing the thing.

        super().__init__(interfaces, orientations)

        self.n_faults = faults.n_faults
        self.n_formations = formations.df.shape[0]

        self.range_var = self.default_range(grid.extent)
        self.c_o = self.default_c_o()

        self.n_universal_eq = self.set_u_grade()

        self.nugget_effect_gradient = 0.01
        self.nugget_effect_scalar = 1e-6

        self.kriging_data = pn.DataFrame([self.range_var, self.c_o, self.n_universal_eq,
                                             self.nugget_effect_gradient, self.nugget_effect_scalar],
                                            columns=['values'],
                                            index=['range', '$C_o$', 'drift equations',
                                                   'nugget grad', 'nugget scalar'])

        self.rescaled_data = RescaledData(interfaces, orientations, grid)

        self.options = pn.DataFrame(columns=['values'],
                                    index=['dtype', 'output', 'theano_optimizer', 'device', 'verbosity'])
        self.default_options()

        self.structure_data = pn.DataFrame([self.is_lith_is_fault()[0], self.is_lith_is_fault()[1],
                                            self.n_faults, self.n_formations, self.nfs,
                                            self.len_formations_i, self.len_series_i,
                                            self.len_series_o],
                                           columns=['values'],
                                           index=['isFault', 'isLith',
                                                  'number faults', 'number formations', 'number formations per series',
                                                  'len formations interfaces', 'len series interfaces',
                                                  'len series orientations'])

        self.rescaling_data = pn.DataFrame([rescaling.rescaling_factor, rescaling.centers],
                                           columns=['values'],
                                           index=['rescaling factor', 'centers'])

    def __repr__(self):

        concat_ = self.get_additional_data()
        return concat_.to_string()

    def _repr_html_(self):
        concat_ = self.get_additional_data()
        return concat_.to_html()

    def get_additional_data(self):
        concat_ = pn.concat([self.structure_data, self.options, self.kriging_data, self.rescaling_data],
                            keys=['Structure', 'Options', 'Kringing', 'Rescaling'])
        return concat_

    def is_lith_is_fault(self):

        is_fault = False
        is_lith = False

        if self.n_faults != 0:
            is_fault = True

        if self.n_formations - 1 > self.n_faults:
            is_lith = True

        return [is_fault, is_lith]

    def default_options(self):
        self.options.at['dtype'] = 'float64'
        self.options.at['output'] = 'geology'
        self.options.at['theano_optimizer'] = 'fast_compile'
        self.options.at['device'] = 'cpu'

    @staticmethod
    def default_range(extent):

        range_var = np.sqrt(
            (extent[0] - extent[1]) ** 2 +
            (extent[2] - extent[3]) ** 2 +
            (extent[4] - extent[5]) ** 2)

        return range_var

    def default_c_o(self):
        c_o = self.range_var ** 2 / 14 / 3
        return c_o

    def set_u_grade(self, **kwargs):

        u_grade = kwargs.get('u_grade', None)

        # =========================
        # Choosing Universal drifts
        # =========================
        if u_grade is None:
            u_grade = np.zeros_like(self.len_series_i)
            u_grade[(self.len_series_i > 1)] = 1

        else:
            u_grade = np.array(u_grade)

        n_universal_eq = np.zeros_like(self.len_series_i)
        n_universal_eq[u_grade == 0] = 0
        n_universal_eq[u_grade == 1] = 3
        n_universal_eq[u_grade == 2] = 9

        self.n_universal_eq = n_universal_eq
        return self.n_universal_eq

    def get_kriging_parameters(self):
        pass

    # @staticmethod
    # def compute_rescaling_parameters(interfaces, orientations, rescaling_factor=None):
    #     # Check which axis is the largest
    #     max_coord = pn.concat(
    #         [orientations.df, interfaces.df]).max()[['X', 'Y', 'Z']]
    #     min_coord = pn.concat(
    #         [orientations.df, interfaces.df]).min()[['X', 'Y', 'Z']]
    #
    #     # Compute rescalin factor if not given
    #     if not rescaling_factor:
    #         rescaling_factor = (2 * np.max(max_coord - min_coord))
    #
    #     # Get the centers of every axis
    #     centers = ((max_coord + min_coord) / 2).astype(float)
    #     return rescaling_factor, centers
    #
    # def set_rescaling_parameters(self, interfaces, orientations, rescaling_factor=None):
    #     self.rescaling_factor, self.centers = self.compute_rescaling_parameters(interfaces, orientations,
    #                                                                             rescaling_factor)


class GeoPhysiscs(object):
    def __init__(self):
        self.gravity = None
        self.magnetics = None

    def create_geophy(self):
        pass

    def set_gravity_precomputations(self):
        pass


class Solution(object):

    def __init__(self):
        self.scalar_field_at_interfaces = 0
        self.scalar_field = np.array([])
        self.lith_block = None
        self.values_block = None
        self.gradient = None


class Interpolator(Solution):
    # TODO assert passed data is rescaled
    def __init__(self, interfaces: Interfaces, orientations: Orientations, grid: GridClass,
                 formations: Formations, faults: Faults, additional_data: AdditionalData, **kwargs):
        # self.verbose = None
        # self.dtype = None
        # self.output = None
        # self.theano_optimizer = None
        # self.is_lith=None
        # self.is_fault=None
        self.interfaces = interfaces
        self.orientations = orientations
        self.grid = grid
        self.additional_data = additional_data
        self.formations = formations
        self.faults = faults

        self.dtype = additional_data.get_additional_data().xs('Options').loc['dtype', 'values']


        self.input_matrices = self.get_input_matrix()
        self.theano_graph = self.create_theano_graph(additional_data, inplace=False)
        if 'compile_theano' in kwargs:
            self.theano_function = self.compile_th_fn(additional_data.options.loc['output'])
        else:
            self.theano_function = None

    def create_theano_graph(self, additional_data: AdditionalData = None, inplace=True):

        import gempy.core.theano_graph as tg
        import importlib
        importlib.reload(tg)

        if additional_data is None:
            additional_data = self.additional_data

        options = additional_data.get_additional_data().xs('Options')
        graph = tg.TheanoGraph(output=options.loc['output', 'values'], optimizer=options.loc['theano_optimizer', 'values'],
                               dtype=options.loc['dtype', 'values'], verbose=options.loc['verbosity', 'values'],
                               is_lith=additional_data.structure_data.loc['isLith', 'values'],
                               is_fault=additional_data.structure_data.loc['isFault', 'values'])
        if inplace:
            self.theano_graph = graph
        else:
            return graph

    def set_theano_shared_parameters(self, **kwargs):
        # TODO: I have to split this one between structure and init data
        # Size of every layer in rests. SHARED (for theano)
        len_rest_form = (self.additional_data.structure_data.loc['len formations interfaces', 'values'] - 1)
        self.theano_graph.number_of_points_per_formation_T.set_value(len_rest_form.astype('int32'))
        self.theano_graph.npf.set_value(np.cumsum(np.concatenate(([0], len_rest_form))).astype('int32'))  # Last value is useless
        # and breaks the basement
        # Cumulative length of the series. We add the 0 at the beginning and set the shared value. SHARED
        self.theano_graph.len_series_i.set_value(
            np.insert(self.additional_data.structure_data.loc['len series interfaces', 'values'] -
                      self.additional_data.structure_data.loc['number formations per series', 'values'], 0, 0).cumsum().astype('int32'))
        # Cumulative length of the series. We add the 0 at the beginning and set the shared value. SHARED
        self.theano_graph.len_series_f.set_value(
            np.insert(self.additional_data.structure_data.loc['len series orientations', 'values'], 0, 0).cumsum().astype('int32'))
        # Setting shared variables
        # Range
        self.theano_graph.a_T.set_value(np.cast[self.dtype](self.additional_data.kriging_data.loc['range', 'values']))
        # Covariance at 0
        self.theano_graph.c_o_T.set_value(np.cast[self.dtype](self.additional_data.kriging_data.loc['$C_o$', 'values']))
        # universal grades
        self.theano_graph.n_universal_eq_T.set_value(
            list(self.additional_data.kriging_data.loc['drift equations', 'values'].astype('int32')))
        # nugget effect
        self.theano_graph.nugget_effect_grad_T.set_value(
            np.cast[self.dtype](self.additional_data.kriging_data.loc['nugget grad', 'values']))
        self.theano_graph.nugget_effect_scalar_T.set_value(
            np.cast[self.dtype](self.additional_data.kriging_data.loc['nugget scalar', 'values']))
        # Just grid. I add a small number to avoid problems with the origin point
        #x_0 = self.compute_x_0()
        self.theano_graph.grid_val_T.set_value(np.cast[self.dtype](self.grid.values_r + 10e-9))
        # Universal grid
        # TODO: this goes inside
       # self.theano_graph.universal_grid_matrix_T.set_value(np.cast[self.dtype](self.compute_universal_matrix(x_0) + 1e-10))
        # Initialization of the block model
        self.theano_graph.final_block.set_value(np.zeros((1, self.grid.values_r.shape[0] + self.interfaces.df.shape[0]),
                                               dtype=self.dtype))
        # Unique number assigned to each lithology
        self.theano_graph.n_formation.set_value(self.formations.df['id'])
        # Final values the lith block takes
        self.theano_graph.formation_values.set_value(self.formations.df['value_0'])
        # Number of formations per series. The function is not pretty but the result is quite clear
        n_formations_per_serie = np.insert(self.additional_data.structure_data.loc['number formations per series', 'values'], 0, 0).\
            astype('int32')
        self.theano_graph.n_formations_per_serie.set_value(n_formations_per_serie)
        # Init the list to store the values at the interfaces. Here we init the shape for the given dataset
        self.theano_graph.final_scalar_field_at_formations.set_value(np.zeros(self.theano_graph.n_formations_per_serie.get_value()[-1],
                                                                     dtype=self.dtype))
        self.theano_graph.final_scalar_field_at_faults.set_value(np.zeros(self.theano_graph.n_formations_per_serie.get_value()[-1],
                                                                 dtype=self.dtype))

        self.theano_graph.n_faults.set_value(self.additional_data.structure_data.loc['number faults', 'values'])
        # Set fault relation matrix
       # self.check_fault_ralation()
        self.theano_graph.fault_relation.set_value(self.faults.faults_relations.values.astype('int32'))

    def get_input_matrix(self):
        # orientations, this ones I tile them inside theano. PYTHON VAR
        dips_position = self.orientations.df[['X_r', 'Y_r', 'Z_r']].values
        dip_angles = self.orientations.df["dip"].values
        azimuth = self.orientations.df["azimuth"].values
        polarity = self.orientations.df["polarity"].values
        interfaces_coord = self.interfaces.df[['X_r', 'Y_r', 'Z_r']].values
        #ref_layer_points = self.pandas_ref_layer_points_rep[['X', 'Y', 'Z']].as_matrix()
        #rest_layer_points = self.pandas_rest_layer_points[['X', 'Y', 'Z']].as_matrix()

        # Set all in a list casting them in the chosen dtype
        idl = [np.cast[self.dtype](xs) for xs in (dips_position, dip_angles, azimuth, polarity, interfaces_coord)]
        return idl

    # def set_x_0(self):
    #     # TODO In principle this goes inside too
    #     x_0 = np.vstack((self.grid.values_r,
    #                      self.interfces.df[['X_r', 'Y_r', 'Z_r']].values
    #                      ))
    #
    #     return x_0

    def compile_th_fn(self, output, inplace=True, **kwargs):
        """
        Compile the theano function given the input_data data.

        Args:
            compute_all (bool): If true the solution gives back the block model of lithologies, the potential field and
             the block model of faults. If False only return the block model of lithologies. This may be important to speed
              up the computation. Default True

        Returns:
            theano.function: Compiled function if C or CUDA which computes the interpolation given the input_data data
            (XYZ of dips, dip, azimuth, polarity, XYZ ref interfaces, XYZ rest interfaces)
        """
        import theano
        self.set_theano_shared_parameters()
        # This are the shared parameters and the compilation of the function. This will be hidden as well at some point
        input_data_T = self.theano_graph.input_parameters_list()

        print('Compiling theano function...')

        if output is 'geology':
            # then we compile we have to pass the number of formations that are faults!!
            th_fn = theano.function(input_data_T,
                                    self.theano_graph.compute_geological_model(),
                                    # mode=NanGuardMode(nan_is_error=True),
                                    on_unused_input='ignore',
                                    allow_input_downcast=False,
                                    profile=False)

        elif output is 'gravity':
            # then we compile we have to pass the number of formations that are faults!!
            th_fn = theano.function(input_data_T,
                                    self.theano_graph.compute_forward_gravity(),
                                    #  mode=NanGuardMode(nan_is_error=True),
                                    on_unused_input='ignore',
                                    allow_input_downcast=False,
                                    profile=False)

        elif output is 'gradients':

            gradients = kwargs.get('gradients', ['Gx', 'Gy', 'Gz'])
            self.theano_graph.gradients = gradients

            # then we compile we have to pass the number of formations that are faults!!
            th_fn = theano.function(input_data_T,
                                    self.theano_graph.compute_geological_model_gradient(
                                        self.additional_data.structure_data['number faults']),
                                    #  mode=NanGuardMode(nan_is_error=True),
                                    on_unused_input='ignore',
                                    allow_input_downcast=False,
                                    profile=False)

        else:
            raise SyntaxError('The output given does not exist. Please use geology, gradients or gravity ')

        if inplace is True:
            self.theano_function = th_fn

        print('Compilation Done!')
        print('Level of Optimization: ', theano.config.optimizer)
        print('Device: ', theano.config.device)
        print('Precision: ', self.dtype)
        print('Number of faults: ', self.additional_data.structure_data.loc['number faults', 'values'])
        return th_fn



