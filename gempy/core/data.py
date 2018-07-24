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

        self.resolution = None
        self.extent = None
        self.values = None

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
        self.df = None

    def set_series(self, interfaces=None, series_distribution=None, order=None):
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
        
        # Default series df. We extract the formations from Interfaces
        if series_distribution is None and interfaces is not None:
            if self.series is None:
                self.series = pn.DataFrame({"Default series": self.interfaces["formation"].unique().astype(list)},
                                           dtype=str)
        # We pass a df or dictionary with the right shape
        else:
            if type(series_distribution) is dict:
                if order is None:
                    order = series_distribution.keys()
                else:
                    assert all(np.in1d(order, list(series_distribution.keys()))), 'Order series must contain the same keys as' \
                                                                       'the passed dictionary ' + str(series_distribution.keys())
                self.series = pn.DataFrame(dict([(k, pn.Series(v)) for k, v in series_distribution.items()]),
                                           columns=order)

            elif type(series_distribution) is pn.core.frame.DataFrame:
                self.series = series_distribution

            else:
                raise AttributeError('series_distribution must be a dictionary, see Docstring for more information')

        if 'basement' not in self.series.iloc[:, -1].values:
            self.series.loc[self.series.shape[0], self.series.columns[-1]] = 'basement'

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

        return self.series

    def map_formations(self, formations_df):
        
        # TODO review
        # Addind the formations of the new series to the formations df
        new_formations = self.series.values.reshape(1, -1)
        # Dropping nans
        new_formations = new_formations[~pn.isna(new_formations)]
        self.set_formations(formation_order=new_formations)
    
    
class Faults(object):
    def __init__(self):
        self.faults_relations = None
        self.faults = None
        self.n_faults = None

    def set_faults(self, series, series_name=None):
        """
        Set a flag to the series that are faults.

        Args:
            series_name(list or array_like): Name of the series which are faults
        """

        try:
            # Check if there is already a df
            self.faults

            try:
                if any(self.faults.columns != series.columns):
                    series_name = self.count_faults()
                    self.faults = pn.DataFrame(index=series.columns, columns=['isFault'])
                    self.faults['isFault'] = self.faults.index.isin(series_name)
            except ValueError:
                series_name = self.count_faults()
                self.faults = pn.DataFrame(index=series.columns, columns=['isFault'])
                self.faults['isFault'] = self.faults.index.isin(series_name)

            if series_name:
                self.faults['isFault'] = self.faults.index.isin(series_name)

        except AttributeError:

            if not series_name:
                series_name = self.count_faults()
                self.faults = pn.DataFrame(index=series.columns, columns=['isFault'])
                self.faults['isFault'] = self.faults.index.isin(series_name)

        # self.interfaces.loc[:, 'isFault'] = self.interfaces['series'].isin(self.faults.index[self.faults['isFault']])
        # self.orientations.loc[:, 'isFault'] = self.orientations['series'].isin(
        #     self.faults.index[self.faults['isFault']])

        self.n_faults = self.faults['isFault'].sum()

    def check_fault_relations(self):
        pass

    def set_fault_relation(self, series, rel_matrix=None):
        """
        Method to set the faults that offset a given sequence and therefore also another fault

        Args:
            rel_matrix (numpy.array): 2D Boolean array with the logic. Rows affect (offset) columns
        """
        # TODO: Change the fault relation automatically every time we add a fault
        try:
            self.faults_relations
            if not rel_matrix:

                rel_matrix = np.zeros((series.df.columns.shape[0],
                                       series.df.columns.shape[0]))

            self.faults_relations = pn.DataFrame(rel_matrix, index=series.df.columns,
                                                 columns=series.df.columns, dtype='bool')
        except AttributeError:

            if rel_matrix is not None:
                self.faults_relations = pn.DataFrame(rel_matrix, index=series.df.columns,
                                                     columns=series.df.columns, dtype='bool')


class Formations(object):
    def __init__(self):
        self.formations = None
        self.sequential_pile = None
        self._formation_values_set = False

    def set_formations(self, formation_values=None, interfaces_df=None, formation_order=None):

        #self.orientations['formation'] = self.orientations['formation'].astype('category')

        if formation_order is None and interfaces_df is not None:
            if self.formations is None:
                interfaces_df['formation'] = interfaces_df['formation'].astype('category')
                # if self._formation_values_set is False:
                formation_order = interfaces_df['formation'].cat.categories
            else:
                # We check if in the df we are setting there is a new formation. if yes we append it to to the cat
                new_cat = interfaces_df['formation'].cat.categories[
                    ~np.in1d(interfaces_df['formation'].cat.categories,
                             self.formations.index)]
                if new_cat.empty:
                    formation_order = self.formations.index
                else:
                    formation_order = np.insert(self.formations.index.get_values(), 0, new_cat)
            # try:
            #     # Check if there is already a df
            #     formation_order = self.formations.index
            #
            # except AttributeError:
            #

        if 'basement' not in formation_order:
            formation_order = np.append(formation_order, 'basement')

        if formation_values is None:
            if self._formation_values_set:
                # Check if there is already a df
                formation_values = self.formations['value'].squeeze()
            else:
                formation_values = np.arange(1, formation_order.shape[0] + 1)
        else:
            self._formation_values_set = True

        if np.atleast_1d(formation_values).shape[0] < np.atleast_1d(formation_order).shape[0]:
            formation_values = np.append(formation_values, formation_values.max() + 1)

        self.formations = pn.DataFrame(index=formation_order,
                                       columns=['value', 'formation_number'])

        self.formations['value'] = formation_values
        self.formations['formation_number'] = np.arange(1, self.formations.shape[0] + 1)

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

    def import_data(self):
        pass

    def order_table(self):
        """
        First we sort the dataframes by the series age. Then we set a unique number for every formation and resort
        the formations. All inplace
        """

        # We order the pandas table by series
        self.df.sort_values(by=['order_series'],
                                    ascending=True, kind='mergesort',
                                    inplace=True)
        # Give formation_number
        if not 'formation_number' in self.df.columns:
            self.map_formations()

        # We order the pandas table by formation (also by series in case something weird happened)
        self.df.sort_values(by=['order_series', 'formation_number'],
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
        self.set_annotations()

    def map_series(self, series):
        # Now we fill the column series in the interfaces and orientations tables with the correspondant series and
        # assigned number to the series
        series_df = series.df

        self.df["series"] = [(i == series_df).sum().idxmax() for i in self.df["formation"]]
        self.df["series"] = self.df["series"].astype('category')
        self.df["order_series"] = [(i == series_df).sum().as_matrix().argmax().astype(int) + 1
                                   for i in self.df["formation"]]
        # We sort the series altough is only important for the computation (we will do it again just before computing)
        # if series_distribution is not None:
        self.df.sort_values(by='order_series', inplace=True)

        self.df['series'].cat.set_categories(series_df.columns, inplace=True)

    def map_formations(self, formations):
        formations_df = formations.df
        formation_order = formations.df.index

        self.df['formation_number'] = self.df['formation'].map(formations_df.iloc[:, 1])
        self.df['formation_value'] = self.df['formation'].map(formations_df.iloc[:, 0])
        self.df['formation'].cat.set_categories(formation_order, inplace=True)

    def map_faults(self, faults):
        faults_df = faults.df
        self.df.loc[:, 'isFault'] = self.df['series'].isin(faults_df.index[self.faults['isFault']])

    def rescale_data(self):
        pass


class Interfaces(object, Data):
    def __init__(self):
  
        self._columns_i_all = ['X', 'Y', 'Z', 'formation', 'series', 'X_std', 'Y_std', 'Z_std',
                               'order_series', 'formation_number']
        self._columns_i_1 = ['X', 'Y', 'Z', 'formation', 'series', 'formation_number', 'order_series', 'isFault']
        self._columns_i_num = ['X', 'Y', 'Z']
        self.df = pn.DataFrame(columns=self._columns_i_1)
        self.df[self._columns_i_num] = self.interfaces[self._columns_i_num].astype(float)
        self.df.itype = 'interfaces'
        
        self.set_basement()
        self.df['isFault'] = self.df['isFault'].astype('bool')

    def set_basement(self):
        try:
            self.df['formation'].cat.add_categories('basement', inplace=True)
        except ValueError:
            pass

        try:
            n_series = self.df['order_series'].unique().max()
        except ValueError:
            n_series = 0

        drop_basement = self.df['formation'] == 'basement'
        original_frame = self.df[~drop_basement]

        try:
            n_formation = original_frame['formation_number'].unique().max() + 1
        except ValueError:
            n_formation = 1
        l = len(self.interfaces)

        if not 'basement' in self.df['formation'].values:

            try:
                columns = {'X': self.extent[0], 'Y': self.extent[2], 'Z': self.extent[4], 'formation': 'basement',
                           'order_series': n_series, 'formation_number': n_formation, 'series': self.series.columns[-1]}
            except AttributeError:
                columns = {'X': self.extent[0], 'Y': self.extent[2], 'Z': self.extent[4], 'formation': 'basement',
                           'order_series': n_series, 'formation_number': n_formation, 'series': 'Default series'}

            for key in columns:
                self.df.at[l, str(key)] = columns[key]

            self.order_table()
        # sef.add_interface(formation='basement', order_series=n_series, formation_number = n_formation)
        else:
            self.modify_interface((drop_basement.index[drop_basement])[0], formation='basement', order_series=n_series,
                                  formation_number=n_formation)

        self.order_table()

    def count_faults(self):
        """
        Read the string names of the formations to detect automatically the number of faults.
        """
        faults_series = []
        for i in self.df['series'].unique():
            try:
                if ('fault' in i or 'Fault' in i) and 'Default' not in i:
                    faults_series.append(i)
            except TypeError:
                pass
        return faults_series

    def set_default_interface(self, extent):
        # TODO this part is to update the formation class
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
        return self.interfaces["formation"].unique()

    def import_data_csv(self, path, **kwargs):
        """
        Method to import interfaces and orientations from csv. The format is the same as the export 3D model data of
        GeoModeller (check in the input_data data folder for an example).

        Args:
            path_i (str): path to the csv table
            path_o (str): path to the csv table
            **kwargs: kwargs of :func: `~pn.read_csv`

        Attributes:
            orientations(pandas.core.frame.DataFrame): Pandas data frame with the orientations data
            Interfaces(pandas.core.frame.DataFrame): Pandas data frame with the interfaces data
        """

        interfaces_read = pn.read_csv(path, **kwargs)
        assert set(['X', 'Y', 'Z', 'formation']).issubset(interfaces_read.columns), \
            "One or more columns do not match with the expected values " + str(interfaces_read.columns)

        c = np.array(self._columns_i_1)
        interfaces_read = interfaces_read.assign(**dict.fromkeys(c[~np.in1d(c, interfaces_read.columns)], False))
        self.set_interfaces(interfaces_read, append=True)
        # self.interfaces[interfaces_read.columns] = interfaces_read[interfaces_read.columns]
            # gagag
        self.update_df()

    def map_series(self, series):
        # Now we fill the column series in the interfaces and orientations tables with the correspondant series and
        # assigned number to the series
        series_df = series.df
        
        self.df["series"] = [(i == series_df).sum().idxmax() for i in self.df["formation"]]
        self.df["series"] = self.df["series"].astype('category')
        self.df["order_series"] = [(i == series_df).sum().as_matrix().argmax().astype(int) + 1
                                           for i in self.df["formation"]]
        # We sort the series altough is only important for the computation (we will do it again just before computing)
        # if series_distribution is not None:
        self.df.sort_values(by='order_series', inplace=True)

        self.df['series'].cat.set_categories(series_df.columns, inplace=True)
        
    def map_formations(self, formations):
        
        formations_df = formations.df
        formation_order = formations.df.index

        self.df['formation_number'] = self.df['formation'].map(formations_df.iloc[:, 1])
        self.df['formation_value'] = self.df['formation'].map(formations_df.iloc[:, 0])
        self.df['formation'].cat.set_categories(formation_order, inplace=True)

    def map_faults(self, faults):

        faults_df = faults.df
        self.df.loc[:, 'isFault'] = self.df['series'].isin(faults_df.index[self.faults['isFault']])

    def set_annotations(self):
        """
        Add a column in the Dataframes with latex names for each input_data paramenter.

        Returns:
            None
        """
        point_num = self.interfaces.groupby('formation_number').cumcount()
        point_l = [r'${\bf{x}}_{\alpha \,{\bf{' + str(f) + '}},' + str(p) + '}$'
                   for p, f in zip(point_num, self.interfaces['formation_number'])]

        self.df['annotations'] = point_l


class Orientations(object, Data):
    def __init__(self):
        
        self._columns_o_all = ['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', 'dip', 'azimuth', 'polarity',
                               'formation', 'series', 'X_std', 'Y_std', 'Z_std', 'dip_std', 'azimuth_std',
                               'order_series', 'formation_number']
        self._columns_o_1 = ['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', 'dip', 'azimuth', 'polarity', 'formation',
                             'series', 'formation_number', 'order_series', 'isFault']
        self._columns_o_num = ['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', 'dip', 'azimuth', 'polarity']
        self.df = pn.DataFrame(columns=self._columns_o_1)
        self.df[self._columns_o_num] = self.orientations[self._columns_o_num].astype(float)
        self.df.itype = 'orientations'
        
        self.calculate_gradient()
        self.df['isFault'] = self.df['isFault'].astype('bool')

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
            (self.df['G_x'] < 0).as_matrix() * (self.df['G_y'] >= 0).as_matrix()] += 360
        self.df['azimuth'][(self.df['G_y'] < 0).as_matrix()] += 180
        self.df['azimuth'][
            (self.df['G_x'] > 0).as_matrix() * (self.df['G_y'] == 0).as_matrix()] = 90
        self.df['azimuth'][
            (self.df['G_x'] < 0).as_matrix() * (self.df['G_y'] == 0).as_matrix()] = 270

    def create_orientation_from_interface(self, indices):
        selected_points = self.interfaces[['X', 'Y', 'Z']].loc[indices].values.T

        center, normal = self.plane_fit(selected_points)
        orientation = get_orientation(normal)

        return np.array([*center, *orientation, *normal])

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

    def import_data_csv(self, path, **kwargs):
        """
        Method to import interfaces and orientations from csv. The format is the same as the export 3D model data of
        GeoModeller (check in the input_data data folder for an example).

        Args:
            path_i (str): path to the csv table
            path_o (str): path to the csv table
            **kwargs: kwargs of :func: `~pn.read_csv`

        Attributes:
            orientations(pandas.core.frame.DataFrame): Pandas data frame with the orientations data
            Interfaces(pandas.core.frame.DataFrame): Pandas data frame with the interfaces data
        """

        orientations_read = pn.read_csv(path, **kwargs)

        assert set(['X', 'Y', 'Z', 'dip', 'azimuth', 'polarity', 'formation']).issubset(orientations_read.columns), \
            "One or more columns do not match with the expected values " + str(orientations_read.columns)

        self.orientations[orientations_read.columns] = orientations_read[orientations_read.columns]

        self.update_df()

    def map_series(self, series_df):
        # Now we fill the column series in the interfaces and orientations tables with the correspondant series and
        # assigned number to the series
        self.df["series"] = [(i == series_df).sum().idxmax() for i in self.df["formation"]]
        self.df["series"] = self.df["series"].astype('category')
        self.df["order_series"] = [(i == series_df).sum().as_matrix().argmax().astype(int) + 1
                                             for i in self.df["formation"]]

        # We sort the series altough is only important for the computation (we will do it again just before computing)
        # if series_distribution is not None:
        self.df.sort_values(by='order_series', inplace=True)

        self.df['series'].cat.set_categories(series_df.columns, inplace=True)

    def map_formations(self, formations):
        formations_df = formations.df
        formation_order = formations.df.index

        self.df['formation_number'] = self.df['formation'].map(formations_df.iloc[:, 1])

        self.df['formation_value'] = self.df['formation'].map(formations_df.iloc[:, 0])

        self.df['formation'].cat.set_categories(formation_order, inplace=True)
    
    def map_faults(self, faults):

        faults_df = faults.df
        self.df.loc[:, 'isFault'] = self.df['series'].isin(
            faults_df.index[faults_df['isFault']])

    def set_annotations(self):
        """
        Add a column in the Dataframes with latex names for each input_data paramenter.

        Returns:
            None
        """

        orientation_num = self.orientations.groupby('formation_number').cumcount()
        foli_l = [r'${\bf{x}}_{\beta \,{\bf{' + str(f) + '}},' + str(p) + '}$'
                   for p, f in zip(orientation_num, self.orientations['formation_number'])]

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


class Structure(object):
    def __init__(self):
        self.len_interfaces = None
        self.len_series_i = None
        self.len_series_o = None
        self.reference_position = None


class AdditionalData(object, Structure):
    def __init__(self):
        self.u_grade = None
        self.range_var = None
        self.c_o = None
        self.nugget_effect_gradient = None
        self.nugget_effect_scalar = None

        self.rescaling_factor = 1
        self.centers = np.array([])

    def default_range(self):
        pass

    def default_c_o(self):
        pass

    def get_kriging_parameters(self):
        pass

    @staticmethod
    def compute_rescaling_parameters(interfaces, orientations, rescaling_factor=None):
        # Check which axis is the largest
        max_coord = pn.concat(
            [orientations.df, interfaces.df]).max()[['X', 'Y', 'Z']]
        min_coord = pn.concat(
            [orientations.df, interfaces.df]).min()[['X', 'Y', 'Z']]

        # Compute rescalin factor if not given
        if not rescaling_factor:
            rescaling_factor = (2 * np.max(max_coord - min_coord))

        # Get the centers of every axis
        centers = ((max_coord + min_coord) / 2).astype(float)
        return rescaling_factor, centers

    def set_rescaling_parameters(self, interfaces, orientations, rescaling_factor=None):
        self.rescaling_factor, self.centers = self.compute_rescaling_parameters(interfaces, orientations,
                                                                                rescaling_factor)


class GeoPhysiscs(object):
    def __init__(self):
        self.gravity = None
        self.magnetics = None

    def create_geophy(self):
        pass

    def set_gravity_precomputations(self):
        pass


class Interpolator(object):
    def __init__(self, input_matrices: np.ndarray):
        self.verbose = None
        self.dtype = None
        self.output = None
        self.theano_optimizer = None
        self.is_lith=None
        self.is_fault=None

        import gempy.theano_graph as tg
        self.input_matrices = input_matrices
        self.theano_graph = tg
        self.theano_function = None

    def compile_th_fn(self, output, **kwargs):
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

        # This are the shared parameters and the compilation of the function. This will be hidden as well at some point
        input_data_T = self.interpolator.tg.input_parameters_list()

        print('Compiling theano function...')

        if output is 'geology':
            # then we compile we have to pass the number of formations that are faults!!
            th_fn = theano.function(input_data_T,
                                    self.interpolator.tg.compute_geological_model(),
                                    # mode=NanGuardMode(nan_is_error=True),
                                    on_unused_input='ignore',
                                    allow_input_downcast=False,
                                    profile=False)

        elif output is 'gravity':
            # then we compile we have to pass the number of formations that are faults!!
            th_fn = theano.function(input_data_T,
                                    self.interpolator.tg.compute_forward_gravity(),
                                    #  mode=NanGuardMode(nan_is_error=True),
                                    on_unused_input='ignore',
                                    allow_input_downcast=False,
                                    profile=False)

        elif output is 'gradients':

            gradients = kwargs.get('gradients', ['Gx', 'Gy', 'Gz'])
            self.interpolator.tg.gradients = gradients

            # then we compile we have to pass the number of formations that are faults!!
            th_fn = theano.function(input_data_T,
                                    self.interpolator.tg.compute_geological_model_gradient(self.geo_data_res.n_faults),
                                    #  mode=NanGuardMode(nan_is_error=True),
                                    on_unused_input='ignore',
                                    allow_input_downcast=False,
                                    profile=False)

        else:
            raise SyntaxError('The output given does not exist. Please use geology, gradients or gravity ')

        print('Compilation Done!')
        print('Level of Optimization: ', theano.config.optimizer)
        print('Device: ', theano.config.device)
        print('Precision: ', self.dtype)
        print('Number of faults: ', self.geo_data_res.n_faults)
        return th_fn


class Solution(object):
    def __init__(self):
        self.scalar_field_at_interfaces = 0
        self.scalar_field = np.array([])
        self.lith_block = None
        self.values_block = None
        self.gradient = None

