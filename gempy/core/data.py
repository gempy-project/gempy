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


class MetaData(object):
    def __init__(self, name_project='default_project'):
        import datetime
        now = datetime.datetime.now()
        self.date = now.strftime(" %Y-%m-%d %H:%M")

        if name_project is 'default_project':
            name_project += self.date

        self.name_project = name_project


class GridClass(object):
    """
    Class to generate grids to pass later on to a InputData class. This class is used to create points where to
    evaluate the geological model.

    Args:
        grid_type (str): type of premade grids provide by GemPy
        **kwargs: see args of the given grid type

    Attributes:
        grid_type (str): type of premade grids provide by GemPy
        resolution (list[int]): [x_min, x_max, y_min, y_max, z_min, z_max]
        extent (list[float]):  [nx, ny, nz]
        values (np.ndarray): coordinates where the model is going to be evaluated
        values_r (np.ndarray): rescaled coordinates where the model is going to be evaluated

    """

    def __init__(self, grid_type=None, **kwargs):

        self.grid_type = grid_type
        self.resolution = np.empty(3)
        self.extent = np.empty(6)
        self.values = np.empty((1,3))
        self.values_r = np.empty((1,3))

        if grid_type is 'regular_grid':
            self.set_regular_grid(**kwargs)
        elif grid_type is 'custom_grid':
            self.create_custom_grid(**kwargs)
        elif grid_type is None:
            pass
        else:
            warnings.warn('No valid grid_type. Grid is empty.')

    def __str__(self):
        return self.values

    def __repr__(self):
        return np.array_repr(self.values)

    def create_custom_grid(self, custom_grid, inplace=True):
        """
        Give the coordinates of an external generated grid

        Args:
            custom_grid (numpy.ndarray like): XYZ (in columns) of the desired coordinates
            inplace (bool): If True, do operation inplace and return None.
        Returns:
              numpy.ndarray: Unraveled 3D numpy array where every row correspond to the xyz coordinates of a regular grid
        """
        assert type(custom_grid) is np.ndarray and custom_grid.shape[1] is 3, 'The shape of new grid must be (n,3)' \
                                                                              ' where n is the number of points of ' \
                                                                              'the grid'

        if inplace is True:
            self.values = custom_grid
        else:
            return custom_grid

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
        """
        Set a regular grid into the values parameters for further computations
        Args:
             extent (list):  [x_min, x_max, y_min, y_max, z_min, z_max]
            resolution (list): [nx, ny, nz]
        """

        self.extent = extent
        self.resolution = resolution
        self.values = self.create_regular_grid_3d(extent, resolution)


class Series(object):
    """
    Series is a class that contains the relation between series/faults and each individual surface/layer. This can be
    illustrated in the sequential pile.

    Args:
        series_distribution (dict or :class:`pn.core.frame.DataFrames`): with the name of the serie as key and the
         name of the formations as values.
        order(Optional[list]): order of the series by default takes the dictionary keys which until python 3.6 are
            random. This is important to set the erosion relations between the different series

    Attributes:
        df (:class:`pn.core.frame.DataFrames`): Pandas data frame containing the series and the formations contained
            on them
        sequential_pile?

    """

    def __init__(self, series_distribution=None, order=None):
        if series_distribution is None:
            self.df = pn.DataFrame({"Default series": [None]}, dtype=str)
        else:
            self.set_series(series_distribution, order=order)

        self.sequential_pile = None

    def __repr__(self):
        return self.df.to_string()

    def _repr_html_(self):
        return self.df.to_html()

    def set_series(self, series_distribution, order=None, add_basement=False):
        """
        Method to define the different series of the project.

        Args:
            series_distribution (dict): with the name of the serie as key and the name of the formations as values.
            order(Optional[list]): order of the series by default takes the dictionary keys which until python 3.6 are
                random. This is important to set the erosion relations between the different series

        Returns:
            self.df: A pandas DataFrame with the series and formations relations
        """

        if isinstance(series_distribution, Interfaces):
            # TODO: Extract not only the formations but also the series from an interface object (in case it exist)
            self.df = pn.DataFrame({"Default series": series_distribution.df["formation"].unique().astype(list)},
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

        if add_basement and 'basement' not in self.df.iloc[:, -1].values:
            self.add_basement()

        return self.df

    def add_basement(self, name='basement'):
        """
        It is in place
        Args:
            name:

        Returns:

        """
        self.df.loc[self.df.shape[0], self.df.columns[-1]] = name


class Faults(object):
    """
    Class that encapsulate faulting related content. Mainly, which formations/surfaces are faults. The fault network
    ---i.e. which faults offset other faults---and fault types---finite vs infinite
        Args:
           series (Series): Series object

        Attributes:
           series (Series): Series object
           faults (:class:`pn.core.frame.DataFrames`): Pandas data frame containing the series and if they are faults or
            not (otherwise they are lithologies) and in case of being fault if is finite
           faults_relations (:class:`pn.core.frame.DataFrames`): Pandas data frame containing the offsetting relations
            between each fault and the rest of the series (either other faults or lithologies)
           n_faults (int): Number of faults in the object

    """
    def __init__(self, series: Series, series_fault=None, rel_matrix=None):

        self.series = series

        self.faults = pn.DataFrame(columns=['isFault', 'isFinite'])
        self.set_faults(series_fault=series_fault)

        self.faults_relations = pn.DataFrame(index=self.series.df.columns, columns=self.series.df.columns, dtype='bool')
        self.set_fault_relation(rel_matrix=rel_matrix)
        self.n_faults = 0

    def __repr__(self):
        return self.faults.to_string()

    def _repr_html_(self):
        return self.faults.to_html()

    def set_faults(self, series_fault=None):
        """
        Set a flag to the series that are faults.

        Args:
            series (Series): Series object
            series_fault(list or Interfaces): Name of the series which are faults
        """

        if series_fault is None:
            series_fault = self.count_faults(self.series.df.columns)

        self.faults = pn.DataFrame(index=self.series.df.columns, columns=['isFault'])
        self.faults['isFault'] = self.faults.index.isin(series_fault)
        self.n_faults = self.faults['isFault'].sum()
        return self.faults

    def check_fault_relations(self):
        # Method to check that only older faults offset newer ones?
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
        pass

    def set_fault_relation(self, rel_matrix=None):
        """
        Method to set the faults that offset a given sequence and therefore also another fault

        Args:
            rel_matrix (numpy.array): 2D Boolean array with the logic. Rows affect (offset) columns
        """
        # TODO: Change the fault relation automatically every time we add a fault
        if not rel_matrix:
            rel_matrix = np.zeros((self.series.df.columns.shape[0],
                                   self.series.df.columns.shape[0]))
        else:
            assert type(rel_matrix) is np.ndarray, 'rel_matrix muxt be a 2D numpy array'
        self.faults_relations = pn.DataFrame(rel_matrix, index=self.series.df.columns,
                                             columns=self.series.df.columns, dtype='bool')

        return self.faults_relations

    @staticmethod
    def count_faults(list_of_names):
        """
        Read the string names of the formations to detect automatically the number of faults if the name
        fault is on the name.
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
    """
    Class that contains the formations of the model and the values of each of them.

    Args:
        values_array (np.ndarray): 2D array with the values of each formation
        values_names (list or np.ndarray): list containing the names of each properties
        formation_names (list or np.ndarray): list contatinig the names of the formations


    Attributes:
        df (:class:`pn.core.frame.DataFrames`): Pandas data frame containing the formations names and the value
         used for each voxel in the final model and the lithological order
        formation_names (list[str]): List in order of the formations
    """
    def __init__(self, values_array=None, values_names=np.empty(0), formation_names=np.empty(0)):
        self.df = pn.DataFrame(columns=['formation', 'id', 'isBasement'])
        self.df['isBasement'] = self.df['isBasement'].astype(bool)
        self.df["formation"] = self.df["formation"].astype('category')
        self.df["id"] = self.df["id"].astype('int32')

        self.formations_names = np.empty(0)
        self._formation_values_set = False
        if values_array is not None:
            self.set_formations(values_array=values_array, values_names=values_names, formation_names=formation_names)

    def __repr__(self):
        return self.df.to_string()

    def _repr_html_(self):
        return self.df.to_html()

    def set_formation_names(self, list_names):
        """
        Method to set the name of the formations in order
        Args:
            list_names (list[str]):

        Returns:
            None
        """
        if type(list_names) is list or type(list_names) is np.ndarray:
            self.formations_names = np.asarray(list_names)
        elif isinstance(list_names, Interfaces):
            self.formations_names = np.asarray(list_names.df['formation'].cat.categories)
        else:
            raise AttributeError('list_names must be either array_like type or Interfaces')

        self.map_formation_names_to_df()

    def set_formation_order(self, list_names):
        self.set_formation_names(list_names)

    def map_formation_names_to_df(self):
        if self.df['formation'].shape[0] == self.formations_names.shape[0]:
            self.df['formation'] = self.formations_names

        elif self.df['formation'].shape[0] < self.formations_names.shape[0]:
            n_to_append = self.df['formation'].shape[0] - self.formations_names.shape[0]
            for i in range(n_to_append):
                self.formations_names = np.append(self.formations_names, 'default_formation_' + str(i))

            if self.df['formation'].shape[0] is not 0:
                warnings.warn('Length of formation_names does not match number of formations')
            self.df['formation'] = self.formations_names

        elif self.df['formation'].shape[0] > self.formations_names.shape[0]:
            warnings.warn('Length of formation_names does not match number of formations')
            self.df['formation'] = self.formations_names[:self.df.shape[0]]

        self.df['formation'] = self.df['formation'].astype('category')

    def map_formations_from_series(self, series):
        """
        Update the a Formations object with a Series object
        Args:
            series (Series):

        Returns:
            None
        """

        # TODO review
        # Addind the formations of the new series to the formations df
        new_formations = series.df.values.reshape(1, -1)
        # Dropping nans
        new_formations = new_formations[~pn.isna(new_formations)]
        self.set_formation_names(new_formations)

    def set_basement(self, basement_formation=None):

        self.df['isBasement'] = self.df['formation'] == basement_formation
        assert self.df['isBasement'].values.astype(bool).sum() <= 1, 'Only one formation can be basement'

    def add_basement(self, name=None, inplace=True):
       # self.set_basement()
        self.df['isBasement'].fillna(False, inplace=True)
        assert self.df['isBasement'].values.astype(bool).sum() < 1, 'Only one formation can be basement'
        if name is None:
            name = 'basement'

        new_df = pn.concat([self.df,
                            pn.DataFrame(data=np.array([name, True, 9999]).reshape(1, -1),
                                         columns=['formation', 'isBasement', 'id'])],
                            sort=False, ignore_index=True
                             )

        if inplace is True:
            self.df = self.set_id(new_df)
            self.set_dtypes()
        else:
            return new_df

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
        """
        Set id of the layers (1 based)
        Args:
            df:

        Returns:

        """
        df['id'] = df.index + 1
        return df

    def set_dtypes(self):
        self.df['isBasement'] = self.df['isBasement'].astype(bool)
        self.df["formation"] = self.df["formation"].astype('category')

    def set_formations(self, values_array, values_names=np.empty(0), formation_names=None):
        """
        Set the df containing the values of each formation for the posterior evaluation (e.g. densities, susceptibility)
        Args:
            values_array (np.ndarray): 2D array with the values of each formation
            values_names (list or np.ndarray): list containing the names of each properties
            formation_names (list or np.ndarray): list contatinig the names of the formations

        Returns:

            Dataframe
        """
        self.df = pn.DataFrame(columns=['formation', 'isBasement', 'id'])
        self.df['isBasement'] = self.df['isBasement'].astype(bool)
        self.df["formation"] = self.df["formation"].astype('category')

        values_names = np.asarray(values_names)
        if type(values_array) is np.ndarray:
            if values_names.size is 0:
                for i in range(values_array.shape[1]):
                    values_names = np.append(values_names, 'value_' + str(i))
            vals_df = pn.DataFrame(values_array, columns=values_names)
            self.set_formation_names(values_names)
        elif isinstance(values_array, pn.core.frame.DataFrame):
            vals_df = values_array

        else:
            raise AttributeError('values_array must be either numpy array or pandas df')

        if formation_names:
            self.set_formation_names(formation_names)

        f_df = pn.concat([self.df, vals_df], sort=False, axis=1, verify_integrity=True, ignore_index=False)

        self.df = self.set_id(f_df)
        self.map_formation_names_to_df()
        self.df['isBasement'].fillna(False, inplace=True)
        return self.df


class Data(object):
    """
    Parent class of the objects which contatin the input parameters: interfaces and orientations. This class contain
    the common methods for both types of data sets.
    """
    def __init__(self):
        self.df = pn.DataFrame()

    def __repr__(self):
        return self.df.to_string()

    def _repr_html_(self):
        return self.df.to_html()

    @staticmethod
    def read_data(filepath, **kwargs):
        """
        Read method of pandas for different types of tabular data
        Args:
            filepath(str):
            **kwargs:  See pandas read_table

        Returns:
             pandas.core.frame.DataFrame: Data frame with the raw data
        """
        if 'sep' not in kwargs:
            kwargs['sep'] = ','

        table = pn.read_table(filepath, **kwargs)

        return table

    def sort_table(self):
        """
        First we sort the dataframes by the series age. Then we set a unique number for every formation and resort
        the formations. All inplace
        """

        # We order the pandas table by formation (also by series in case something weird happened)
        self.df.sort_values(by=['order_series', 'id'],
                            ascending=True, kind='mergesort',
                            inplace=True)

    def map_series_to_data(self, series):
        """
        Method to map a series object to a data object merging on formations
        Args:
            series (Series):

        Returns:
            self.df
        """
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

    def map_formations_to_data(self, formations, sort=True, inplace=True):
        """
        Method to map formation ids and values to a data object on formations
        Args:
            formations (Formations):
            sort (bool): If true the table will be sorted in the formation order after mapping
            inplace (bool): If True, do operation inplace and return None.

        Returns:
            pandas.core.frame.DataFrame: Data frame with the raw data
        """
        i_df = self.df.drop('id', axis=1)
        new_df = pn.merge(i_df, formations.df, on=['formation'], how='left')

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

    def map_faults_to_data(self, faults, inplace=True):
        """
        Method to map a faults object into the data object on formations. Either if the formation is fault or not
        Args:
            faults (Faults):
            inplace (bool): If True, do operation inplace and return None.

        Returns:
            pandas.core.frame.DataFrame: Data frame with the raw data

        """
        faults_df = faults.faults

        if any(self.df['series'].isna()):
            warnings.warn('Some points do not have series/fault')
        if inplace:

            self.df.loc[:, 'isFault'] = self.df['series'].isin(faults_df.index[faults_df['isFault']])
        else:
            return self.df['series'].isin(faults_df.index[faults_df['isFault']])

    def set_dypes(self):
        """
        Method to set each column of the dataframe to the right data type. Inplace
        Returns:

        """
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
        # TODO DEP?
        self.df[['Xr', 'Yr', 'Zr']] = (self.df[['X', 'Y', 'Z']] - centers) / rescaling_factor + 0.5001


class Interfaces(Data):
    """
    Data child with specific methods to manipulate interface data. It is initialize without arguments to give
    flexibility to the origin of the data

    Attributes:
          df (:class:`pn.core.frame.DataFrames`): Pandas data frame containing the necessary information respect
            the interface points of the model
    """
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

    def read_interfaces(self, filepath, debug=False, inplace=False, append=False, **kwargs):
        """
        Read tabular using pandas tools and if inplace set it properly to the Interace object
        Args:
            filepath:
            debug:
            inplace:
            append:
            **kwargs:

        Returns:

        """
        if 'sep' not in kwargs:
            kwargs['sep'] = ','

        table = pn.read_table(filepath, **kwargs)

        if debug is True:
            print('Debugging activated. Changes won\'t be saved.')
            return table
        else:
            assert set(['X', 'Y', 'Z', 'formation']).issubset(table.columns), \
                "One or more columns do not match with the expected values " + str(table.columns)

            if inplace:
                c = np.array(self._columns_i_1)
                interfaces_read = table.assign(**dict.fromkeys(c[~np.in1d(c, table.columns)], np.nan))
                self.set_interfaces(interfaces_read, append=append)
                self.df['formation'] = self.df['formation'].astype('category')

            else:
                return table

    def set_interfaces(self, interf_dataframe, append=False):
        """
        Method to change or append a Dataframe to interfaces in place. A equivalent Pandas Dataframe with
        ['X', 'Y', 'Z', 'formation'] has to be passed.

        Args:
            interf_dataframe: pandas.core.frame.DataFrame with the data
            append: Bool: if you want to append the new data frame or substitute it
        """
        assert set(self._columns_i_1).issubset(interf_dataframe.columns), \
            "One or more columns do not match with the expected values " + str(self._columns_i_1)

        interf_dataframe[self._columns_i_num] = interf_dataframe[self._columns_i_num].astype(float, copy=True)
        try:
            interf_dataframe[['formation_number', 'order_series']] = interf_dataframe[
                ['formation_number', 'order_series']].astype(int, copy=True)
        except ValueError:
            warnings.warn('No formation_number or order_series in the file')
            pass
        interf_dataframe['formation'] = interf_dataframe['formation'].astype('category', copy=True)
        interf_dataframe['series'] = interf_dataframe['series'].astype('category', copy=True)

        if append:
            self.df = self.df.append(interf_dataframe)

        else:

            self.df = interf_dataframe[self._columns_i_1]

        self.df = self.df[~self.df[['X', 'Y', 'Z']].isna().any(1)]

        if not self.df.index.is_unique:
            self.df.reset_index(drop=True, inplace=True)

    def set_basement(self):
        # TODO in principle it is not necessary anymore
        pass

    def set_default_interface(self, formation: Formations, grid: GridClass):
        """
        Set a default point at the middle of the extent area to be able to start making the model
        Args:
            formation:
            grid:

        Returns:

        """
        formation_name = formation.formations_names[0]
        extent = grid.extent

        self.set_interfaces(pn.DataFrame({'X': [(extent[1] - extent[0]) / 2],
                                          'Y': [(extent[3] - extent[2]) / 2],
                                          'Z': [(extent[4] - extent[5]) / 2],
                                          'formation': [formation_name], 'order_series': [0],
                                          'formation_number': [1], 'series': ['Default series'],
                                          'isFault': False}))

    def get_formations(self):
        """
        Returns:
             pandas.core.frame.DataFrame: Returns a list of formations

        """
        return self.df["formation"].unique()

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
    """
    Data child with specific methods to manipulate orientation data. It is initialize without arguments to give
    flexibility to the origin of the data

    Attributes:
        df (:class:`pn.core.frame.DataFrames`): Pandas data frame containing the necessary information respect
         the orientations of the model
    """
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

    def calculate_gradient(self):
        """
        Calculate the gradient vector of module 1 given dip and azimuth to be able to plot the orientations
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

    def create_orientation_from_interface(interfaces: Interfaces, indices):
        """
        Create and set orientations from at least 3 points df
        Args:
            indices_array (array-like): 1D or 2D array with the pandas indices of the
              :attr:`gempy.data_management.InputData.interfaces`. If 2D every row of the 2D matrix will be used to create an
              orientation
        """
        selected_points = interfaces.df[['X', 'Y', 'Z']].loc[indices].values.T

        center, normal = plane_fit(selected_points)
        orientation = get_orientation(normal)

        return np.array([*center, *orientation, *normal])

    def set_default_orientation(self, grid: GridClass):
        """
        Set a default point at the middle of the extent area to be able to start making the model
        """

        extent = grid.extent

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
        """
        Read tabular using pandas tools and if inplace set it properly to the Orientations object
        Args:
            filepath:
            debug:
            inplace:
            append:
            **kwargs:

        Returns:

        """
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
            self.df = foliat_Dataframe[self._columns_o_1]

        self.calculate_gradient()

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
    """
    Auxiliary class to rescale the coordinates between 0 and 1 to increase stability

    Attributes:
        interfaces (Interfaces)
        orientaions (Orientations)
        grid (GridClass)
        rescaling_factor (float): value which divide all coordinates
        centers (list[float]): New center of the coordinates after shifting

    Args:
        interfaces (Interfaces)
        orientaions (Orientations)
        grid (GridClass)
        rescaling_factor (float): value which divide all coordinates
        centers (list[float]): New center of the coordinates after shifting
    """
    def __init__(self, interfaces: Interfaces, orientations: Orientations, grid: GridClass,
                 rescaling_factor=None, centers=None):

        self.interfaces = interfaces
        self.orientations = orientations
        self.grid = grid
        self.centers = centers
        self.rescaling_factor = rescaling_factor

        self.rescale_data(rescaling_factor=rescaling_factor, centers=centers)

    def rescale_data(self, rescaling_factor=None, centers=None, inplace=True):

        max_coord, min_coord = self.max_min_coord(self.interfaces, self.orientations)
        if not rescaling_factor:
            self.rescaling_factor = self.compute_rescaling_factor(self.interfaces, self.orientations,
                                                                  max_coord, min_coord)
        else:
            self.rescaling_factor = rescaling_factor
        if not centers:
            self.centers = self.compute_data_center(self.interfaces, self.orientations,
                                                    max_coord, min_coord)
        else:
            self.centers = centers

        self.set_rescaled_interfaces()
        self.set_rescaled_orientations()
        self.set_rescaled_grid()

    def get_rescaled_data(self):
        """
        Get the rescaled coordinates
        Returns:

        """
        # TODO return an image of the interface and orientations df with the X_r.. columns
        pass

    @staticmethod
    def max_min_coord(interfaces=None, orientations=None):
        """
        Find the maximum and minimum location of any input data in each cartesian coordinate
        Args:
            interfaces (Interfaces):
            orientations (Orientations):

        Returns:
            tuple: max[XYZ], min[XYZ]
        """
        if interfaces is None:
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
                            max_coord=None, min_coord=None, inplace=True):
        """
        Calculate the center of the data once it is shifted between 0 and 1
        Args:
            interfaces:
            orientations:
            max_coord:
            min_coord:

        Returns:

        """

        if max_coord is None or min_coord is None:
            max_coord, min_coord = self.max_min_coord(interfaces, orientations)

        # Get the centers of every axis
        centers = ((max_coord + min_coord) / 2).astype(float)
        if inplace is True:
            self.centers = centers
        return centers

    def compute_rescaling_factor(self, interfaces=None, orientations=None,
                                 max_coord=None, min_coord=None, inplace=True):
        """
        Calculate the rescaling factor of the data to keep all coordinates between 0 and 1
        Args:
            interfaces:
            orientations:
            max_coord:
            min_coord:

        Returns:

        """

        if max_coord is None or min_coord is None:
            max_coord, min_coord = self.max_min_coord(interfaces, orientations)
        rescaling_factor_val = (2 * np.max(max_coord - min_coord))
        if inplace is True:
            self.rescaling_factor = rescaling_factor_val
        return rescaling_factor_val

    @staticmethod
    def rescale_interfaces(interfaces, rescaling_factor, centers):
        # Change the coordinates of interfaces
        new_coord_interfaces = (interfaces.df[['X', 'Y', 'Z']] -
                                centers) / rescaling_factor + 0.5001
        return new_coord_interfaces

    def set_rescaled_interfaces(self, inplace=True):
        """
        Set the rescaled coordinates into the interfaces df
        Returns:

        """

        if inplace is True:
            self.interfaces.df[['X_r', 'Y_r', 'Z_r']] = self.rescale_interfaces(self.interfaces, self.rescaling_factor,
                                                                                self.centers)
        else:
            return self.interfaces

    @staticmethod
    def rescale_orientations(orientations, rescaling_factor, centers):
        # Change the coordinates of orientations
        new_coord_orientations = (orientations.df[['X', 'Y', 'Z']] -
                                  centers) / rescaling_factor + 0.5001
        return new_coord_orientations

    def set_rescaled_orientations(self, inplace=True):
        """
        Set the rescaled coordinates into the orientations df
        Returns:

        """
        self.orientations.df[['X_r', 'Y_r', 'Z_r']] = self.rescale_orientations(self.orientations, self.rescaling_factor,
                                                                                self.centers)

        if inplace is False:
            return self.orientations

    @staticmethod
    def rescale_grid(grid, rescaling_factor, centers):
        new_grid_extent = (grid.extent - np.repeat(centers, 2)) / rescaling_factor + 0.5001
        new_grid_values = (grid.values - centers.values) / rescaling_factor + 0.5001
        return new_grid_extent, new_grid_values

    def set_rescaled_grid(self):
        """
             Set the rescaled coordinates and extent into a grid object
        """
        self.grid.extent_r, self.grid.values_r = self.rescale_grid(self.grid, self.rescaling_factor, self.centers)


class Structure(object):
    """
    The structure class analyse the different lenths of subset in the interface and orientations df to pass them to
    the theano function

    Attributes:
        len_formations_i (list): length of each formation/fault in interfaces
        len_series_i (list) : length of each series in interfaces
        len_series_o (list) : length of each series in orientations
        nfs (list): number of formations per series
        ref_position (list): location of the first point of each formation/fault in interface

    Args:
        interfaces (Interfaces)
        orientations (Orientations)
    """
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
        self.len_series_o = orientations.df['order_series'].value_counts(sort=False).values
        return self.len_series_o

    def set_ref_position(self):
        self.ref_position = np.insert(self.len_formations_i[:-1], 0, 0).cumsum()
        return self.ref_position

    def set_number_of_formations_per_series(self, interfaces):
        self.nfs = interfaces.df.groupby('order_series').formation.nunique().values.cumsum()
        return self.nfs


class AdditionalData(Structure, RescaledData):
    #TODO IMP: split this class in each of the 3 types of extra data since not all of them are input
    """
    Class that encapsulate all auxiliary parameters and options: Structure, Options, Kriging
    parameters and Rescaling factors

    Attributes:


    Args:
        interfaces (Interfaces)
        orientaions (Orientations)
        grid (GridClass)
        faults (Faults)
        formations (Formations)
        rescaling (rescaling)

    """
    def __init__(self, interfaces: Interfaces, orientations: Orientations, grid: GridClass,
                 faults: Faults, formations:Formations, rescaling: RescaledData):
        # TODO: probably not all the attributes need to be present until I do a check before computing the thing.
        # TODO IMP: Right now there are two copies for most of the paramenters. One in self and one in the df
        #           this may lead to important confusion and bugs

        self.interfaces = interfaces
        self.orientations = orientations
        self.faults = faults
        self.formations = formations
        self.grid = grid
        self.rescaled_data = rescaling

        super().__init__(interfaces, orientations)

        self.n_faults = faults.n_faults
        self.n_formations = formations.df.shape[0]

        self.range_var = self.default_range(grid.extent)
        self.c_o = self.default_c_o()

        self.n_universal_eq = self.set_u_grade(None)

        self.nugget_effect_gradient = 0.01
        self.nugget_effect_scalar = 1e-6

        self.kriging_data = pn.DataFrame([self.range_var, self.c_o, self.n_universal_eq,
                                          self.nugget_effect_gradient, self.nugget_effect_scalar],
                                         columns=['values'],
                                         index=['range', '$C_o$', 'drift equations',
                                                'nugget grad', 'nugget scalar'])


        self.options = pn.DataFrame(columns=['values'],
                                    index=['dtype', 'output', 'theano_optimizer', 'device', 'verbosity'])
        self.default_options()

        self.structure_data = pn.DataFrame([self.is_lith(), self.is_fault(),
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

    def update_rescaling_data(self):
        self.rescaling_data.at['rescaling factor', 'values'] = self.rescaled_data.rescaling_factor
        self.rescaling_data.at['centers', 'values'] = self.rescaled_data.centers

    def update_default_kriging(self):

        self.range_var = self.default_range(self.grid.extent)
        self.c_o = self.default_c_o()

        self.n_universal_eq = self.set_u_grade(None)

        self.nugget_effect_gradient = 0.01
        self.nugget_effect_scalar = 1e-6

        self.kriging_data = pn.DataFrame([self.range_var, self.c_o, self.n_universal_eq,
                                          self.nugget_effect_gradient, self.nugget_effect_scalar],
                                         columns=['values'],
                                         index=['range', '$C_o$', 'drift equations',
                                                'nugget grad', 'nugget scalar'])

    def update_structure(self):

        super().__init__(self.interfaces, self.orientations)

        self.structure_data = pn.DataFrame([self.is_lith(), self.is_fault(),
                                            self.faults.n_faults,  self.formations.df.shape[0], self.nfs,
                                            self.len_formations_i, self.len_series_i,
                                            self.len_series_o],
                                           columns=['values'],
                                           index=['isFault', 'isLith',
                                                  'number faults', 'number formations', 'number formations per series',
                                                  'len formations interfaces', 'len series interfaces',
                                                  'len series orientations'])

    def get_additional_data(self):
        concat_ = pn.concat([self.structure_data, self.options, self.kriging_data, self.rescaling_data],
                            keys=['Structure', 'Options', 'Kringing', 'Rescaling'])
        return concat_

    def is_lith(self):
        """
        Check if there is lithologies in the data and/or faults
        Returns:
            list(bool)
        """
        is_lith = False
        if self.formations.df.shape[0] - 1 > self.n_faults:
            is_lith = True
        return is_lith

    def is_fault(self):
        is_fault = False
        if self.faults.n_faults != 0:
            is_fault = True
        return is_fault

    def default_options(self):
        """
        Set default options.
        Returns:

        """
        self.options.at['dtype'] = 'float64'
        self.options.at['output'] = 'geology'
        self.options.at['theano_optimizer'] = 'fast_compile'
        self.options.at['device'] = 'cpu'

    @staticmethod
    def default_range(extent):
        """
        Set default kriging range
        Args:
            extent:

        Returns:

        """
        try:
            range_var = np.sqrt(
                (extent[0] - extent[1]) ** 2 +
                (extent[2] - extent[3]) ** 2 +
                (extent[4] - extent[5]) ** 2)
        except TypeError:
            range_var = np.nan

        return range_var

    def default_c_o(self):
        """
        Set default covariance at 0
        Returns:

        """
        c_o = self.range_var ** 2 / 14 / 3
        return c_o

    def set_u_grade(self, u_grade):
        """
        Set default universal grade. Transform polinomial grades to number of equations
        Args:
            **kwargs:

        Returns:

        """
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
        return self.kriging_data


class GeoPhysiscs(object):
    def __init__(self):
        self.gravity = None
        self.magnetics = None

    def create_geophy(self):
        pass

    def set_gravity_precomputations(self):
        pass


class Solution(object):

    def __init__(self, additional_data: AdditionalData=None, formations: Formations=None, values=None):

        self.additional_data = additional_data
        self.formations = formations

        if values is None:

            self.scalar_field_at_interfaces = 0
            self.scalar_field_lith = np.array([])
            self.scalar_field_faults = np.array([])

            self.lith_block = None
            self.fault_blocks = None
            self.values_block = None
            self.gradient = None
        else:
            self.set_values(values)

    def set_values(self, values):

        lith = values[0]
        faults = values[1]
        self.scalar_field_at_interfaces = values[2]

        self.scalar_field_lith = lith[0]
        self.lith_block = lith[1]

        if self.additional_data.options.loc['output', 'values'] is 'gradients':
            self.values_block = lith[2:-3]
            self.gradient = lith[-3:]
        else:
            self.values_block = lith[2:]

        self.scalar_field_faults = faults[::2]
        self.fault_blocks = faults[1::2]

    def get_surface(self, ):
        pass

    def get_surface_old(potential_block, interp_data, pot_int, n_formation, step_size, original_scale):
        """
        Get an individual surface

        """

        assert n_formation >= 0, 'Number of the formation has to be positive'

        # In case the values are separated by series I put all in a vector
        pot_int = interp_data.potential_at_interfaces.sum(axis=0)

        from skimage import measure

        if not potential_block.max() > pot_int[n_formation]:
            pot_int[n_formation] = potential_block.max()
            print('Potential field of the surface is outside the block. Probably is due to float errors')

        if not potential_block.min() < pot_int[n_formation]:
            pot_int[n_formation] = potential_block.min()
            print('Potential field of the surface is outside the block. Probably is due to float errors')

        vertices_p, simplices_p, normals, values = measure.marching_cubes_lewiner(
            potential_block.reshape(interp_data.geo_data_res.resolution[0],
                                    interp_data.geo_data_res.resolution[1],
                                    interp_data.geo_data_res.resolution[2]),
            pot_int[n_formation],
            step_size=step_size,
            spacing=((interp_data.geo_data_res.extent[1] - interp_data.geo_data_res.extent[0]) /
                     interp_data.geo_data_res.resolution[0],
                     (interp_data.geo_data_res.extent[3] - interp_data.geo_data_res.extent[2]) /
                     interp_data.geo_data_res.resolution[1],
                     (interp_data.geo_data_res.extent[5] - interp_data.geo_data_res.extent[4]) /
                     interp_data.geo_data_res.resolution[2]))

        if original_scale:
            vertices_p = interp_data.rescaling_factor * vertices_p + _np.array([interp_data._geo_data.extent[0],
                                                                                interp_data._geo_data.extent[2],
                                                                                interp_data._geo_data.extent[
                                                                                    4]]).reshape(1, 3)
        else:
            vertices_p += np.array([interp_data.geo_data_res.extent[0],
                                     interp_data.geo_data_res.extent[2],
                                     interp_data.geo_data_res.extent[4]]).reshape(1, 3)
        return vertices_p, simplices_p

    def get_surfaces(self, interp_data, potential_lith=None, potential_fault=None, n_formation='all',
                     step_size=1, original_scale=True):
        """
        Compute vertices and simplices of the interfaces for its vtk visualization and further
        analysis

        Args:
            interp_data (:class:`gempy.data_management.InputData`)
            potential_lith (ndarray): 1D numpy array with the solution of the computation of the model
             containing the scalar field of potentials (second row of lith solution)
            potential_fault (ndarray): 1D numpy array with the solution of the computation of the model
             containing the scalar field of the faults (every second row of fault solution)
            n_formation (int or 'all'): Positive integer with the number of the formation of which the surface is returned.
             use method get_formation_number() to get a dictionary back with the values
            step_size (int): resolution of the method. This is every how many voxels the marching cube method is applied
            original_scale (bool): choosing if the coordinates of the vertices are given in the original or the rescaled
             coordinates

        Returns:
            vertices, simpleces
        """
        # try:
        #     getattr(interp_data, 'potential_at_interfaces')
        # except AttributeError:
        #     raise AttributeError('You need to compute the model first')

        vertices = []
        simplices = []

        n_formations = self.formations.df['id']
        n_faults = self.additional_data.structure_data.loc['number faults', 'values']

        potential_fault = self.scalar_field_faults
        potential_lith = self.scalar_field_lith
        # Looping the scalar fields of the faults

        if potential_fault is not None:
            assert len(np.atleast_2d(
                potential_fault)) == n_faults, 'You need to pass a potential field per fault'

            pot_int = self.scalar_field_at_interfaces[:n_faults + 1]
            for n in n_formations[:n_faults]:
                #TODO
                v, s = self.get_surface(np.atleast_2d(potential_fault)[n], interp_data, pot_int, n,
                                        step_size=step_size, original_scale=original_scale)
                vertices.append(v)
                simplices.append(s)

        # Looping the scalar fields of the lithologies
        if potential_lith is not None:
            pot_int = interp_data.potential_at_interfaces[interp_data.geo_data_res.n_faults:]

            # Compute the vertices of the lithologies
            if n_formation == 'all':

                for n in n_formations[
                         interp_data.geo_data_res.n_faults:]:  # interp_data.geo_data_res.interfaces['formation_number'][~interp_data.geo_data_res.interfaces['isFault']].unique():
                    # if n == 0:
                    #     continue
                    # else:
                    v, s = get_surface(potential_lith, interp_data, pot_int, n,
                                       step_size=step_size, original_scale=original_scale)
                    vertices.append(v)
                    simplices.append(s)
            else:
                vertices, simplices = get_surface(potential_lith, interp_data, pot_int, n_formation,
                                                  step_size=step_size, original_scale=original_scale)

        return vertices, simplices

class Interpolator(object):
    # TODO assert passed data is rescaled
    def __init__(self, interfaces: Interfaces, orientations: Orientations, grid: GridClass,
                 formations: Formations, faults: Faults, additional_data: AdditionalData, **kwargs):

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
        self.theano_graph.a_T.set_value(np.cast[self.dtype](self.additional_data.kriging_data.loc['range', 'values'] /
                                                            self.additional_data.rescaling_data.loc['rescaling factor', 'values']))
        # Covariance at 0
        self.theano_graph.c_o_T.set_value(np.cast[self.dtype](self.additional_data.kriging_data.loc['$C_o$', 'values'] /
                                                              self.additional_data.rescaling_data.loc[
                                                                  'rescaling factor', 'values']
                                                              ))
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
        self.theano_graph.n_formation.set_value(self.formations.df['id'].values.astype('int32'))
        # Final values the lith block takes
        try:
            self.theano_graph.formation_values.set_value(self.formations.df['value_0'].values)
        except KeyError:
            self.theano_graph.formation_values.set_value(self.formations.df['id'].values.astype(self.dtype))
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

    def compile_th_fn(self, output=None, inplace=True, **kwargs):
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
        if output is None:
            output = self.additional_data.options.loc['output', 'values']

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



