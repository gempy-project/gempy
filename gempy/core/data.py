import sys
from os import path

# This is for sphenix to find the packages
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import os
import numpy as np
import pandas as pn
from typing import Union
import warnings
from gempy.core.checkers import check_for_nans
from gempy.utils.meta import _setdoc
from gempy.plot.sequential_pile import StratigraphicPile

pn.options.mode.chained_assignment = None


class MetaData(object):
    """
    Set of attibutes and methods that are not related directly with the geological model but more with the project

    Args:
        project_name (str): Name of the project. This is use as default value for some I/O actions

    Attributes:
        date (str): Time of the creations of the project
        project_name (str): Name of the project. This is use as default value for some I/O actions
    """

    def __init__(self, project_name='default_project'):
        import datetime
        now = datetime.datetime.now()
        self.date = now.strftime(" %Y-%m-%d %H:%M")

        if project_name is 'default_project':
            project_name += self.date

        self.project_name = project_name


class Grid(object):
    """
    Class to generate grids. This class is used to create points where to
    evaluate the geological model. So far only regular grids and custom_grids are implemented.

    Args:
        grid_type (str): type of pre-made grids provide by GemPy
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
        self.extent = np.empty(6, dtype='float64')
        self.values = np.empty((1, 3))
        self.values_r = np.empty((1, 3))
        if grid_type is 'regular_grid':
            self.set_regular_grid(**kwargs)
        elif grid_type is 'custom_grid':
            self.set_custom_grid(**kwargs)
        elif grid_type is None:
            pass
        else:
            warnings.warn('No valid grid_type. Grid is empty.')

    def __str__(self):
        return 'Grid Object. Values: \n' + np.array2string(self.values)

    def __repr__(self):
        return 'Grid Object. Values: \n' + np.array_repr(self.values)

    def set_custom_grid(self, custom_grid: np.ndarray):
        """
        Give the coordinates of an external generated grid

        Args:
            custom_grid (numpy.ndarray like): XYZ (in columns) of the desired coordinates

        Returns:
              numpy.ndarray: Unraveled 3D numpy array where every row correspond to the xyz coordinates of a regular
               grid
        """
        custom_grid = np.atleast_2d(custom_grid)
        assert type(custom_grid) is np.ndarray and custom_grid.shape[1] is 3, 'The shape of new grid must be (n,3)' \
                                                                              ' where n is the number of points of ' \
                                                                              'the grid'

        self.values = custom_grid

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

        dx, dy, dz = (extent[1] - extent[0]) / resolution[0], (extent[3] - extent[2]) / resolution[0], \
                     (extent[5] - extent[4]) / resolution[0]

        g = np.meshgrid(
            np.linspace(extent[0] + dx / 2, extent[1] - dx / 2, resolution[0], dtype="float64"),
            np.linspace(extent[2] + dy / 2, extent[3] - dy / 2, resolution[1], dtype="float64"),
            np.linspace(extent[4] + dz / 2, extent[5] - dz / 2, resolution[2], dtype="float64"), indexing="ij"
        )

        values = np.vstack(tuple(map(np.ravel, g))).T.astype("float64")
        return values

    def set_regular_grid(self, extent, resolution):
        """
        Set a regular grid into the values parameters for further computations
        Args:
             extent (list):  [x_min, x_max, y_min, y_max, z_min, z_max]
            resolution (list): [nx, ny, nz]
        """

        self.extent = np.asarray(extent, dtype='float64')
        self.resolution = np.asarray(resolution)
        self.values = self.create_regular_grid_3d(extent, resolution)


class Series(object):
    """
    Series is a class that contains the relation between series/df and each individual surface/layer. This can be
    illustrated in the sequential pile.

    Args:
        series_distribution (dict or :class:`pn.core.frame.DataFrames`): with the name of the serie as key and the
         name of the surfaces as values.
        order(Optional[list]): order of the series by default takes the dictionary keys which until python 3.6 are
            random. This is important to set the erosion relations between the different series

    Attributes:
        categories_df (:class:`pn.core.frame.DataFrames`): Pandas data frame containing the series and the surfaces contained
            on them
        sequential_pile?

    """

    def __init__(self, faults, series_order=None, ):

        self.faults = faults

        if series_order is None:
            series_order = ['Default series']

        self.df = pn.DataFrame(np.array([[1, np.nan]]), index=pn.CategoricalIndex(series_order, ordered=False),
                               columns=['order_series', 'BottomRelation'])

        self.df['order_series'] = self.df['order_series'].astype(int)
        self.df['BottomRelation'] = pn.Categorical(['Erosion'], categories=['Erosion', 'Onlap'])

    def __repr__(self):
        return self.df.to_string()

    def _repr_html_(self):
        return self.df.to_html()

    def update_order_series(self):
        """
        Inex of df is categorical and order, but we need to numerate that order to map it later on to the Data dfs
        """
        self.df.at[:, 'order_series'] = pn.RangeIndex(1, self.df.shape[0] + 1)

    def set_series_index(self, series_order: Union[pn.DataFrame, list, np.ndarray], update_order_series=True):
        """
        Rewrite the index of the series df
        Args:
            series_order:
            update_order_series:

        Returns:

        """
        if isinstance(series_order, SurfacePoints):
            try:
                list_of_series = series_order.df['series'].unique()
            except KeyError:
                raise KeyError('Interface does not have series attribute')
        elif type(series_order) is list or type(series_order) is np.ndarray:
            list_of_series = np.atleast_1d(series_order)

        else:
            raise AttributeError('series_order is not neither list or SurfacePoints object.')

        series_idx = list_of_series
        # Categoriacal index does not have inplace
        # This update the categories
        self.df.index = self.df.index.set_categories(series_idx, rename=True)
        self.faults.df.index = self.faults.df.index.set_categories(series_idx, rename=True)
        self.faults.faults_relations_df.index = self.faults.faults_relations_df.index.set_categories(series_idx, rename=True)
        self.faults.faults_relations_df.columns = self.faults.faults_relations_df.columns.set_categories(series_idx, rename=True)

        # But we need to update the values too
        # TODO: isnt this the behaviour we get fif we do not do the rename=True?
        for c in series_order:
            self.df.loc[c, 'BottomRelation'] = 'Erosion'
            self.faults.df.loc[c, 'isFault'] = False
            self.faults.faults_relations_df.loc[c, c] = False

        self.faults.faults_relations_df.fillna(False, inplace=True)

        if update_order_series is True:
            self.update_order_series()

    def add_series(self, series_list: Union[str, list], update_order_series=True):
        series_list = np.atleast_1d(series_list)

        # Remove from the list categories that already exist
        series_list = series_list[~np.in1d(series_list, self.df.index.categories)]

        idx = self.df.index.add_categories(series_list)
        self.df.index = idx
        self.update_faults_index()

        for c in series_list:
            self.df.loc[c, 'BottomRelation'] = 'Erosion'
            self.faults.df.loc[c] = [False, False]
            self.faults.faults_relations_df.loc[c, c] = False

        self.faults.faults_relations_df.fillna(False, inplace=True)

        if update_order_series is True:
            self.update_order_series()

    def delete_series(self, indices, update_order_series=True):
        self.df.drop(indices, inplace=True)
        self.faults.df.drop(indices, inplace=True)
        self.faults.faults_relations_df.drop(indices, axis=0, inplace=True)
        self.faults.faults_relations_df.drop(indices, axis=1, inplace=True)

        idx = self.df.index.remove_unused_categories()
        self.df.index = idx
        self.update_faults_index()

        if update_order_series is True:
            self.update_order_series()

    @_setdoc(pn.CategoricalIndex.rename_categories.__doc__)
    def rename_series(self, new_categories:Union[dict, list]):
        idx = self.df.index.rename_categories(new_categories)
        self.df.index = idx
        self.update_faults_index()

    @_setdoc([pn.CategoricalIndex.reorder_categories.__doc__, pn.CategoricalIndex.sort_values.__doc__])
    def reorder_series(self, new_categories:list):
        idx = self.df.index.reorder_categories(new_categories).sort_values()
        self.df.index = idx
        self.update_faults_index()

    def modify_order_series(self, new_value: int, idx: str):

        group = self.df['order_series']
        assert np.isin(new_value, group), 'new_value must exist already in the order_surfaces group.'
        old_value = group[idx]
        self.df['order_series'] = group.replace([new_value, old_value], [old_value, new_value])
        self.sort_series()
        self.update_faults_index()

        self.faults.sort_faults()
        return self

    def sort_series(self):
        self.df.sort_values(by='order_series', inplace=True)
        self.df.index = self.df.index.reorder_categories(self.df.index.get_values())

    def update_faults_index(self):
        idx = self.df.index
        self.faults.df.index = idx
        self.faults.faults_relations_df.index = idx
        self.faults.faults_relations_df.columns = idx

        # TODO: This is a hack for qgrid

        #  We need to add the qgrid special columns to categories
        self.faults.faults_relations_df.columns = self.faults.faults_relations_df.columns.add_categories(
            ['index', 'qgrid_unfiltered_index'])

    def map_isFault_from_faults_DEP(self, faults):
        # TODO is this necessary?
        self.df['isFault'] = self.df.index.map(faults.faults['isFault'])


class Faults(object):
    """
    Class that encapsulate faulting related content. Mainly, which surfaces/surfaces are faults. The fault network
    ---i.e. which faults offset other faults---and fault types---finite vs infinite
        Args:
            series (Series): Series object
            series_fault (list): List with the name of the series that are faults
            rel_matrix (numpy.array): 2D Boolean array with the logic. Rows affect (offset) columns

        Attributes:
           series (Series): Series object
           df (:class:`pn.core.frame.DataFrames`): Pandas data frame containing the series and if they are faults or
            not (otherwise they are lithologies) and in case of being fault if is finite
           faults_relations_df (:class:`pn.core.frame.DataFrames`): Pandas data frame containing the offsetting relations
            between each fault and the rest of the series (either other faults or lithologies)
           n_faults (int): Number of faults in the object
    """

    def __init__(self, series_fault=None, rel_matrix=None):

        self.df = pn.DataFrame(np.array([[False, False]]), index=pn.CategoricalIndex(['Default series']),
                               columns=['isFault', 'isFinite'], dtype=bool)

        self.set_is_fault(series_fault=series_fault)
        self.faults_relations_df = pn.DataFrame(index=pn.CategoricalIndex(['Default series']),
                                                columns=pn.CategoricalIndex(['Default series', '']), dtype='bool')
        self.set_fault_relation(rel_matrix=rel_matrix)
        self.n_faults = 0

    def __repr__(self):
        return self.df.to_string()

    def _repr_html_(self):
        return self.df.to_html()

    def sort_faults(self):
        self.df.sort_index(inplace=True)
        self.faults_relations_df.sort_index(inplace=True)
        self.faults_relations_df.sort_index(axis=1, inplace=True)

    def set_is_fault(self, series_fault=None):
        """
        Set a flag to the series that are df.

        Args:
            series (Series): Series object
            series_fault(list or SurfacePoints): Name of the series which are df
        """
        series_fault = np.atleast_1d(series_fault)
        self.df['isFault'].fillna(False, inplace=True)

        if series_fault is None:
            series_fault = self.count_faults(self.df.index)

        if series_fault[0] is not None:
            assert np.isin(series_fault, self.df.index).all(), 'series_faults must already ' \
                                                                                      'exist in the the series df.'
            self.df.loc[series_fault, 'isFault'] = self.df.loc[series_fault, 'isFault'] ^ True

        self.n_faults = self.df['isFault'].sum()

        return self.df

    def set_is_finite_fault(self, series_fault=None):
        """
        Toggles given series' finite fault property.

        Args:
            series_fault (list): Name of the series
        """
        if series_fault[0] is not None:
            # check if given series is/are in dataframe
            assert np.isin(series_fault, self.df.index).all(), "series_fault must already exist" \
                                                                "in the series DataFrame."
            assert self.df.loc[series_fault].isFault.all(), "series_fault contains non-fault series" \
                                                            ", which can't be set as finite faults."
            # if so, toggle True/False for given series or list of series
            self.df.loc[series_fault, "isFinite"] = self.df.loc[series_fault, 'isFinite'] ^ True

        return self.df

    def set_fault_relation(self, rel_matrix=None):
        """
        Method to set the df that offset a given sequence and therefore also another fault

        Args:
            rel_matrix (numpy.array): 2D Boolean array with the logic. Rows affect (offset) columns
        """
        # TODO: block the lower triangular matrix of being changed
        if rel_matrix is None:
            rel_matrix = np.zeros((self.df.index.shape[0],
                                   self.df.index.shape[0]))
        else:
            assert type(rel_matrix) is np.ndarray, 'rel_matrix muxt be a 2D numpy array'
        self.faults_relations_df = pn.DataFrame(rel_matrix, index=self.df.index,
                                                columns=self.df.index, dtype='bool')

        return self.faults_relations_df

    @staticmethod
    def count_faults(list_of_names):
        """
        Read the string names of the surfaces to detect automatically the number of df if the name
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


class Surfaces(object):
    """
    Class that contains the surfaces of the model and the values of each of them.

    Args:
        values_array (np.ndarray): 2D array with the values of each surface
        properties names (list or np.ndarray): list containing the names of each properties
        surface_names (list or np.ndarray): list contatinig the names of the surfaces


    Attributes:
        df (:class:`pn.core.frame.DataFrames`): Pandas data frame containing the surfaces names and the value
         used for each voxel in the final model and the lithological order
        surface_names (list[str]): List in order of the surfaces
    """

    def __init__(self, series: Series, values_array=None, properties_names=None, surface_names=None,
                 ):

        self.series = series
        df_ = pn.DataFrame(columns=['surface', 'series', 'order_surfaces', 'isBasement', 'id'])
        self.df = df_.astype({'surface': str, 'series': 'category',
                              'order_surfaces': int, 'isBasement': bool,
                              'id': int})

        self.df['series'].cat.add_categories(['Default series'], inplace=True)

        if surface_names is not None:
            self.set_surfaces_names(surface_names)
        if values_array is not None:
            self.set_surfaces_values(values_array=values_array, properties_names=properties_names)
        self.sequential_pile = StratigraphicPile(self.series, self.df)

    def __repr__(self):
        return self.df.to_string()

    def _repr_html_(self):
        return self.df.to_html()

    def update_sequential_pile(self):
        """
        Method to update the sequential pile plot
        Returns:

        """
        self.sequential_pile = StratigraphicPile(self.series, self.df)

# region set surface names
    def set_surfaces_names(self, list_names: list, update_df=True):
        """
         Method to set the names of the surfaces in order. This applies in the surface column of the df
         Args:
             list_names (list[str]):

         Returns:
             None
         """
        if type(list_names) is list or type(list_names) is np.ndarray:
            list_names = np.asarray(list_names)

        else:
            raise AttributeError('list_names must be either array_like type')

        # Deleting all columns if they exist
        # TODO check if some of the names are in the df and not deleting them?
        self.df.drop(self.df.index, inplace=True)

        self.df['surface'] = list_names

        # Changing the name of the series is the only way to mutate the series object from surfaces
        if update_df is True:
            self.map_series()
            self.set_id()
            self.set_basement()
            self.set_order_surfaces()
            self.update_sequential_pile()
        return True

    def set_default_surface_name(self):
        if self.df.shape[0] == 0:
            # TODO DEBUG: I am not sure that surfaces always has at least one entry. Check it
            self.set_surfaces_names(['surface1', 'basement'])
        return self

    def set_surfaces_names_from_surface_points(self, surface_points):
        self.set_surfaces_names(surface_points.df['surface'].unique())

    def add_surface(self, surface_list: Union[pn.DataFrame, list], update_df=True):
        surface_list = np.atleast_1d(surface_list)

        # Remove from the list categories that already exist
        surface_list = surface_list[~np.in1d(surface_list, self.df['surface'].values)]

        for c in surface_list:
            idx = self.df.index.max()
            if idx is np.nan:
                idx = -1
            self.df.loc[idx + 1, 'surface'] = c
        if update_df is True:
            self.map_series()
            self.set_id()
            self.set_basement()
            self.set_order_surfaces()
            self.update_sequential_pile()
        return True

    def delete_surface(self, indices, update_id=True):
        # TODO passing names of the surface instead the index
        self.df.drop(indices, inplace=True)
        if update_id is True:
            self.set_id()
            self.set_basement()
            self.set_order_surfaces()
            self.update_sequential_pile()
        return True

    @_setdoc([pn.CategoricalIndex.reorder_categories.__doc__, pn.CategoricalIndex.sort_values.__doc__])
    def reorder_surfaces(self, list_names):
        """"""

        # check if list_names are all already in the columns
        assert self.df['surface'].shape[0] == len(list_names), 'list_names and the surface column mush have the same' \
                                                                 'lenght'
        assert self.df['surface'].isin(list_names).all(), 'Every element of list_names must already exist in the df'

        self.df['surface'] = list_names
        self.set_basement()


    @_setdoc(pn.Series.replace.__doc__)
    def rename_surfaces(self, old_value=None, new_value=None, **kwargs):
        if np.isin(new_value, self.df['surface']).any():
            print('Two surfaces cannot have the same name.')
        else:
            self.df['surface'].replace(old_value, new_value, inplace=True, **kwargs)
        return True

    def set_order_surfaces(self):
        #self.df['order_surfaces'] = 1
        self.df['order_surfaces'] = self.df.groupby('series').cumcount() + 1

    def modify_order_surfaces(self, new_value: int, idx: int, series: str = None):

        if series is None:
            series = self.df.loc[idx, 'series']

        group = self.df.groupby('series').get_group(series)['order_surfaces']
        assert np.isin(new_value, group), 'new_value must exist already in the order_surfaces group.'
        old_value = group[idx]
        self.df.loc[group.index, 'order_surfaces'] = group.replace([new_value, old_value], [old_value, new_value])
        self.sort_surfaces()
        self.set_basement()

    def sort_surfaces(self):

        self.df.sort_values(by=['series', 'order_surfaces'], inplace=True)
        self.set_id()
        return self.df

    def set_basement(self):
        """


        Returns:
            True
        """
        # TODO - FIXED-Waiting for testing: it has to be always on the bottom. I am not sure what the hell I am doing
        #  here

        self.df['isBasement'] = False
        idx = self.df.last_valid_index()
        if idx is not None:
            self.df.loc[idx, 'isBasement'] = True

        # TODO add functionality of passing the basement and calling reorder to push basement surface to the bottom
        #  of the data frame

        assert self.df['isBasement'].values.astype(bool).sum() <= 1, 'Only one surface can be basement'
# endregion

# set_series
    def map_series(self, mapping_object: Union[dict, pn.Categorical] = None, idx=None):
        """

        Args:
            mapping_object:

        Returns:

        """

        # Updating surfaces['series'] categories
        self.df['series'].cat.set_categories(self.series.df.index, inplace=True)

        # TODO Fixing this. It is overriding the formtions already mapped
        if mapping_object is not None:
            # If none is passed and series exist we will take the name of the first series as a default

            if type(mapping_object) is dict:

                s = []
                f = []
                for k, v in mapping_object.items():
                    for form in np.atleast_1d(v):
                        s.append(k)
                        f.append(form)

                # TODO does series_mapping have to be in self?
                new_series_mapping = pn.DataFrame([pn.Categorical(s, self.series.df.index)],
                                                   f, columns=['series'])

            elif isinstance(mapping_object, pn.Categorical):
                # This condition is for the case we have surface on the index and in 'series' the category
                new_series_mapping = mapping_object

            else:
                raise AttributeError(str(type(mapping_object))+' is not the right attribute type.')

            # Checking which surfaces are on the list to be mapped
            b = self.df['surface'].isin(new_series_mapping.index)
            idx = self.df.index[b]

            # Mapping
            self.df.loc[idx, 'series'] = self.df.loc[idx, 'surface'].map(new_series_mapping['series'])

        # Fill nans
        self.df['series'].fillna(self.series.df.index.values[-1], inplace=True)

        # Reorganize the pile
        self.set_order_surfaces()
        self.sort_surfaces()
        self.set_basement()

# endregion

# region set_id
    def set_id(self, id_list: list = None):
        """
        Set id of the layers (1 based)
        Args:
            df:

        Returns:

        """
        if id_list is None:
            id_list = self.df.reset_index().index + 1

        self.df['id'] = id_list

        return self.df
# endregion

    def add_surfaces_values(self, values_array, properties_names=np.empty(0)):
        values_array = np.atleast_2d(values_array)
        properties_names = np.asarray(properties_names)
        if properties_names.shape[0] != values_array.shape[0]:
            for i in range(values_array.shape[0]):
                properties_names = np.append(properties_names, 'value_' + str(i))

        for e, p_name in enumerate(properties_names):
            try:
                self.df.loc[:, p_name] = values_array[e]
            except ValueError:
                raise ValueError('value_array must have the same length in axis 0 as the number of surfaces')
        return self

    def delete_surface_values(self, properties_names):
        properties_names = np.asarray(properties_names)
        self.df.drop(properties_names, axis=1, inplace=True)
        return True

    def set_surfaces_values(self, values_array, properties_names=np.empty(0)):
        # Check if there are values columns already
        old_prop_names = self.df.columns[~self.df.columns.isin(['surface', 'series', 'order_surfaces',
                                                                'id', 'isBasement'])]
        # Delete old
        self.delete_surface_values(old_prop_names)

        # Create new
        self.add_surfaces_values(values_array, properties_names)
        return self

    def modify_surface_values(self):
        """Method to modify values using loc of pandas"""
        raise NotImplementedError


class GeometricData(object):
    """
    Parent class of the objects which contatin the input parameters: surface_points and orientations. This class contain
    the common methods for both types of data sets.
    """

    def __init__(self, surfaces: Surfaces):

        self.surfaces = surfaces
        self.df = pn.DataFrame()
       # self.agg_index = self.df.index

    def __repr__(self):
        return self.df.to_string()

    def _repr_html_(self):
        return self.df.to_html()

    def update_series_category(self):
        self.df['series'].cat.set_categories(self.surfaces.df['series'].cat.categories, inplace=True)

    def set_dependent_properties(self):
        # series
        self.df['series'] = 'Default series'
        self.df['series'] = self.df['series'].astype('category', copy=True)
        self.df['series'].cat.set_categories(self.surfaces.df['series'].cat.categories, inplace=True)

        # id
        self.df['id'] = np.nan

        # order_series
        self.df['order_series'] = 1

    @staticmethod
    def read_data(file_path, **kwargs):
        """
        Read method of pandas for different types of tabular data
        Args:
            file_path(str):
            **kwargs:  See pandas read_table

        Returns:
             pandas.core.frame.DataFrame: Data frame with the raw data
        """
        if 'sep' not in kwargs:
            kwargs['sep'] = ','

        table = pn.read_csv(file_path, **kwargs)

        return table

    def sort_table(self):
        """
        First we sort the dataframes by the series age. Then we set a unique number for every surface and resort
        the surfaces. All inplace
        """

        # We order the pandas table by surface (also by series in case something weird happened)
        self.df.sort_values(by=['order_series', 'surface'],
                            ascending=True, kind='mergesort',
                            inplace=True)
        return self.df

    def map_data_from_series(self, series, property:str, idx=None):
        """

        """
        if idx is None:
            idx = self.df.index

        self.df.loc[idx, property] = self.df['series'].map(series.df[property])

    def add_series_categories_from_series(self, series: Series):
        self.df['series'].cat.set_categories(series.df.index, inplace=True)
        return True

    def add_surface_categories_from_surfaces(self, surfaces: Surfaces):
        self.df['surface'].cat.set_categories(surfaces.df['surface'], inplace=True)
        return True

    def map_data_from_surfaces(self, surfaces, property:str, idx=None):
        """Map properties of surfaces---series, id, values--- into a data df"""

        if idx is None:
            idx = self.df.index

        if property is 'series':
            if surfaces.df.loc[~surfaces.df['isBasement']]['series'].isna().sum() != 0:
                raise AttributeError('Surfaces does not have the correspondent series assigned. See'
                                     'Surfaces.map_series_from_series.')

        self.df.loc[idx, property] = self.df.loc[idx, 'surface'].map(surfaces.df.set_index('surface')[property])

    def map_data_from_faults(self, faults, idx=None):
        """
        Method to map a df object into the data object on surfaces. Either if the surface is fault or not
        Args:
            faults (Faults):

        Returns:
            pandas.core.frame.DataFrame: Data frame with the raw data

        """
        if idx is None:
            idx = self.df.index

        if any(self.df['series'].isna()):
            warnings.warn('Some points do not have series/fault')

        self.df.loc[idx, 'isFault'] = self.df.loc[idx, 'series'].map(faults.df['isFault'])

    def set_dypes_DEP(self):
        """
        Method to set each column of the dataframe to the right data type. Inplace
        Returns:

        """
        # Choose types
        self.df['surface'] = self.df['surface'].astype('category', copy=True)
        self.df['series'] = self.df['series'].astype('category', copy=True)
        self.df['isFault'] = self.df['isFault'].astype('bool')
        try:
            self.df[['order_series', 'id']] = self.df[
                ['order_series', 'id']].astype(int, copy=True)
        except ValueError:
            warnings.warn('You may have non-finite values (NA or inf) on the dataframe')


class SurfacePoints(GeometricData):
    """
    Data child with specific methods to manipulate interface data. It is initialize without arguments to give
    flexibility to the origin of the data

    Attributes:
          df (:class:`pn.core.frame.DataFrames`): Pandas data frame containing the necessary information respect
            the interface points of the model
    """

    def __init__(self, surfaces: Surfaces, coord=None, surface=None):

        super().__init__(surfaces)
        self._columns_i_all = ['X', 'Y', 'Z', 'surface', 'series', 'X_std', 'Y_std', 'Z_std',
                               'order_series', 'surface_number']
        self._columns_i_1 = ['X', 'Y', 'Z', 'X_r', 'Y_r', 'Z_r', 'surface', 'series', 'id',
                             'order_series', 'isFault']
        self._columns_i_num = ['X', 'Y', 'Z', 'X_r', 'Y_r', 'Z_r']
        #self.df = pn.DataFrame(columns=self._columns_i_1)
        if (np.array(sys.version_info[:2]) <= np.array([3, 6])).all():
            self.df: pn.DataFrame

        self.set_surface_points(coord, surface)

    def set_surface_points(self, coord: np.ndarray = None, surface: list = None):
        self.df = pn.DataFrame(columns=['X', 'Y', 'Z', 'X_r', 'Y_r', 'Z_r', 'surface'], dtype=float)

        if coord is not None and surface is not None:
            self.df[['X', 'Y', 'Z']] = pn.DataFrame(coord)
            self.df['surface'] = surface

        self.df['surface'] = self.df['surface'].astype('category', copy=True)
        self.df['surface'].cat.set_categories(self.surfaces.df['surface'].values, inplace=True)

        # Choose types
        self.set_dependent_properties()

        assert ~self.df['surface'].isna().any(), 'Some of the surface passed does not exist in the Formation' \
                                                 'object. %s' % self.df['surface'][self.df['surface'].isna()]

    def add_surface_points(self, X, Y, Z, surface, idx=None):
        # TODO: Add the option to pass the surface number

        if idx is None:
            idx = self.df.index.max()
            if idx is np.nan:
                idx = -1

        coord_array = np.array([X, Y, Z])
        assert coord_array.ndim == 1, 'Adding an interface only works one by one.'
        self.df.loc[idx + 1, ['X', 'Y', 'Z']] = coord_array

        try:
            self.df.loc[idx + 1, 'surface'] = surface
        # ToDO test this
        except ValueError as error:
            self.del_surface_points(idx + 1)
            print('The surface passed does not exist in the pandas categories. This may imply that'
                  'does not exist in the surface object either.')
            raise ValueError(error)

    def del_surface_points(self, idx):

        self.df.drop(idx, inplace=True)

    def modify_surface_points(self, idx, **kwargs):
        """
         Allows modification of the x,y and/or z-coordinates of an interface at specified dataframe index.

         Args:
             index: dataframe index of the orientation point
             **kwargs: X, Y, Z (int or float), surface

         Returns:
             None
         """

        # Check idx exist in the df
        assert np.isin(np.atleast_1d(idx), self.df.index).all(), 'Indices must exist in the dataframe to be modified.'

        # Check the properties are valid
        assert np.isin(list(kwargs.keys()), ['X', 'Y', 'Z', 'surface']).all(), 'Properties must be one or more of the' \
                                                                                 'following: \'X\', \'Y\', \'Z\', ' \
                                                                                 '\'surface\''
        # stack properties values
        values = np.array(list(kwargs.values()))

        # If we pass multiple index we need to transpose the numpy array
        if type(idx) is list:
            values = values.T

        # Selecting the properties passed to be modified
        self.df.loc[idx, list(kwargs.keys())] = values

    def read_surface_points(self, file_path, debug=False, inplace=False, append=False, kwargs_pandas:dict = {}, **kwargs, ):
        """
        Read tabular using pandas tools and if inplace set it properly to the Interace object
        Args:
            file_path:
            debug:
            inplace:
            append:
            **kwargs:

        Returns:

        """
        if 'sep' not in kwargs:
            kwargs['sep'] = ','

        coord_x_name = kwargs.get('coord_x_name', "X")
        coord_y_name = kwargs.get('coord_y_name', "Y")
        coord_z_name = kwargs.get('coord_z_name', "Z")
        surface_name = kwargs.get('surface_name', "formation")
        if 'sep' not in kwargs_pandas:
            kwargs_pandas['sep'] = ','

        table = pn.read_csv(file_path, **kwargs_pandas)

        if 'update_surfaces' in kwargs:
            if kwargs['update_surfaces'] is True:
                self.surfaces.add_surface(table[surface_name].unique())

        if debug is True:
            print('Debugging activated. Changes won\'t be saved.')
            return table
        else:
            assert set([coord_x_name, coord_y_name, coord_z_name, surface_name]).issubset(table.columns), \
                "One or more columns do not match with the expected values " + str(table.columns)

            if inplace:
                c = np.array(self._columns_i_1)
                surface_points_read = table.assign(**dict.fromkeys(c[~np.in1d(c, table.columns)], np.nan))
                self.set_surface_points(surface_points_read[[coord_x_name, coord_y_name, coord_z_name]],
                                        surface=surface_points_read[surface_name])
            else:
                return table

    def set_default_surface_points(self):
        """
        Set a default point at the middle of the extent area to be able to start making the model
        Args:
            surface:
            grid:

        Returns:

        """
        if self.df.shape[0] == 0:
            self.add_surface_points(0.00001, 0.00001, 0.00001, self.surfaces.df['surface'].iloc[0])

    def get_surfaces(self):
        """
        Returns:
             pandas.core.frame.DataFrame: Returns a list of surfaces

        """
        return self.df["surface"].unique()

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


class Orientations(GeometricData):
    """
    Data child with specific methods to manipulate orientation data. It is initialize without arguments to give
    flexibility to the origin of the data

    Attributes:
        df (:class:`pn.core.frame.DataFrames`): Pandas data frame containing the necessary information respect
         the orientations of the model
    """

    def __init__(self, surfaces: Surfaces, coord=None, pole_vector=None, orientation=None, surface=None):
        super().__init__(surfaces)
        self._columns_o_all = ['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', 'dip', 'azimuth', 'polarity',
                               'surface', 'series', 'id', 'order_series', 'surface_number']
        self._columns_o_1 = ['X', 'Y', 'Z', 'X_r', 'Y_r', 'Z_r', 'G_x', 'G_y', 'G_z', 'dip', 'azimuth', 'polarity',
                             'surface', 'series', 'id', 'order_series', 'isFault']
        self._columns_o_num = ['X', 'Y', 'Z', 'X_r', 'Y_r', 'Z_r', 'G_x', 'G_y', 'G_z', 'dip', 'azimuth', 'polarity']
        if (np.array(sys.version_info[:2]) <= np.array([3, 6])).all():
            self.df: pn.DataFrame

        self.set_orientations(coord, pole_vector, orientation, surface)
     #   self.df = pn.DataFrame(columns=self._columns_o_1)
     #   self.df[self._columns_o_num] = self.df[self._columns_o_num].astype(float)
     #   self.df.itype = 'orientations'
     #   self.calculate_gradient()

    def set_orientations(self, coord: np.ndarray = None, pole_vector: np.ndarray = None,
                         orientation: np.ndarray = None, surface: list = None):
        """
        Pole vector has priority over orientation
        Args:
            coord:
            pole_vector:
            orientation:
            surface:

        Returns:

        """
        self.df = pn.DataFrame(columns=['X', 'Y', 'Z', 'X_r', 'Y_r', 'Z_r', 'G_x', 'G_y', 'G_z', 'dip',
                                        'azimuth', 'polarity', 'surface'], dtype=float)

        self.df['surface'] = self.df['surface'].astype('category', copy=True)
        self.df['surface'].cat.set_categories(self.surfaces.df['surface'].values, inplace=True)

        pole_vector = check_for_nans(pole_vector)
        orientation = check_for_nans(orientation)

        if coord is not None and ((pole_vector is not None) or (orientation is not None)) and surface is not None:
            #self.df = pn.DataFrame(coord, columns=['X', 'Y', 'Z', 'X_r', 'Y_r', 'Z_r'], dtype=float)

            self.df[['X', 'Y', 'Z']] = pn.DataFrame(coord)
            self.df['surface'] = surface
            if pole_vector is not None:
                self.df['G_x'] = pole_vector[:, 0]
                self.df['G_y'] = pole_vector[:, 1]
                self.df['G_z'] = pole_vector[:, 2]
                self.calculate_orientations()

                if orientation is not None:
                    warnings.warn('If pole_vector and orientation are passed pole_vector is used/')
            else:
                if orientation is not None:
                    self.df['azimuth'] = orientation[:, 0]
                    self.df['dip'] = orientation[:, 1]
                    self.df['polarity'] = orientation[:, 2]
                    self.calculate_gradient()
                else:
                    raise AttributeError('At least pole_vector or orientation should have been passed to reach'
                                         'this point. Check previous condition')

        self.df['surface'] = self.df['surface'].astype('category', copy=True)
        self.df['surface'].cat.set_categories(self.surfaces.df['surface'].values, inplace=True)

        # Check that the minimum parameters are passed. Otherwise create an empty df
        # if coord is None or ((pole_vector is None) and (orientation is None)) or surface is None:
        #     self.df = pn.DataFrame(columns=['X', 'Y', 'Z', 'X_r', 'Y_r', 'Z_r', 'G_x', 'G_y', 'G_z', 'dip',
        #                                     'azimuth', 'polarity', 'surface'])
        # else:
        #     self.df = pn.DataFrame(coord, columns=['X', 'Y', 'Z', 'X_r', 'Y_r', 'Z_r'], dtype=float)
        #     self.df['surface'] = surface
        #     if pole_vector is not None:
        #         self.df['G_x'] = pole_vector[:, 0]
        #         self.df['G_y'] = pole_vector[:, 1]
        #         self.df['G_z'] = pole_vector[:, 2]
        #         self.calculate_orientations()
        #
        #         if orientation is not None:
        #             warnings.warn('If pole_vector and orientation are passed pole_vector is used/')
        #     else:
        #         if orientation is not None:
        #             self.df['azimuth'] = orientation[:, 0]
        #             self.df['dip'] = orientation[:, 1]
        #             self.df['polarity'] = orientation[:, 2]
        #             self.calculate_gradient()
        #         else:
        #             raise AttributeError('At least pole_vector or orientation should have been passed to reach'
        #                                  'this point. Check previous condition')
        # Choose types
        #  self.df[self._columns_i_num] = self.df[self._columns_i_num].astype(float)

        self.set_dependent_properties()
        assert ~self.df['surface'].isna().any(), 'Some of the surface passed does not exist in the Formation' \
                                                 'object. %s' % self.df['surface'][self.df['surface'].isna()]

    def add_orientation(self, X, Y, Z, surface, pole_vector: np.ndarray = None,
                        orientation: np.ndarray = None, idx=None):
        if pole_vector is None and orientation is None:
            raise AttributeError('Either pole_vector or orientation must have a value. If both are passed pole_vector'
                                 'has preference')

        if idx is None:
            idx = self.df.index.max()
            if idx is np.nan:
                idx = 0

        if pole_vector is not None:
            self.df.loc[idx, ['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z']] = np.array([X, Y, Z, *pole_vector])
            self.df.loc[idx, 'surface'] = surface

            self.calculate_orientations(idx)

            if orientation is not None:
                warnings.warn('If pole_vector and orientation are passed pole_vector is used/')
        else:
            if orientation is not None:
                self.df.loc[idx, ['X', 'Y', 'Z', 'azimuth', 'dip', 'polarity']] = np.array(
                    [X, Y, Z, *orientation])
                self.df.loc[idx, 'surface'] = surface

                self.calculate_gradient(idx)
            else:
                raise AttributeError('At least pole_vector or orientation should have been passed to reach'
                                     'this point. Check previous condition')

    def del_orientation(self, idx):

        self.df.drop(idx, inplace=True)

    def modify_orientations(self, idx, **kwargs):
        """
         Allows modification of the x,y and/or z-coordinates of an interface at specified dataframe index.

         Args:
             index: dataframe index of the orientation point
             **kwargs: X, Y, Z, 'G_x', 'G_y', 'G_z', 'dip', 'azimuth', 'polarity', 'surface' (int or float), surface

         Returns:
             None
         """

        # Check idx exist in the df
        assert np.isin(np.atleast_1d(idx), self.df.index).all(), 'Indices must exist in the dataframe to be modified.'

        # Check the properties are valid
        assert np.isin(list(kwargs.keys()), ['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', 'dip',
                                             'azimuth', 'polarity', 'surface']).all(),\
            'Properties must be one or more of the following: \'X\', \'Y\', \'Z\', \'G_x\', \'G_y\', \'G_z\', \'dip,\''\
            '\'azimuth\', \'polarity\', \'surface\''

        # stack properties values
        values = np.array(list(kwargs.values()))

        # If we pass multiple index we need to transpose the numpy array
        if type(idx) is list:
            values = values.T

        # Selecting the properties passed to be modified
        self.df.loc[idx, list(kwargs.keys())] = values

        if np.isin(list(kwargs.keys()), ['G_x', 'G_y', 'G_z']).any():
            self.calculate_orientations(idx)
        else:
            if np.isin(list(kwargs.keys()), ['azimuth', 'dip', 'polarity']).any():
                self.calculate_gradient(idx)

    def calculate_gradient(self, idx=None):
        """
        Calculate the gradient vector of module 1 given dip and azimuth to be able to plot the orientations
        """
        # TODO @Elisa is this already the last version?
        if idx is None:
            self.df['G_x'] = np.sin(np.deg2rad(self.df["dip"].astype('float'))) * \
                             np.sin(np.deg2rad(self.df["azimuth"].astype('float'))) * \
                             self.df["polarity"].astype('float') + 1e-12
            self.df['G_y'] = np.sin(np.deg2rad(self.df["dip"].astype('float'))) * \
                             np.cos(np.deg2rad(self.df["azimuth"].astype('float'))) * \
                             self.df["polarity"].astype('float') + 1e-12
            self.df['G_z'] = np.cos(np.deg2rad(self.df["dip"].astype('float'))) * \
                             self.df["polarity"].astype('float') + 1e-12
        else:
            self.df.loc[idx, 'G_x'] = np.sin(np.deg2rad(self.df.loc[idx, "dip"].astype('float'))) * \
                                      np.sin(np.deg2rad(self.df.loc[idx, "azimuth"].astype('float'))) * \
                                      self.df.loc[idx, "polarity"].astype('float') + 1e-12
            self.df.loc[idx, 'G_y'] = np.sin(np.deg2rad(self.df.loc[idx, "dip"].astype('float'))) * \
                                      np.cos(np.deg2rad(self.df.loc[idx, "azimuth"].astype('float'))) * \
                                      self.df.loc[idx, "polarity"].astype('float') + 1e-12
            self.df.loc[idx, 'G_z'] = np.cos(np.deg2rad(self.df.loc[idx, "dip"].astype('float'))) * \
                                      self.df.loc[idx, "polarity"].astype('float') + 1e-12

    def calculate_orientations(self, idx=None):
        """
        Calculate and update the orientation data (azimuth and dip) from gradients in the data frame.

        Authors: Elisa Heim, Miguel de la Varga
        """
        if idx is None:
            self.df['polarity'] = 1
            self.df["dip"] = np.rad2deg(np.nan_to_num(np.arccos(self.df["G_z"] / self.df["polarity"])))

            self.df["azimuth"] = np.rad2deg(np.nan_to_num(np.arctan2(self.df["G_x"] / self.df["polarity"],
                                                                     self.df["G_y"] / self.df["polarity"])))
            self.df["azimuth"][self.df["azimuth"] < 0] += 360  # shift values from [-pi, 0] to [pi,2*pi]
            self.df["azimuth"][self.df["dip"] < 0.001] = 0  # because if dip is zero azimuth is undefined

        else:

            self.df.loc[idx, 'polarity'] = 1
            self.df.loc[idx, "dip"] = np.rad2deg(np.nan_to_num(np.arccos(self.df.loc[idx, "G_z"] /
                                                                         self.df.loc[idx, "polarity"])))

            self.df.loc[idx, "azimuth"] = np.rad2deg(np.nan_to_num(
                np.arctan2(self.df.loc[idx, "G_x"] / self.df.loc[idx, "polarity"],
                           self.df.loc[idx, "G_y"] / self.df.loc[idx, "polarity"])))

            self.df["azimuth"][self.df["azimuth"] < 0] += 360  # shift values from [-pi, 0] to [pi,2*pi]
            self.df["azimuth"][self.df["dip"] < 0.001] = 0  # because if dip is zero azimuth is undefined

    @staticmethod
    def create_orientation_from_interface_UPDATE(surface_points: SurfacePoints, indices):
        # TODO test!!!!
        """
        Create and set orientations from at least 3 points categories_df
        Args:
            indices
        """
        selected_points = surface_points.df[['X', 'Y', 'Z']].loc[indices].values.T

        center, normal = plane_fit(selected_points)
        orientation = get_orientation(normal)

        return np.array([*center, *orientation, *normal])

    def set_default_orientation(self):
        # TODO: TEST
        """
        Set a default point at the middle of the extent area to be able to start making the model
        """
        if self.df.shape[0] == 0:
            # TODO DEBUG: I am not sure that surfaces always has at least one entry. Check it
            self.add_orientation(.00001, .00001, .00001,
                                 self.surfaces.df['surface'].iloc[0],
                                 [0, 0, 1],
                                 )

    def read_orientations(self, filepath, debug=False, inplace=True, append=False, kwargs_pandas = {}, **kwargs):
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
        if 'sep' not in kwargs_pandas:
            kwargs_pandas['sep'] = ','

        coord_x_name = kwargs.get('coord_x_name', "X")
        coord_y_name = kwargs.get('coord_y_name', "Y")
        coord_z_name = kwargs.get('coord_z_name', "Z")
        G_x_name = kwargs.get('G_x_name', 'G_x')
        G_y_name = kwargs.get('G_y_name', 'G_y')
        G_z_name = kwargs.get('G_z_name', 'G_z')
        azimuth_name = kwargs.get('azimuth_name', 'azimuth')
        dip_name = kwargs.get('dip_name', 'dip')
        polarity_name = kwargs.get('polarity_name', 'polarity')
        surface_name = kwargs.get('surface_name', "formation")

        table = pn.read_csv(filepath, **kwargs_pandas)

        if 'update_surfaces' in kwargs:
            if kwargs['update_surfaces'] is True:
                self.surfaces.add_surface(table[surface_name].unique())

        if debug is True:
            print('Debugging activated. Changes won\'t be saved.')
            return table

        else:
            assert set([coord_x_name, coord_y_name, coord_z_name, dip_name, azimuth_name,
                        polarity_name, surface_name]).issubset(table.columns), \
                "One or more columns do not match with the expected values " + str(table.columns)

            if inplace:
                # self.categories_df[table.columns] = table
                c = np.array(self._columns_o_1)
                orientations_read = table.assign(**dict.fromkeys(c[~np.in1d(c, table.columns)], np.nan))
                self.set_orientations(coord=orientations_read[[coord_x_name, coord_y_name, coord_z_name]],
                                      pole_vector=orientations_read[[G_x_name, G_y_name, G_z_name]].values,
                                      orientation=orientations_read[[azimuth_name, dip_name, polarity_name]].values,
                                      surface=orientations_read[surface_name])
            else:
                return table

    def set_orientations_df_DEP(self, foliat_dataframe, append=False, order_table=True):
        """
          Method to change or append a Dataframe to orientations in place.  A equivalent Pandas Dataframe with
        ['X', 'Y', 'Z', 'dip', 'azimuth', 'polarity', 'surface'] has to be passed.

          Args:
              interf_Dataframe: pandas.core.frame.DataFrame with the data
              append: Bool: if you want to append the new data frame or substitute it
          """
        assert set(self._columns_o_1).issubset(
            foliat_dataframe.columns), "One or more columns do not match with the expected values " + \
                                       str(self._columns_o_1)

        foliat_dataframe[self._columns_o_num] = foliat_dataframe[self._columns_o_num].astype(float, copy=True)

        if append:
            self.df = self.orientations.df.append(foliat_dataframe)
        else:
            self.df = foliat_dataframe[self._columns_o_1]

        self.calculate_gradient()

    def set_annotations(self):
        """
        Add a column in the Dataframes with latex names for each input_data paramenter.

        Returns:
            None
        """

        orientation_num = self.df.groupby('id').cumcount()
        foli_l = [r'${\bf{x}}_{\beta \,{\bf{' + str(f) + '}},' + str(p) + '}$'
                  for p, f in zip(orientation_num, self.df['id'])]

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

    else:
        raise ValueError('The values of normal are not valid.')

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
        surface_points (SurfacePoints):
        orientaions (Orientations):
        grid (Grid):
        rescaling_factor (float): value which divide all coordinates
        centers (list[float]): New center of the coordinates after shifting

    Args:
        surface_points (SurfacePoints):
        orientations (Orientations):
        grid (Grid):
        rescaling_factor (float): value which divide all coordinates
        centers (list[float]): New center of the coordinates after shifting
    """

    def __init__(self, surface_points: SurfacePoints, orientations: Orientations, grid: Grid,
                 rescaling_factor: float = None, centers: Union[list, pn.DataFrame] = None):

        self.surface_points = surface_points
        self.orientations = orientations
        self.grid = grid
       # self.centers = centers

        self.df = pn.DataFrame(np.array([rescaling_factor, centers]).reshape(1, -1),
                               index=['values'],
                               columns=['rescaling factor', 'centers'])

       # self.rescaling_factor = rescaling_factor
        self.rescale_data(rescaling_factor=rescaling_factor, centers=centers)

    def __repr__(self):
        return self.df.T.to_string()

    def _repr_html_(self):
        return self.df.T.to_html()

    def modify_rescaling_parameters(self, property, value):
        assert np.isin(property, self.df.columns).all(), 'Valid properties are: ' + np.array2string(self.df.columns)

        if property == 'centers':
            try:
                assert value.shape[0] is 3

                self.df.loc['values', property] = value

            except AssertionError:
                print('centers length must be 3: XYZ')

        else:
            self.df.loc['values', property] = value

    def rescale_data(self, rescaling_factor=None, centers=None):
        """
        Rescale surface_points, orientations---adding columns in the categories_df---and grid---adding values_r attribute
        Args:
            rescaling_factor:
            centers:

        Returns:

        """
        max_coord, min_coord = self.max_min_coord(self.surface_points, self.orientations)
        if rescaling_factor is None:
            self.df['rescaling factor'] = self.compute_rescaling_factor(self.surface_points, self.orientations,
                                                                  max_coord, min_coord)
        else:
            self.df['rescaling factor'] = rescaling_factor
        if centers is None:
            self.df.at['values', 'centers'] = self.compute_data_center(self.surface_points, self.orientations,
                                                    max_coord, min_coord)
        else:
            self.df.at['values', 'centers'] = centers

        self.set_rescaled_surface_points()
        self.set_rescaled_orientations()
        self.set_rescaled_grid()
        return True

    def get_rescaled_surface_points(self):
        """
        Get the rescaled coordinates. return an image of the interface and orientations categories_df with the X_r..
         columns
        Returns:

        """
        # TODO return an image of the interface and orientations categories_df with the X_r.. columns
        warnings.warn('This method is not developed yet')
        return self.surface_points.df[['X_r', 'Y_r', 'Z_r']],

    def get_rescaled_orientations(self):
        """
        Get the rescaled coordinates. return an image of the interface and orientations categories_df with the X_r..
         columns
        Returns:

        """
        # TODO return an image of the interface and orientations categories_df with the X_r.. columns
        warnings.warn('This method is not developed yet')
        return self.orientations.df[['X_r', 'Y_r', 'Z_r']]

    @staticmethod
    def max_min_coord(surface_points=None, orientations=None):
        """
        Find the maximum and minimum location of any input data in each cartesian coordinate
        Args:
            surface_points (SurfacePoints):
            orientations (Orientations):

        Returns:
            tuple: max[XYZ], min[XYZ]
        """
        if surface_points is None:
            if orientations is None:
                raise AttributeError('You must pass at least one Data object')
            else:
                df = orientations.df
        else:
            if orientations is None:
                df = surface_points.df
            else:
                df = pn.concat([orientations.df, surface_points.df], sort=False)

        max_coord = df.max()[['X', 'Y', 'Z']]
        min_coord = df.min()[['X', 'Y', 'Z']]
        return max_coord, min_coord

    def compute_data_center(self, surface_points=None, orientations=None,
                            max_coord=None, min_coord=None, inplace=True):
        """
        Calculate the center of the data once it is shifted between 0 and 1
        Args:
            surface_points:
            orientations:
            max_coord:
            min_coord:

        Returns:

        """

        if max_coord is None or min_coord is None:
            max_coord, min_coord = self.max_min_coord(surface_points, orientations)

        # Get the centers of every axis
        centers = ((max_coord + min_coord) / 2).astype(float).values
        if inplace is True:
            self.df.at['values', 'centers'] = centers
        return centers

    def update_centers(self, surface_points=None, orientations=None, max_coord=None, min_coord=None):
        # TODO this should update the additional data
        self.compute_data_center(surface_points, orientations, max_coord, min_coord, inplace=True)

    def compute_rescaling_factor(self, surface_points=None, orientations=None,
                                 max_coord=None, min_coord=None, inplace=True):
        """
        Calculate the rescaling factor of the data to keep all coordinates between 0 and 1
        Args:
            surface_points:
            orientations:
            max_coord:
            min_coord:

        Returns:

        """

        if max_coord is None or min_coord is None:
            max_coord, min_coord = self.max_min_coord(surface_points, orientations)
        rescaling_factor_val = (2 * np.max(max_coord - min_coord))
        if inplace is True:
            self.df['rescaling factor'] = rescaling_factor_val
        return rescaling_factor_val

    def update_rescaling_factor(self, surface_points=None, orientations=None,
                                max_coord=None, min_coord=None):
        self.compute_rescaling_factor(surface_points, orientations, max_coord, min_coord, inplace=True)

    @staticmethod
    @_setdoc([compute_data_center.__doc__, compute_rescaling_factor.__doc__])
    def rescale_surface_points(surface_points, rescaling_factor, centers, idx: list = None):
        """
        Rescale surface_points
        Args:
            surface_points:
            rescaling_factor:
            centers:

        Returns:

        """

        if idx is None:
            idx = surface_points.df.index

        # Change the coordinates of surface_points
        new_coord_surface_points = (surface_points.df.loc[idx, ['X', 'Y', 'Z']] -
                                centers) / rescaling_factor + 0.5001

        new_coord_surface_points.rename(columns={"X": "X_r", "Y": "Y_r", "Z": 'Z_r'}, inplace=True)
        return new_coord_surface_points

    def set_rescaled_surface_points(self, idx: list = None):
        """
        Set the rescaled coordinates into the surface_points categories_df
        Returns:

        """
        if idx is None:
            idx = self.surface_points.df.index
            # if idx.empty:
            #     idx = 0

        self.surface_points.df.loc[idx, ['X_r', 'Y_r', 'Z_r']] = self.rescale_surface_points(self.surface_points,
                                                                                     self.df.loc['values', 'rescaling factor'],
                                                                                     self.df.loc['values', 'centers'],
                                                                                     idx=idx)

        return True

    def rescale_data_point(self, data_points: np.ndarray, rescaling_factor=None, centers=None):
        if rescaling_factor is None:
            rescaling_factor = self.df.loc['values', 'rescaling factor']
        if centers is None:
            centers = self.df.loc['values', 'centers']

        rescaled_data_point = (data_points - centers) / rescaling_factor + 0.5001

        return rescaled_data_point

    @staticmethod
    @_setdoc([compute_data_center.__doc__, compute_rescaling_factor.__doc__])
    def rescale_orientations(orientations, rescaling_factor, centers, idx: list = None):
        """
        Rescale orientations
        Args:
            orientations:
            rescaling_factor:
            centers:

        Returns:

        """
        if idx is None:
            idx = orientations.df.index

            # if idx.empty:
            #     idx = 0
        # Change the coordinates of orientations
        new_coord_orientations = (orientations.df.loc[idx, ['X', 'Y', 'Z']] -
                                  centers) / rescaling_factor + 0.5001

        new_coord_orientations.rename(columns={"X": "X_r", "Y": "Y_r", "Z": 'Z_r'}, inplace=True)

        return new_coord_orientations

    def set_rescaled_orientations(self, idx: list = None):
        """
        Set the rescaled coordinates into the orientations categories_df
        Returns:

        """

        if idx is None:
            idx = self.orientations.df.index

        self.orientations.df.loc[idx, ['X_r', 'Y_r', 'Z_r']] = self.rescale_orientations(self.orientations,
                                                                                         self.df.loc['values', 'rescaling factor'],
                                                                                         self.df.loc['values', 'centers'],
                                                                                         idx=idx)
        return True

    @staticmethod
    def rescale_grid(grid, rescaling_factor, centers: pn.DataFrame):
        new_grid_extent = (grid.extent - np.repeat(centers, 2)) / rescaling_factor + 0.5001
        new_grid_values = (grid.values - centers) / rescaling_factor + 0.5001
        return new_grid_extent, new_grid_values

    def set_rescaled_grid(self):
        """
             Set the rescaled coordinates and extent into a grid object
        """

        self.grid.extent_r, self.grid.values_r = self.rescale_grid(self.grid, self.df.loc['values', 'rescaling factor'],
                                                                   self.df.loc['values', 'centers'])


class Structure(object):
    """
    The structure_data class analyse the different lenths of subset in the interface and orientations categories_df to pass them to
    the theano function.

    Attributes:

        len_surfaces_i (list): length of each surface/fault in surface_points
        len_series_i (list) : length of each series in surface_points
        len_series_o (list) : length of each series in orientations
        nfs (list): number of surfaces per series
        ref_position (list): location of the first point of each surface/fault in interface

    Args:
        surface_points (SurfacePoints)
        orientations (Orientations)
    """

    def __init__(self, surface_points: SurfacePoints, orientations: Orientations, surfaces: Surfaces, faults: Faults):

        self.surface_points = surface_points
        self.orientations = orientations
        self.surfaces = surfaces
        self.faults = faults

        df_ = pn.DataFrame(np.array(['False', 'False', -1, -1, -1, -1, -1, -1, -1],).reshape(1,-1),
                           index=['values'],
                           columns=['isLith', 'isFault',
                                    'number faults', 'number surfaces', 'number series',
                                    'number surfaces per series',
                                    'len surfaces surface_points', 'len series surface_points',
                                    'len series orientations'])

        self.df = df_.astype({'isLith': bool, 'isFault': bool, 'number faults': int,
                              'number surfaces': int, 'number series':int})

        self.update_structure_from_input()

    def __repr__(self):
        return self.df.T.to_string()

    def _repr_html_(self):
        return self.df.T.to_html()

    def modify_rescaling_parameters(self, property, value):
        assert self.df.columns.isin(property), 'Valid properties are: ' + np.array2string(self.df.columns)
        self.df.loc['values', property] = value

    def update_structure_from_input(self):
        self.set_length_surfaces_i()
        self.set_series_and_length_series_i()
        self.set_length_series_o()
        self.set_number_of_surfaces_per_series()
        self.set_number_of_faults()
        self.set_number_of_surfaces()
        self.set_is_lith_is_fault()

    def set_length_surfaces_i(self):
        # ==================
        # Extracting lengths
        # ==================
        # Array containing the size of every surface. SurfacePoints
        lssp = self.surface_points.df.groupby('surface')['order_series'].count().values
        lssp_nonzero = lssp[np.nonzero(lssp)]

        self.df.at['values', 'len surfaces surface_points'] = lssp_nonzero#self.surface_points.df['id'].value_counts(sort=False).values

        return True

    def set_series_and_length_series_i(self):
        # Array containing the size of every series. SurfacePoints.
        len_series_i = self.surface_points.df['order_series'].value_counts(sort=False).values

        if len_series_i.shape[0] is 0:
            len_series_i = np.insert(len_series_i, 0, 0)

        self.df.at['values','len series surface_points'] = len_series_i
        self.df['number series'] = len(len_series_i)
        return True

    def set_length_series_o(self):
        # Array containing the size of every series. orientations.
        self.df.at['values', 'len series orientations'] = self.orientations.df['order_series'].value_counts(sort=False).values
        return True

    def set_ref_position(self):
        # TODO DEP? Ah ja, this is what is done now in theano
        self.ref_position = np.insert(self.len_surfaces_i[:-1], 0, 0).cumsum()
        return self.ref_position

    def set_number_of_surfaces_per_series(self):
        self.df.at['values', 'number surfaces per series'] = self.surface_points.df.groupby('order_series').surface.nunique().values
        return True

    def set_number_of_faults(self):
        # Number of faults existing in the surface_points df
        self.df.at['values', 'number faults'] = self.faults.df.loc[self.surface_points.df['series'].unique(), 'isFault'].sum()
        return True

    def set_number_of_surfaces(self):
        # Number of surfaces existing in the surface_points df
        self.df.at['values', 'number surfaces'] = self.surface_points.df['surface'].nunique()

        return True

    def set_is_lith_is_fault(self):
        """
         TODO Update string
        Check if there is lithologies in the data and/or df
        Returns:
            list(bool)
        """
        self.df['isLith'] = True if self.df.loc['values', 'number series'] >= self.df.loc['values', 'number faults'] else False
        self.df['isFault'] = True if self.df.loc['values', 'number faults'] > 0 else False

        return True


class Options(object):
    def __init__(self):
        df_ = pn.DataFrame(np.array(['float64', 'geology', 'fast_compile', 'cpu', None]).reshape(1, -1),
                           index=['values'],
                           columns=['dtype', 'output', 'theano_optimizer', 'device', 'verbosity'])
        self.df = df_.astype({'dtype': 'category', 'output' : 'category',
                              'theano_optimizer' : 'category', 'device': 'category',
                              'verbosity': object})

        self.df['dtype'].cat.set_categories(['float32', 'float64'], inplace=True)
        self.df['theano_optimizer'].cat.set_categories(['fast_run', 'fast_compile'], inplace=True)
        self.df['device'].cat.set_categories(['cpu', 'cuda'], inplace=True)
        self.df['output'].cat.set_categories(['geology', 'gradients'], inplace=True)
        self.df.at['values', 'verbosity'] = []

    def __repr__(self):
        return self.df.T.to_string()

    def _repr_html_(self):
        return self.df.T.to_html()

    def modify_options(self, property, value):
        assert np.isin(property, self.df.columns).all(), 'Valid properties are: ' + np.array2string(self.df.columns)
        self.df.loc['values', property] = value

    def default_options(self):
        """
        Set default options.

        Returns:

        """
        self.df['dtype'] = 'float64'
        self.df['output'] = 'geology'
        self.df['theano_optimizer'] = 'fast_compile'
        self.df['device'] = 'cpu'


class KrigingParameters(object):
    def __init__(self, grid: Grid, structure: Structure):
        self.structure = structure
        self.grid = grid

        df_ = pn.DataFrame(np.array([np.nan, np.nan, 3, 0.01, 1e-6]).reshape(1, -1),
                               index=['values'],
                               columns=['range', '$C_o$', 'drift equations',
                                        'nugget grad', 'nugget scalar'])

        self.df = df_.astype({'drift equations': object})
        self.set_default_range()
        self.set_default_c_o()
        self.set_u_grade()

    def __repr__(self):
        return self.df.T.to_string()

    def _repr_html_(self):
        return self.df.T.to_html()

    def modify_kriging_parameters(self, property:str, value, **kwargs):
        u_grade_sep = kwargs.get('u_grade_sep', ',')

        assert np.isin(property, self.df.columns).all(), 'Valid properties are: ' + np.array2string(self.df.columns)

        if property == 'drift equations':
            if type(value) is str:
                value = np.fromstring(value[1:-1], sep=u_grade_sep)
            try:
                assert value.shape[0] is self.structure.df.loc[
                    'values', 'len series surface_points'].shape[0]

                self.df.at['values', property] = value

            except AssertionError:
                print('u_grade length must be the same as the number of series')

        else:
            self.df.loc['values', property] = value

    def set_default_range(self, extent=None):
        """
        Set default kriging_data range
        Args:
            extent:

        Returns:

        """
        if extent is None:
            extent = self.grid.extent
        try:
            range_var = np.sqrt(
                (extent[0] - extent[1]) ** 2 +
                (extent[2] - extent[3]) ** 2 +
                (extent[4] - extent[5]) ** 2)
        except TypeError:
            warnings.warn('The extent passed or if None the extent of the grid object has some type of problem',
                          TypeError)
            range_var = np.nan

        self.df['range'] = range_var

        return range_var

    def set_default_c_o(self, range_var=None):
        if range_var is None:
            range_var = self.df['range']

        self.df['$C_o$'] = range_var ** 2 / 14 / 3
        return self.df['$C_o$']

    def set_u_grade(self, u_grade: list = None):
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

            len_series_i = self.structure.df.loc['values', 'len series surface_points']
            u_grade = np.zeros_like(len_series_i)
            u_grade[(len_series_i > 1)] = 1

        else:
            u_grade = np.array(u_grade)

        # Transformin grade to number of equations
        n_universal_eq = np.zeros_like(u_grade)
        n_universal_eq[u_grade == 0] = 0
        n_universal_eq[u_grade == 1] = 3
        n_universal_eq[u_grade == 2] = 9

        self.df.at['values', 'drift equations'] = n_universal_eq
        return self.df['drift equations']


class AdditionalData(object):
    def __init__(self, surface_points: SurfacePoints, orientations: Orientations, grid: Grid,
                 faults: Faults, surfaces: Surfaces, rescaling: RescaledData):

        self.structure_data = Structure(surface_points, orientations, surfaces, faults)
        self.options = Options()
        self.kriging_data = KrigingParameters(grid, self.structure_data)
        self.rescaling_data = rescaling

    def __repr__(self):

        concat_ = self.get_additional_data()
        return concat_.to_string()

    def _repr_html_(self):
        concat_ = self.get_additional_data()
        return concat_.to_html()

    def get_additional_data(self):
        concat_ = pn.concat([self.structure_data.df, self.options.df, self.kriging_data.df, self.rescaling_data.df],
                            axis=1, keys=['Structure', 'Options', 'Kringing', 'Rescaling'])
        return concat_.T

    # def update_rescaling_data(self):
    #     #TODO check uses and if they are still relevant
    #     self.rescaling_data['values', 'rescaling factor'] = self.rescaling_data.df['rescaling factor']
    #     self.rescaling_data['values', 'centers'] = self.rescaling_data.df['centers']

    def update_default_kriging(self):
        self.kriging_data.set_default_range()
        self.kriging_data.set_default_c_o()
        self.kriging_data.set_u_grade()
        self.kriging_data.df['nugget grad'] = 0.01
        self.kriging_data.df['nugget scalar'] = 1e-6

    def update_structure(self):
        self.structure_data.update_structure_from_input()


class Solution(object):
    """
    TODO: update this
    This class store the output of the interpolation and the necessary objects to visualize and manipulate this data.
    Depending on the chosen output in Additional Data -> Options a different number of solutions is returned:
        if output is geology:
            1) Lithologies: block and scalar field
            2) Faults: block and scalar field for each faulting plane

        if output is gradients:
            1) Lithologies: block and scalar field
            2) Faults: block and scalar field for each faulting plane
            3) Gradients of scalar field x
            4) Gradients of scalar field y
            5) Gradients of scalar field z

    Attributes:
        additional_data (AdditionalData):
        surfaces (Surfaces)
        grid (Grid)
        scalar_field_at_surface_points (np.ndarray): Array containing the values of the scalar field at each interface. Axis
        0 is each series and axis 1 contain each surface in order
         lith_block (np.ndndarray): Array with the id of each layer evaluated in each point of
         `attribute:GridClass.values`
        fault_block (np.ndarray): Array with the id of each fault block evaluated in each point of
         `attribute:GridClass.values`
        scalar_field_lith(np.ndarray): Array with the scalar field of each layer evaluated in each point of
         `attribute:GridClass.values`
        scalar_field_lith(np.ndarray): Array with the scalar field of each fault segmentation evaluated in each point of
        `attribute:GridClass.values`
        values_block (np.ndarray):   Array with the properties of each layer evaluated in each point of
         `attribute:GridClass.values`. Axis 0 represent different properties while axis 1 contain each evaluated point
        gradient (np.ndarray):  Array with the gradient of the layars evaluated in each point of
        `attribute:GridClass.values`. Axis 0 contain Gx, Gy, Gz while axis 1 contain each evaluated point

    Args:
        additional_data (AdditionalData):
        surfaces (Surfaces):
        grid (Grid):
        values (np.ndarray): values returned by `function: gempy.compute_model` function
    """

    def __init__(self, additional_data: AdditionalData = None, grid: Grid = None,
                 surface_points: SurfacePoints = None, values=None):

        self.additional_data = additional_data
        self.grid = grid
        self.surface_points = surface_points

        if values is None:

            self.scalar_field_at_surface_points = np.array([])
            self.scalar_field_lith = np.array([])
            self.scalar_field_faults = np.array([])

            self.lith_block = np.empty(0)
            self.fault_blocks = np.empty(0)
            self.values_block = np.empty(0)
            self.gradient = np.empty(0)
        else:
            self.set_values(values)

        self.vertices = {}
        self.edges = {}

    def __repr__(self):
        return '\nLithology ids \n  %s \n' \
               'Lithology scalar field \n  %s \n' \
               'Fault block \n  %s' \
               % (np.array2string(self.lith_block), np.array2string(self.scalar_field_lith),
                  np.array2string(self.fault_blocks))

    def set_values(self, values: Union[list, np.ndarray], compute_mesh: bool=True):
        # TODO ============ Set asserts of give flexibility 20.09.18 =============
        """
        Set all solution values to the correspondant attribute
        Args:
            values (np.ndarray): values returned by `function: gempy.compute_model` function
            compute_mesh (bool): if true compute automatically the grid

        Returns:

        """
        lith = values[0]
        faults = values[1]
        self.scalar_field_at_surface_points = values[2]

        self.scalar_field_lith = lith[1]
        self.lith_block = lith[0]
        self.values_block = lith[1:]

        try:
            if self.additional_data.options.df.loc['values', 'output'] is 'gradients':
                self.values_block = lith[2:-3]
                self.gradient = lith[-3:]
            else:
                self.values_block = lith[2:]
        except AttributeError:
            self.values_block = lith[2:]

        self.scalar_field_faults = faults[1::2]
        self.fault_blocks = faults[::2]
        assert len(np.atleast_2d(
            self.scalar_field_faults)) == self.additional_data.structure_data.df.loc['values', 'number faults'], \
            'The number of df computed does not match to the number of df in the input data.'

        # TODO I do not like this here
        if compute_mesh is True:
            try:
                self.compute_all_surfaces()
            except RuntimeError:
                warnings.warn('It is not possible to compute the mesh.')

    def compute_surface_regular_grid(self, surface_id: int, scalar_field, **kwargs):
        """
        Compute the surface (vertices and edges) of a given surface by computing marching cubes (by skimage)
        Args:
            surface_id (int): id of the surface to be computed
            scalar_field: scalar field grid
            **kwargs: skimage.measure.marching_cubes_lewiner args

        Returns:
            list: vertices, simplices, normals, values
        """

        from skimage import measure
        assert surface_id >= 0, 'Number of the surface has to be positive'
        # In case the values are separated by series I put all in a vector
        pot_int = self.scalar_field_at_surface_points.sum(axis=0)

        # Check that the scalar field of the surface is whithin the boundaries
        if not scalar_field.max() > pot_int[surface_id]:
            pot_int[surface_id] = scalar_field.max()
            print('Scalar field value at the surface %i is outside the grid boundaries. Probably is due to an error'
                  'in the implementation.' % surface_id)

        if not scalar_field.min() < pot_int[surface_id]:
            pot_int[surface_id] = scalar_field.min()
            print('Scalar field value at the surface %i is outside the grid boundaries. Probably is due to an error'
                  'in the implementation.' % surface_id)

        vertices, simplices, normals, values = measure.marching_cubes_lewiner(
            scalar_field.reshape(self.grid.resolution[0],
                                 self.grid.resolution[1],
                                 self.grid.resolution[2]),
            pot_int[surface_id],
            spacing=((self.grid.extent[1] - self.grid.extent[0]) / self.grid.resolution[0],
                     (self.grid.extent[3] - self.grid.extent[2]) / self.grid.resolution[1],
                     (self.grid.extent[5] - self.grid.extent[4]) / self.grid.resolution[2]),
            **kwargs
        )

        return [vertices, simplices, normals, values]

    @_setdoc(compute_surface_regular_grid.__doc__)
    def compute_all_surfaces(self, **kwargs):
        """
        Compute all surfaces.

        Args:
            **kwargs: Marching_cube args

        Returns:

        See Also:
        """
        n_surfaces = self.additional_data.structure_data.df.loc['values', 'number surfaces']
        n_faults = self.additional_data.structure_data.df.loc['values', 'number faults']

        surfaces_names = self.surface_points.df['surface'].unique()

        surfaces_cumsum = np.arange(0, n_surfaces)

        if n_faults > 0:
            for n in surfaces_cumsum[:n_faults]:
                v, s, norm, val = self.compute_surface_regular_grid(n, np.atleast_2d(self.scalar_field_faults)[n],
                                                                    **kwargs)
                self.vertices[surfaces_names[n]] = v
                self.edges[surfaces_names[n]] = s

        if n_faults < n_surfaces:

            for n in surfaces_cumsum[n_faults:]:
                # TODO ======== split each_scalar_field ===========
                v, s, norms, val = self.compute_surface_regular_grid(n, self.scalar_field_lith, **kwargs)

                # TODO Use setters for this
                self.vertices[surfaces_names[n]] = v
                self.edges[surfaces_names[n]] = s

        return self.vertices, self.edges

    def set_vertices(self, surface_name, vertices):
        self.vertices[surface_name] = vertices

    def set_edges(self, surface_name, edges):
        self.edges[surface_name] = edges
