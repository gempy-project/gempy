import sys
from os import path

import numpy as np
import pandas as pn
from typing import Union
import warnings
import re
try:
    import ipywidgets as widgets
    ipywidgets_import = True
except ModuleNotFoundError:
    VTK_IMPORT = False

pn.options.mode.chained_assignment = None

# This is for sphenix to find the packages
#sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from gempy.core.grid_modules import grid_types
from gempy.core.checkers import check_for_nans
from gempy.utils.meta import _setdoc
from gempy.plot.sequential_pile import StratigraphicPile


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
    def __init__(self, **kwargs):

        extent = kwargs.get('extent', [0, 1000, 0, 1000, -1000, 0])

        self.extent = np.atleast_1d(extent)
        self.values = np.empty((0, 3))
        self.values_r = np.empty((0, 3))
        self.length = np.empty(0)
        self.grid_types = np.array(['regular', 'custom', 'topography', 'gravity'])
        self.active_grids = np.zeros(4, dtype=bool)
        # All grid types must have values

        # Init optional grids
        self.custom_grid = None
        self.custom_grid_grid_active = False
        self.topography = None
        self.topography_grid_active = False
        self.gravity_grid = None
        self.gravity_grid_active = False

        # Init basic grid empty
        self.regular_grid = self.set_regular_grid(**kwargs)
        self.regular_grid_active = False

    def __str__(self):
        return 'Grid Object. Values: \n' + np.array2string(self.values)

    def __repr__(self):
        return 'Grid Object. Values: \n' + np.array_repr(self.values)

    def set_regular_grid(self, *args, **kwargs):

        self.regular_grid = grid_types.RegularGrid(*args, **kwargs)
        if 'extent' in kwargs:
            self.extent = np.atleast_1d(kwargs['extent'])

        self.set_active('regular')
        return self.regular_grid

    def set_custom_grid(self, custom_grid: np.ndarray):
        self.custom_grid = grid_types.CustomGrid(custom_grid)
        self.set_active('custom')

    def set_topography(self, source='random', **kwargs):
        self.topography = grid_types.Topography(self.regular_grid)

        if source == 'random':
            self.topography.load_random_hills(**kwargs)
        elif source == 'gdal':
            filepath = kwargs.get('filepath', None)
            if filepath is not None:
                self.topography.load_from_gdal(filepath)
            else:
                print('to load a raster file, a path to the file must be provided')
        elif source == 'npy':
            filepath = kwargs.get('filepath', None)
            if filepath is not None:
                self.topography.load_from_saved(filepath)
            else:
                print('path to .npy file must be provided')
        else:
            print('source must be random, gdal or npy')

        self.topography.show()
        self.set_active('topography')

    def set_gravity_grid(self):
        self.gravity_grid = grid_types.GravityGrid()
        self.active_grids = np.zeros(4, dtype=bool)
        self.set_active('gravity')

    def deactivate_all_grids(self):
        self.active_grids = np.zeros(4, dtype=bool)
        self.update_grid_values()
        return self.active_grids

    def set_active(self, grid_name: Union[str, np.ndarray]):
        where = self.grid_types == grid_name
        self.active_grids += where
        self.update_grid_values()

    def set_inactive(self, grid_name: str):
        where = self.grid_types == grid_name
        self.active_grids -= where
        self.update_grid_values()

    def update_grid_values(self):
        self.length = np.empty(0)
        self.values = np.empty((0, 3))
        lengths = [0]
        try:
            for e, grid_types in enumerate([self.regular_grid, self.custom_grid, self.topography, self.gravity_grid]):
                if self.active_grids[e]:
                    self.values = np.vstack((self.values, grid_types.values))
                    lengths.append(grid_types.values.shape[0])
                else:
                    lengths.append(0)
        except AttributeError:
            raise AttributeError('Grid type do not exist yet. Set the grid before activate it.')

        self.length = np.array(lengths).cumsum()

    def get_grid_args(self, grid_name: str):
        assert type(grid_name) is str, 'Only one grid type can be retrieve'
        assert grid_name in self.grid_types, 'possible grid types are ' + str(self.grid_types)
        where = np.where(self.grid_types == grid_name)[0][0]
        return self.length[where], self.length[where+1]

    def get_grid(self, grid_name: str):
        assert type(grid_name) is str, 'Only one grid type can be retrieve'

        l_0, l_1 = self.get_grid_args(grid_name)
        return self.values[l_0:l_1]


class Series(object):
    """
    Series is a class that contains the relation between series/df and each individual surface/layer. This can be
    illustrated in the sequential pile.

    Args:
        faults: (dict or :class:`pn.core.frame.DataFrames`): with the name of the serie as key and the
         name of the surfaces as values.
        series_order(Optional[list]): order of the series by default takes the dictionary keys which until python 3.6 are
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
        self.df['BottomRelation'] = pn.Categorical(['Erosion'], categories=['Erosion', 'Onlap', 'Fault'])

    def __repr__(self):
        return self.df.to_string()

    def _repr_html_(self):
        return self.df.to_html()

    def update_order_series(self):
        """
        Index of df is categorical and order, but we need to numerate that order to map it later on to the Data dfs
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
            self.faults.df.loc[c] = [False, False]
            self.faults.faults_relations_df.loc[c, c] = False

        self.faults.faults_relations_df.fillna(False, inplace=True)

        if update_order_series is True:
            self.update_order_series()

    def set_bottom_relation(self, series: Union[str, list], bottom_relation: Union[str, list]):
        self.df.loc[series, 'BottomRelation'] = bottom_relation

        if self.faults.df.loc[series, 'isFault'] is True:
            self.faults.set_is_fault(series, toggle=True)

        elif bottom_relation == 'Fault':
            self.faults.df.loc[series, 'isFault'] = True

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
    def reorder_series(self, new_categories: Union[list, np.ndarray]):
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

        #  This is a hack for qgrid
        #  We need to add the qgrid special columns to categories
        self.faults.faults_relations_df.columns = self.faults.faults_relations_df.columns.add_categories(
            ['index', 'qgrid_unfiltered_index'])


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

        self.faults_relations_df = pn.DataFrame(index=pn.CategoricalIndex(['Default series']),
                                                columns=pn.CategoricalIndex(['Default series', '']), dtype='bool')

        self.set_is_fault(series_fault=series_fault)
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

    def set_is_fault(self, series_fault: Union[str, list, np.ndarray] = None, toggle=False):
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
            if toggle is True:
                self.df.loc[series_fault, 'isFault'] = self.df.loc[series_fault, 'isFault'] ^ True
            else:
                self.df.loc[series_fault, 'isFault'] = True
            self.df['isFinite'] = np.bitwise_and(self.df['isFault'], self.df['isFinite'])
            # Update default fault relations
            for a_series in series_fault:
                col_pos = self.faults_relations_df.columns.get_loc(a_series)
                self.faults_relations_df.iloc[col_pos, col_pos + 1:] = True

        self.n_faults = self.df['isFault'].sum()

        return self.df

    def set_is_finite_fault(self, series_fault=None, toggle=False):
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
            if toggle is True:
                self.df.loc[series_fault, 'isFinite'] = self.df.loc[series_fault, 'isFinite'] ^ True
            else:
                self.df.loc[series_fault, 'isFinite'] = self.df.loc[series_fault, 'isFinite']

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

        self.faults_relations_df.iloc[np.tril(np.ones(self.df.index.shape[0])).astype(bool)] = False

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


class Colors:
    def __init__(self, surfaces):
        self.surfaces = surfaces

    def generate_colordict_DEP(self, out = False):
        '''generate colordict that assigns black to faults and random colors to surfaces'''
        gp_defcols = [
            ['#325916', '#5DA629', '#78CB68', '#84C47A', '#129450'],
            ['#F2930C', '#F24C0C', '#E00000', '#CF522A', '#990902'],
            ['#26BEFF', '#227dac', '#443988', '#2A186C', '#0F5B90']]

        for i, series in enumerate(self.surfaces.df['series'].unique()):
            form_in_series = list(self.surfaces.df.loc[self.surfaces.df['series'] == series, 'surface'])
            newcols = gp_defcols[i][:len(form_in_series)]
            if i == 0:
                colordict = dict(zip(form_in_series, newcols))
            else:
                colordict.update(zip(form_in_series, newcols))
        if out:
            return colordict
        else:
            self.colordict = colordict

    def generate_colordict(self, out = False):
        """generate colordict that assigns black to faults and random colors to surfaces"""
        gp_defcols = ['#015482','#9f0052','#ffbe00','#728f02','#443988','#ff3f20','#325916','#5DA629']
        test = len(gp_defcols) >= len(self.surfaces.df)

        if test is False:
            from matplotlib._color_data import XKCD_COLORS as morecolors
            gp_defcols += list(morecolors.values())

        colordict = dict(zip(list(self.surfaces.df['surface']), gp_defcols[:len(self.surfaces.df)]))
        self.colordict_default = colordict
        if out:
            return colordict
        else:
            self.colordict = colordict

    def change_colors(self, cdict = None):
        ''' Updates the colors in self.colordict and in surfaces_df.
        Args:
            cdict: dict with surface names mapped to hex color codes, e.g. {'layer1':'#6b0318'}
            if None: opens jupyter widget to change colors interactively.

        Returns: None

        '''
        assert ipywidgets_import, 'ipywidgets not imported. Make sure the library is installed.'

        if cdict is not None:
            self._update_colors(cdict)
            return self.surfaces

        else:
            items = [widgets.ColorPicker(description=surface, value=color)
                     for surface, color in self.colordict.items()]

            colbox = widgets.VBox(items)
            print('Click to select new colors.')
            display(colbox)

            def on_change(v):
                self.colordict[v['owner'].description] = v['new']  # update colordict
                self._set_colors()

            for cols in colbox.children:
                cols.observe(on_change, 'value')

    def _update_colors(self, cdict=None):
        ''' Updates the colors in self.colordict and in surfaces_df.
        Args:
            cdict: dict with surface names mapped to hex color codes, e.g. {'layer1':'#6b0318'}

        Returns: None

        '''
        if cdict == None:
            # assert if one surface does not have color
            try:
                self._add_colors()
            except AttributeError:
                self.generate_colordict()
        else:
            for surf, color in cdict.items(): # map new colors to surfaces
                # assert this because user can set it manually
                assert surf in list(self.surfaces.df['surface']), str(surf) + ' is not a model surface'
                assert re.search(r'^#(?:[0-9a-fA-F]{3}){1,2}$', color), str(color) + ' is not a HEX color code'
                self.colordict[surf] = color

        self._set_colors()

    def _add_colors(self):
        '''assign color to last entry of surfaces df or check isnull and assign color there'''
        # can be done easier
        new_colors = self.generate_colordict(out=True)
        form2col = list(self.surfaces.df.loc[self.surfaces.df['color'].isnull(), 'surface'])
        # this is the dict in-build function to update colors
        self.colordict.update(dict(zip(form2col, [new_colors[x] for x in form2col])))

    def _set_colors(self):
        '''sets colordict in surfaces dataframe'''
        for surf, color in self.colordict.items():
            self.surfaces.df.loc[self.surfaces.df['surface'] == surf, 'color'] = color

    def set_default_colors(self, surfaces = None):
        if surfaces is not None:
            self.colordict[surfaces] = self.colordict_default[surfaces]
        self._set_colors()

    def make_faults_black(self, series_fault):
        faults_list = list(self.surfaces.df[self.surfaces.df.series.isin(series_fault)]['surface'])
        for fault in faults_list:
            if self.colordict[fault] == '#527682':
                self.set_default_colors(fault)
            else:
                self.colordict[fault] = '#527682'
                self._set_colors()

    def reset_default_colors(self):
        self.generate_colordict()
        self._set_colors()
        return self.surfaces


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

        self._columns = ['surface', 'series', 'order_surfaces', 'isBasement', 'color', 'vertices', 'edges', 'id']
        self._columns_vis_drop = ['vertices', 'edges',]
        self._n_properties = len(self._columns) -1
        self.series = series
        self.colors = Colors(self)

        df_ = pn.DataFrame(columns=self._columns)
        self.df = df_.astype({'surface': str, 'series': 'category',
                              'order_surfaces': int, 'isBasement': bool,
                              'color': bool, 'id': int, 'vertices': object, 'edges': object})

        if (np.array(sys.version_info[:2]) <= np.array([3, 6])).all():
            self.df: pn.DataFrame

        self.df['series'].cat.add_categories(['Default series'], inplace=True)
        if surface_names is not None:
            self.set_surfaces_names(surface_names)

        if values_array is not None:
            self.set_surfaces_values(values_array=values_array, properties_names=properties_names)

      #  self.sequential_pile = StratigraphicPile(self.series, self.df)

    def __repr__(self):
        return self.df.to_string()

    def _repr_html_(self):
        #return self.df.to_html()
        c_ = self.df.columns[~(self.df.columns.isin(self._columns_vis_drop))]

        return self.df[c_].style.applymap(self.background_color, subset=['color']).render()

    def background_color(self, value):
        if type(value) == str:
            return "background-color: %s" % value

    def update_sequential_pile(self):
        """
        Method to update the sequential pile plot
        Returns:

        """
        pass #self.sequential_pile = StratigraphicPile(self.series, self.df)

# region set formation names
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
            self.update_id()
            self.set_basement()
            self.update_order_surfaces()
            self.colors._update_colors()
            self.update_sequential_pile()
        return True

    def set_default_surface_name(self):
        if self.df.shape[0] == 0:
            # TODO DEBUG: I am not sure that surfaces always has at least one entry. Check it
            self.set_surfaces_names(['surface1', 'basement'])
        return self

    def set_surfaces_names_from_surface_points(self, surface_points):
        self.set_surfaces_names(surface_points.df['surface'].unique())

    def add_surface(self, surface_list: Union[str, list], update_df=True):
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
            self.update_id()
            self.set_basement()
            self.update_order_surfaces()
            self.colors._update_colors()
            #self.update_sequential_pile()
        return True

    def delete_surface(self, indices: Union[int, str, list, np.ndarray], update_id=True):

        indices = np.atleast_1d(indices)
        if indices.dtype == int:
            self.df.drop(indices, inplace=True)
        else:
            self.df.drop(self.df.index[self.df['surface'].isin(indices)], inplace=True)
        if update_id is True:
            self.update_id()
            self.set_basement()
            self.update_order_surfaces()
           # self.update_sequential_pile()
        return True

    @_setdoc(pn.Series.replace.__doc__)
    def rename_surfaces(self, to_replace:Union[str, list, dict],  **kwargs):
        if np.isin(to_replace, self.df['surface']).any():
            print('Two surfaces cannot have the same name.')
        else:
            self.df['surface'].replace(to_replace,  inplace=True, **kwargs)
        return True

    def update_order_surfaces(self):
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
        self.update_id()
        return self.df

    def set_basement(self):
        """


        Returns:
            True
        """

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
        self.update_order_surfaces()
        self.sort_surfaces()
        self.set_basement()

# endregion

# region update_id
    def update_id(self, id_list: list = None):
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

    def add_surfaces_values(self, values_array: Union[np.ndarray, list], properties_names: list = np.empty(0)):
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

    def set_surfaces_values(self, values_array: Union[np.ndarray, list], properties_names: list = np.empty(0)):
        # Check if there are values columns already
        old_prop_names = self.df.columns[~self.df.columns.isin(['surface', 'series', 'order_surfaces',
                                                                'id', 'isBasement', 'color'])]
        # Delete old
        self.delete_surface_values(old_prop_names)

        # Create new
        self.add_surfaces_values(values_array, properties_names)
        return self

    def modify_surface_values(self, idx, properties_names, values):
        """Method to modify values using loc of pandas"""
        properties_names = np.atleast_1d(properties_names)
        assert ~np.isin(properties_names, ['surface', 'series', 'order_surfaces', 'id', 'isBasement', 'color']),\
            'only property names can be modified with this method'

        self.df.loc[idx, properties_names] = values


class GeometricData(object):
    """
    Parent class of the objects which contatin the input parameters: surface_points and orientations. This class contain
    the common methods for both types of data sets.
    """

    def __init__(self, surfaces: Surfaces):

        self.surfaces = surfaces
        self.df = pn.DataFrame()

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
        self.df.sort_values(by=['order_series', 'id'],
                            ascending=True, kind='mergesort',
                            inplace=True)
        return self.df

    def map_data_from_series(self, series, property:str, idx=None):
        """

        """
        if idx is None:
            idx = self.df.index

        idx = np.atleast_1d(idx)
        self.df.loc[idx, property] = self.df['series'].map(series.df[property])

    def set_series_categories_from_series(self, series: Series):
        self.df['series'].cat.set_categories(series.df.index, inplace=True)
        return True

    def set_surface_categories_from_surfaces(self, surfaces: Surfaces):
        self.df['surface'].cat.set_categories(surfaces.df['surface'], inplace=True)
        return True

    def map_data_from_surfaces(self, surfaces, property:str, idx=None):
        """Map properties of surfaces---series, id, values--- into a data df"""

        if idx is None:
            idx = self.df.index
        idx = np.atleast_1d(idx)
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
        idx = np.atleast_1d(idx)
        if any(self.df['series'].isna()):
            warnings.warn('Some points do not have series/fault')

        self.df.loc[idx, 'isFault'] = self.df.loc[[idx], 'series'].map(faults.df['isFault'])


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

    def add_surface_points(self, X, Y, Z, surface, idx:Union[int, list, np.ndarray] = None):
        # TODO: Add the option to pass the surface number

        if idx is None:
            idx = self.df.index.max()
            if idx is np.nan:
                idx = 0
            else:
                idx += 1

        coord_array = np.array([X, Y, Z])
        assert coord_array.ndim == 1, 'Adding an interface only works one by one.'
        self.df.loc[idx, ['X', 'Y', 'Z']] = coord_array

        try:
            self.df.loc[idx, 'surface'] = surface
        # ToDO test this
        except ValueError as error:
            self.del_surface_points(idx)
            print('The surface passed does not exist in the pandas categories. This may imply that'
                  'does not exist in the surface object either.')
            raise ValueError(error)

        self.map_data_from_surfaces(self.surfaces, 'series', idx=idx)
        self.map_data_from_surfaces(self.surfaces, 'id', idx=idx)
        self.map_data_from_series(self.surfaces.series, 'order_series', idx=idx)

        self.sort_table()
        return self, idx

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
        idx = np.array(idx, ndmin=1)
        keys = list(kwargs.keys())
        is_surface = np.isin('surface', keys).all()

        # Check idx exist in the df
        assert np.isin(np.atleast_1d(idx), self.df.index).all(), 'Indices must exist in the dataframe to be modified.'

        # Check the properties are valid
        assert np.isin(list(kwargs.keys()), ['X', 'Y', 'Z', 'surface']).all(),\
            'Properties must be one or more of the following: \'X\', \'Y\', \'Z\', ' '\'surface\''
        # stack properties values
        values = np.array(list(kwargs.values()))

        # If we pass multiple index we need to transpose the numpy array
        if type(idx) is list or type(idx) is np.ndarray:
            values = values.T

        # Selecting the properties passed to be modified
        self.df.loc[idx, list(kwargs.keys())] = values

        if is_surface:
            self.map_data_from_surfaces(self.surfaces, 'series', idx=idx)
            self.map_data_from_surfaces(self.surfaces, 'id', idx=idx)
            self.map_data_from_series(self.surfaces.series, 'order_series', idx=idx)
            self.sort_table()

    def read_surface_points(self, file_path, debug=False, inplace=False,
                            kwargs_pandas:dict = {}, **kwargs, ):
        """
        Read tabular using pandas tools and if inplace set it properly to the surface points object
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

    def update_annotations(self):
        """
        Add a column in the Dataframes with latex names for each input_data paramenter.

        Returns:
            None
        """
        point_num = self.df.groupby('id').cumcount()
        point_l = [r'${\bf{x}}_{\alpha \,{\bf{' + str(f) + '}},' + str(p) + '}$'
                   for p, f in zip(point_num, self.df['id'])]

        self.df['annotations'] = point_l
        return self


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
            else:
                idx += 1

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

        self.map_data_from_surfaces(self.surfaces, 'series', idx=idx)
        self.map_data_from_surfaces(self.surfaces, 'id', idx=idx)
        self.map_data_from_series(self.surfaces.series, 'order_series', idx=idx)

        self.sort_table()

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

        idx = np.array(idx, ndmin=1)
        keys = list(kwargs.keys())
        is_surface = np.isin('surface', keys).all()

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

        if is_surface:
            self.map_data_from_surfaces(self.surfaces, 'series', idx=idx)
            self.map_data_from_surfaces(self.surfaces, 'id', idx=idx)
            self.map_data_from_series(self.surfaces.series, 'order_series', idx=idx)
            self.sort_table()

    def calculate_gradient(self, idx=None):
        """
        Calculate the gradient vector of module 1 given dip and azimuth to be able to plot the orientations
        """
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

    def create_orientation_from_interface(self, surface_points: SurfacePoints, indices):
        # TODO test!!!!
        """
        Create and set orientations from at least 3 points categories_df
        Args:
            surface_points
            indices
        """
        selected_points = surface_points.df[['X', 'Y', 'Z']].loc[indices].values.T

        center, normal = self.plane_fit(selected_points)
        orientation = self.get_orientation(normal)

        return np.array([*center, *orientation, *normal])

    def set_default_orientation(self):
        """
        Set a default point at the middle of the extent area to be able to start making the model
        """
        if self.df.shape[0] == 0:
            self.add_orientation(.00001, .00001, .00001,
                                 self.surfaces.df['surface'].iloc[0],
                                 [0, 0, 1],
                                 )

    def read_orientations(self, filepath, debug=False, inplace=True, kwargs_pandas = {}, **kwargs):
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

    def update_annotations(self):
        """
        Add a column in the Dataframes with latex names for each input_data paramenter.

        Returns:
            None
        """

        orientation_num = self.df.groupby('id').cumcount()
        foli_l = [r'${\bf{x}}_{\beta \,{\bf{' + str(f) + '}},' + str(p) + '}$'
                  for p, f in zip(orientation_num, self.df['id'])]

        self.df['annotations'] = foli_l

    @staticmethod
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

    @staticmethod
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

        self.df = pn.DataFrame(np.array([rescaling_factor, centers]).reshape(1, -1),
                               index=['values'],
                               columns=['rescaling factor', 'centers'])

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
        return self.surface_points.df[['X_r', 'Y_r', 'Z_r']],

    def get_rescaled_orientations(self):
        """
        Get the rescaled coordinates. return an image of the interface and orientations categories_df with the X_r..
         columns
        Returns:

        """
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

    def set_rescaled_surface_points(self, idx: Union[list, np.ndarray] = None):
        """
        Set the rescaled coordinates into the surface_points categories_df
        Returns:

        """
        if idx is None:
            idx = self.surface_points.df.index

        self.surface_points.df.loc[idx, ['X_r', 'Y_r', 'Z_r']] = self.rescale_surface_points(self.surface_points,
                                                                                     self.df.loc['values', 'rescaling factor'],
                                                                                     self.df.loc['values', 'centers'],
                                                                                     idx=idx)

        return self.surface_points

    def rescale_data_point(self, data_points: np.ndarray, rescaling_factor=None, centers=None):
        """This method now is very similar to set_rescaled_surface_points passing an index"""
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

        # Change the coordinates of orientations
        new_coord_orientations = (orientations.df.loc[idx, ['X', 'Y', 'Z']] -
                                  centers) / rescaling_factor + 0.5001

        new_coord_orientations.rename(columns={"X": "X_r", "Y": "Y_r", "Z": 'Z_r'}, inplace=True)

        return new_coord_orientations

    def set_rescaled_orientations(self, idx: Union[list, np.ndarray] = None):
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
        lssp = self.surface_points.df.groupby('id')['order_series'].count().values
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
        return self.df

    def set_length_series_o(self):
        # Array containing the size of every series. orientations.
        self.df.at['values', 'len series orientations'] = self.orientations.df['order_series'].value_counts(sort=False).values
        return self.df

    def set_number_of_surfaces_per_series(self):
        self.df.at['values', 'number surfaces per series'] = self.surface_points.df.groupby('order_series').surface.nunique().values
        return self.df

    def set_number_of_faults(self):
        # Number of faults existing in the surface_points df
        self.df.at['values', 'number faults'] = self.faults.df['isFault'].sum()#.loc[self.surface_points.df['series'].unique(), 'isFault'].sum()
        return self.df

    def set_number_of_surfaces(self):
        # Number of surfaces existing in the surface_points df
        self.df.at['values', 'number surfaces'] = self.surface_points.df['surface'].nunique()

        return self.df

    def set_is_lith_is_fault(self):
        """
         TODO Update string
        Check if there is lithologies in the data and/or df
        Returns:
            list(bool)
        """
        self.df['isLith'] = True if self.df.loc['values', 'number series'] >= self.df.loc['values', 'number faults'] else False
        self.df['isFault'] = True if self.df.loc['values', 'number faults'] > 0 else False

        return self.df


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
                value = np.fromstring(value[1:-1], sep=u_grade_sep, dtype=int)
            try:
                assert value.shape[0] is self.structure.df.loc['values', 'len series surface_points'].shape[0]
                self.df.at['values', property] = value

            except AssertionError:
                print('u_grade length must be the same as the number of series')

        else:
            self.df.loc['values', property] = value

    def str2int_u_grage(self, **kwargs):
        u_grade_sep = kwargs.get('u_grade_sep', ',')
        value = self.df.loc['values', 'drift equations']
        if type(value) is str:
            value = np.fromstring(value[1:-1], sep=u_grade_sep, dtype=int)
        try:
            assert value.shape[0] is self.structure.df.loc['values', 'len series surface_points'].shape[0]
            self.df.at['values', 'drift equations'] = value

        except AssertionError:
            print('u_grade length must be the same as the number of series')

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
                            axis=1, keys=['Structure', 'Options', 'Kriging', 'Rescaling'])
        return concat_.T

    def update_default_kriging(self):
        self.kriging_data.set_default_range()
        self.kriging_data.set_default_c_o()
        self.kriging_data.set_u_grade()
        self.kriging_data.df['nugget grad'] = 0.01
        self.kriging_data.df['nugget scalar'] = 1e-6

    def update_structure(self):
        self.structure_data.update_structure_from_input()
        self.kriging_data.set_u_grade()

