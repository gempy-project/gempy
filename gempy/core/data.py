import re
import sys
import warnings
from typing import Union

import numpy as np
import pandas as pn

try:
    import ipywidgets as widgets
    ipywidgets_import = True
except ModuleNotFoundError:
    VTK_IMPORT = False

# This is for sphenix to find the packages
from gempy.core.grid_modules import grid_types
from gempy.core.checkers import check_for_nans
from gempy.utils.meta import setdoc, setdoc_pro
import gempy.utils.docstring as ds

pn.options.mode.chained_assignment = None


class MetaData(object):
    """Class containing metadata of the project.

    Set of attributes and methods that are not related directly with the geological model but more with the project

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


@setdoc_pro([grid_types.RegularGrid.__doc__, grid_types.CustomGrid.__doc__])
class Grid(object):
    """ Class to generate grids.

    This class is used to create points where to
    evaluate the geological model. This class serves a container which transmit the XYZ coordinates to the
    interpolator. There are several type of grids objects will feed into the Grid class

    Args:
         **kwargs: See below

    Keyword Args:
         regular (:class:`gempy.core.grid_modules.grid_types.RegularGrid`): [s0]
         custom (:class:`gempy.core.grid_modules.grid_types.CustomGrid`): [s1]
         topography (:class:`gempy.core.grid_modules.grid_types.Topography`): [s2]
         section TODO @ Elisa
         gravity (:class:`gempy.core.grid_modules.grid_types.Gravity`):

    Attributes:
        values (np.ndarray): coordinates where the model is going to be evaluated. This are the coordinates
         concatenation of all active grids.
        values_r (np.ndarray): rescaled coordinates where the model is going to be evaluated
        length (np.ndarray):I a array which contain the slicing index for each grid type in order. The first element will
         be 0, the second the length of the regular grid; the third custom and so on. This can be used to slice the
         solutions correspondent to each of the grids
        grid_types(np.ndarray[str]): names of the current grids of GemPy
        active_grids(np.ndarray[bool]): boolean array which control which type of grid is going to be computed and
         hence on the property `values`.
        regular_grid (:class:`gempy.core.grid_modules.grid_types.RegularGrid`)
        custom_grid (:class:`gempy.core.grid_modules.grid_types.CustomGrid`)
        topography (:class:`gempy.core.grid_modules.grid_types.Topography`)
        section TODO @ Elisa
        gravity_grid (:class:`gempy.core.grid_modules.grid_types.Gravity`)
    """

    def __init__(self, **kwargs):

        self.values = np.empty((0, 3))
        self.values_r = np.empty((0, 3))
        self.length = np.empty(0)
        self.grid_types = np.array(['regular', 'custom', 'topography', 'sections', 'centered'])
        self.active_grids = np.zeros(5, dtype=bool)
        # All grid types must have values

        # Init optional grids
        self.custom_grid = None
        self.custom_grid_grid_active = False
        self.topography = None
        self.topography_grid_active = False
        self.sections = None
        self.sections_grid_active = False
        self.centered_grid = None
        self.centered_grid_active = False

        # Init basic grid empty
        self.regular_grid = self.set_regular_grid(**kwargs)
        self.regular_grid_active = False

    def __str__(self):
        return 'Grid Object. Values: \n' + np.array2string(self.values)

    def __repr__(self):
        return 'Grid Object. Values: \n' + np.array_repr(self.values)

    @setdoc(grid_types.RegularGrid.__doc__)
    def set_regular_grid(self, *args, **kwargs):
        """
        Set a new regular grid and activate it.

        Args:
            extent (np.ndarray): [x_min, x_max, y_min, y_max, z_min, z_max]
            resolution (np.ndarray): [nx, ny, nz]

        RegularGrid Docs
        """
        self.regular_grid = grid_types.RegularGrid(*args, **kwargs)
        self.set_active('regular')
        return self.regular_grid

    @setdoc_pro(ds.coord)
    def set_custom_grid(self, custom_grid: np.ndarray):
        """
        Set a new regular grid and activate it.

        Args:
            custom_grid (np.array): [s0]

        """
        self.custom_grid = grid_types.CustomGrid(custom_grid)
        self.set_active('custom')

    def set_topography(self, source='random', **kwargs):
        """Set a topography grid and activate it. TODO @Elisa"""
        self.topography = grid_types.Topography(self.regular_grid)

        if source == 'random':
            self.topography.load_random_hills(**kwargs)
        elif source == 'gdal':
            filepath = kwargs.get('filepath', None)
            if filepath is not None:
                self.topography.load_from_gdal(filepath)
            else:
                print('to load a raster file, a path to the file must be provided')
        elif source == 'saved':
            filepath = kwargs.get('filepath', None)
            if filepath is not None:
                self.topography.load_from_saved(filepath)
            else:
                print('path to .npy file must be provided')
        else:
            raise AttributeError('source must be random, gdal or saved')

        self.topography.show()
        self.set_active('topography')

    def set_section_grid(self, section_dict):
        self.sections = grid_types.Sections(self.regular_grid, section_dict)
        self.set_active('sections')
        return self.sections

    @setdoc(grid_types.CenteredGrid.set_centered_grid.__doc__)
    def set_centered_grid(self, centers, radio, resolution=None):
        """Initialize gravity grid. Deactivate the rest of the grids"""
        self.centered_grid = grid_types.CenteredGrid(centers, radio, resolution)
       # self.active_grids = np.zeros(4, dtype=bool)
        self.set_active('centered')

    def deactivate_all_grids(self):
        self.active_grids = np.zeros(5, dtype=bool)
        self.update_grid_values()
        return self.active_grids

    def set_active(self, grid_name: Union[str, np.ndarray]):
        """
        Set active a given or several grids
        Args:
            grid_name (str, list):

        """
        where = self.grid_types == grid_name
        self.active_grids[where] = True
        self.update_grid_values()
        return self.active_grids

    def set_inactive(self, grid_name: str):
        where = self.grid_types == grid_name
        self.active_grids *= ~where
        self.update_grid_values()
        return self.active_grids

    def update_grid_values(self):
        """
        Copy XYZ coordinates from each specific grid to Grid.values for those which are active.

        Returns:
            values

        """
        self.length = np.empty(0)
        self.values = np.empty((0, 3))
        lengths = [0]
        try:
            for e, grid_types in enumerate([self.regular_grid, self.custom_grid, self.topography, self.sections, self.centered_grid]):
                if self.active_grids[e]:
                    self.values = np.vstack((self.values, grid_types.values))
                    lengths.append(grid_types.values.shape[0])
                else:
                    lengths.append(0)
        except AttributeError:
            raise AttributeError('Grid type does not exist yet. Set the grid before activating it.')

        self.length = np.array(lengths).cumsum()
        return self.values

    def get_grid_args(self, grid_name: str):
        assert type(grid_name) is str, 'Only one grid type can be retrieved'
        assert grid_name in self.grid_types, 'possible grid types are ' + str(self.grid_types)
        where = np.where(self.grid_types == grid_name)[0][0]
        return self.length[where], self.length[where+1]

    def get_grid(self, grid_name: str):
        assert type(grid_name) is str, 'Only one grid type can be retrieved'

        l_0, l_1 = self.get_grid_args(grid_name)
        return self.values[l_0:l_1]

    def get_section_args(self, section_name: str):
        #assert type(section_name) is str, 'Only one section type can be retrieved'
        l0, l1 = self.get_grid_args('sections')
        where = np.where(self.sections.names == section_name)[0][0]
        return l0 + self.sections.length[where], l0 + self.sections.length[where+1]

class Faults(object):
    """
    Class that encapsulate faulting related content. Mainly, which surfaces/surfaces are faults. The fault network
    ---i.e. which faults offset other faults---and fault types---finite vs infinite.

    Args:
        series_fault(str, list[str]): Name of the series which are faults
        rel_matrix (numpy.array[bool]): 2D Boolean array with boolean logic. Rows affect (offset) columns

    Attributes:
       df (:class:`pn.core.frame.DataFrames`): Pandas data frame containing the series as index and if they are faults
        or not (otherwise they are lithologies) and in case of being fault if is finite
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

    # def sort_faults(self):
    #     self.df.sort_index(inplace=True)
    #     self.faults_relations_df.sort_index(inplace=True)
    #     self.faults_relations_df.sort_index(axis=1, inplace=True)

    def set_is_fault(self, series_fault: Union[str, list, np.ndarray] = None, toggle=False, offset_faults=False):
        """
        Set a flag to the series that are faults.

        Args:
            series_fault(str, list[str]): Name of the series which are faults
            toggle (bool): if True, passing a name which is already True will set it False.
            offset_faults (bool): If True by default faults offset other faults

        Returns:
            Faults

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
                # set the faults offset all younger
                self.faults_relations_df.iloc[col_pos, col_pos + 1:] = True

                if offset_faults is False:
                    # set the faults does not offset the younger faults
                    self.faults_relations_df.iloc[col_pos] = ~self.df['isFault'] & \
                                                             self.faults_relations_df.iloc[col_pos]

        self.n_faults = self.df['isFault'].sum()

        return self

    def set_is_finite_fault(self, series_finite: Union[str, list, np.ndarray] = None, toggle=False):
        """
        Toggles given series' finite fault property.

        Args:
            series_finite (str, list[str]): Name of the series which are finite
            toggle (bool): if True, passing a name which is already True will set it False.

        Returns:
            Fault
        """
        if series_finite[0] is not None:
            # check if given series is/are in dataframe
            assert np.isin(series_finite, self.df.index).all(), "series_fault must already exist" \
                                                                "in the series DataFrame."
            assert self.df.loc[series_finite].isFault.all(), "series_fault contains non-fault series" \
                                                             ", which can't be set as finite faults."
            # if so, toggle True/False for given series or list of series
            if toggle is True:
                self.df.loc[series_finite, 'isFinite'] = self.df.loc[series_finite, 'isFinite'] ^ True
            else:
                self.df.loc[series_finite, 'isFinite'] = self.df.loc[series_finite, 'isFinite']

        return self

    def set_fault_relation(self, rel_matrix=None):
        """
        Method to set the df that offset a given sequence and therefore also another fault.

        Args:
            rel_matrix (numpy.array[bool]): 2D Boolean array with boolean logic. Rows affect (offset) columns
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


@setdoc_pro(Faults.__doc__)
class Series(object):
    """ Class that contains the functionality and attributes related to the series. Notice that series does not only
    refers to stratigraphic series but to any set of surfaces which will be interpolated together (comfortably).

    Args:
        faults (:class:`Faults`): [s0]
        series_names(Optional[list]): name of the series. They are also ordered

    Attributes:
        df (:class:`pn.core.frame.DataFrames`): Pandas data frame containing the series and the surfaces contained
            on them. TODO describe df columns
        faults (:class:`Faults`)
    """

    def __init__(self, faults, series_names: list = None):

        self.faults = faults

        if series_names is None:
            series_names = ['Default series']

        self.df = pn.DataFrame(np.array([[1, np.nan]]), index=pn.CategoricalIndex(series_names, ordered=False),
                               columns=['order_series', 'BottomRelation'])

        self.df['order_series'] = self.df['order_series'].astype(int)
        self.df['BottomRelation'] = pn.Categorical(['Erosion'], categories=['Erosion', 'Onlap', 'Fault'])

    def __repr__(self):
        return self.df.to_string()

    def _repr_html_(self):
        return self.df.to_html()

    def reset_order_series(self):
        """
        Reset the column order series to monotonic ascendant values.
        """
        self.df.at[:, 'order_series'] = pn.RangeIndex(1, self.df.shape[0] + 1)

    @setdoc_pro(reset_order_series.__doc__)
    def set_series_index(self, series_order: Union[list, np.ndarray], reset_order_series=True):
        """
        Rewrite the index of the series df

        Args:
            series_order (list, :class:`SurfacePoints`): List with names and order of series. If :class:`SurfacePoints`
            is passed then the unique values will be taken.
            reset_order_series (bool): if true [s0]

        Returns:
             :class:`Series`: Series
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

        # Categorical index does not have inplace
        # This update the categories
        self.df.index = self.df.index.set_categories(series_idx, rename=True)
        self.faults.df.index = self.faults.df.index.set_categories(series_idx, rename=True)
        self.faults.faults_relations_df.index = self.faults.faults_relations_df.index.set_categories(
            series_idx, rename=True)
        self.faults.faults_relations_df.columns = self.faults.faults_relations_df.columns.set_categories(
            series_idx, rename=True)

        # But we need to update the values too
        for c in series_order:
            self.df.loc[c, 'BottomRelation'] = 'Erosion'
            self.faults.df.loc[c] = [False, False]
            self.faults.faults_relations_df.loc[c, c] = False

        self.faults.faults_relations_df.fillna(False, inplace=True)

        if reset_order_series is True:
            self.reset_order_series()
        return self

    def set_bottom_relation(self, series_list: Union[str, list], bottom_relation: Union[str, list]):
        """Set the bottom relation between the series and the one below.

        Args:
            series_list (str, list): name or list of names of the series to apply the functionality
            bottom_relation (str{Onlap, Erode, Fault}, list[str]):

        Returns:
            Series
        """
        self.df.loc[series_list, 'BottomRelation'] = bottom_relation

        if self.faults.df.loc[series_list, 'isFault'] is True:
            self.faults.set_is_fault(series_list, toggle=True)

        elif bottom_relation == 'Fault':
            self.faults.df.loc[series_list, 'isFault'] = True
        return self

    @setdoc_pro(reset_order_series.__doc__)
    def add_series(self, series_list: Union[str, list], reset_order_series=True):
        """ Add series to the df

        Args:
            series_list (str, list): name or list of names of the series to apply the functionality
            reset_order_series (bool): if true [s0]

        Returns:
            Series
        """
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

        if reset_order_series is True:
            self.reset_order_series()
        return self

    @setdoc_pro([reset_order_series.__doc__, pn.DataFrame.drop.__doc__])
    def delete_series(self, indices: Union[str, list], reset_order_series=True):
        """[s1]

        Args:
            indices (str, list): name or list of names of the series to apply the functionality
            reset_order_series (bool): if true [s0]

        Returns:
            Series
        """
        self.df.drop(indices, inplace=True)
        self.faults.df.drop(indices, inplace=True)
        self.faults.faults_relations_df.drop(indices, axis=0, inplace=True)
        self.faults.faults_relations_df.drop(indices, axis=1, inplace=True)

        idx = self.df.index.remove_unused_categories()
        self.df.index = idx
        self.update_faults_index()

        if reset_order_series is True:
            self.reset_order_series()
        return self

    @setdoc_pro(pn.CategoricalIndex.rename_categories.__doc__)
    def rename_series(self, new_categories: Union[dict, list]):
        """
        [s0]

        Args:
            new_categories (list, dict):
                * list-like: all items must be unique and the number of items in the new categories must match the
                  existing number of categories.

                * dict-like: specifies a mapping from old categories to new. Categories not contained in the mapping are
                  passed through and extra categories in the mapping are ignored.
        Returns:

        """
        idx = self.df.index.rename_categories(new_categories)
        self.df.index = idx
        self.update_faults_index()

        return self

    @setdoc_pro([pn.CategoricalIndex.reorder_categories.__doc__, pn.CategoricalIndex.sort_values.__doc__])
    def reorder_series(self, new_categories: Union[list, np.ndarray]):
        """[s0] [s1]

        Args:
            new_categories (list): list with all series names in the desired order.

        Returns:
            Series
        """
        idx = self.df.index.reorder_categories(new_categories).sort_values()
        self.df.index = idx
        self.update_faults_index()
        return self

    def modify_order_series(self, new_value: int, series_name: str):
        """
        Replace to the new location the old series

        Args:
            new_value (int): New location
            series_name (str): name of the series to be moved

        Returns:
            Series
        """
        group = self.df['order_series']
        assert np.isin(new_value, group), 'new_value must exist already in the order_surfaces group.'
        old_value = group[series_name]
        self.df['order_series'] = group.replace([new_value, old_value], [old_value, new_value])
        self.sort_series()
        self.update_faults_index()

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


class Colors:
    # TODO @elisa
    def __init__(self, surfaces):
        self.surfaces = surfaces

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
            self.update_colors(cdict)
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

    def update_colors(self, cdict=None):
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


@setdoc_pro(Series.__doc__)
class Surfaces(object):
    """
    Class that contains the surfaces of the model and the values of each of them.

    Args:
        surface_names (list or np.ndarray): list containing the names of the surfaces
        series (:class:`Series`): [s0]
        values_array (np.ndarray): 2D array with the values of each surface
        properties names (list or np.ndarray): list containing the names of each properties

    Attributes:
        df (:class:`pn.core.frame.DataFrames`): Pandas data frame containing the surfaces names mapped to series and
            the value used for each voxel in the final model.
        series (:class:`Series`)
        colors (:class:`Colors`)

    """

    def __init__(self, series: Series, surface_names=None, values_array=None, properties_names=None):

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

    def __repr__(self):
        return self.df.to_string()

    def _repr_html_(self):
        c_ = self.df.columns[~(self.df.columns.isin(self._columns_vis_drop))]

        return self.df[c_].style.applymap(self.background_color, subset=['color']).render()

    def update_id(self, id_list: list = None):
        """
        Set id of the layers (1 based)
        Args:
            id_list (list):

        Returns:
             :class:`Surfaces`:

        """

        if id_list is None:
            id_list = self.df.reset_index().index + 1

        self.df['id'] = id_list

        return self

    @staticmethod
    def background_color(value):
        if type(value) == str:
            return "background-color: %s" % value

# region set formation names
    def set_surfaces_names(self, surfaces_list: list, update_df=True):
        """
         Method to set the names of the surfaces in order. This applies in the surface column of the df

         Args:
             surfaces_list (list[str]):  list of names of surfaces. They are ordered.
             update_df (bool): Update Surfaces.df columns with the default values

         Returns:
             :class:`Surfaces`:
         """
        if type(surfaces_list) is list or type(surfaces_list) is np.ndarray:
            surfaces_list = np.asarray(surfaces_list)

        else:
            raise AttributeError('list_names must be either array_like type')

        # Deleting all columns if they exist
        # TODO check if some of the names are in the df and not deleting them?
        self.df.drop(self.df.index, inplace=True)
        self.df['surface'] = surfaces_list

        # Changing the name of the series is the only way to mutate the series object from surfaces
        if update_df is True:
            self.map_series()
            self.update_id()
            self.set_basement()
            self.reset_order_surfaces()
            self.colors.update_colors()
        return self

    def set_default_surface_name(self):
        """
        Set the minimum number of surfaces to compute a model i.e. surfaces_names: surface1 and basement

        Returns:
             :class:`Surfaces`:

        """
        if self.df.shape[0] == 0:
            # TODO DEBUG: I am not sure that surfaces always has at least one entry. Check it
            self.set_surfaces_names(['surface1', 'basement'])
        return self

    def set_surfaces_names_from_surface_points(self, surface_points):
        """
        Set surfaces names from a :class:`Surface_points` object. This can be useful if the surface points are imported
        from a table.

        Args:
            surface_points (:class:`Surface_points`):

        Returns:

        """
        self.set_surfaces_names(surface_points.df['surface'].unique())
        return self

    def add_surface(self, surface_list: Union[str, list], update_df=True):
        """ Add surface to the df.

        Args:
            surface_list (str, list): name or list of names of the surfaces to apply the functionality
            update_df (bool): Update Surfaces.df columns with the default values

        Returns:
             :class:`Surfaces`:

        """

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
            self.reset_order_surfaces()
            self.colors.update_colors()
        return self

    @setdoc_pro([update_id.__doc__, pn.DataFrame.drop.__doc__])
    def delete_surface(self, indices: Union[int, str, list, np.ndarray], update_id=True):
        """[s1]

        Args:
            indices (str, list): name or list of names of the series to apply the functionality
            update_id (bool): if true [s0]

        Returns:
             :class:`Surfaces`:

        """
        indices = np.atleast_1d(indices)
        if indices.dtype == int:
            self.df.drop(indices, inplace=True)
        else:
            self.df.drop(self.df.index[self.df['surface'].isin(indices)], inplace=True)

        if update_id is True:
            self.update_id()
            self.set_basement()
            self.reset_order_surfaces()
        return self

    @setdoc(pn.Series.replace.__doc__)
    def rename_surfaces(self, to_replace: Union[str, list, dict],  **kwargs):
        if np.isin(to_replace, self.df['surface']).any():
            print('Two surfaces cannot have the same name.')
        else:
            self.df['surface'].replace(to_replace,  inplace=True, **kwargs)
        return self

    def reset_order_surfaces(self):
        self.df['order_surfaces'] = self.df.groupby('series').cumcount() + 1

    def modify_order_surfaces(self, new_value: int, idx: int, series_name: str = None):
        """
          Replace to the new location the old series

          Args:
              new_value (int): New location
              idx (int): Index of the surface to be moved
              series_name (str): name of the series to be moved

          Returns:
             :class:`Surfaces`:

          """
        if series_name is None:
            series_name = self.df.loc[idx, 'series']

        group = self.df.groupby('series').get_group(series_name)['order_surfaces']
        assert np.isin(new_value, group), 'new_value must exist already in the order_surfaces group.'
        old_value = group[idx]
        self.df.loc[group.index, 'order_surfaces'] = group.replace([new_value, old_value], [old_value, new_value])
        self.sort_surfaces()
        self.set_basement()
        return self

    def sort_surfaces(self):
        """Sort surfaces by series and order_surfaces"""

        self.df.sort_values(by=['series', 'order_surfaces'], inplace=True)
        self.update_id()
        return self.df

    def set_basement(self):
        """
        Set isBasement property to true to the last series of the df.

        Returns:
             :class:`Surfaces`:

        """

        self.df['isBasement'] = False
        idx = self.df.last_valid_index()
        if idx is not None:
            self.df.loc[idx, 'isBasement'] = True

        # TODO add functionality of passing the basement and calling reorder to push basement surface to the bottom
        #  of the data frame
        assert self.df['isBasement'].values.astype(bool).sum() <= 1, 'Only one surface can be basement'
        return self

# endregion

# set_series
    def map_series(self, mapping_object: Union[dict, pn.DataFrame] = None):
        """
        Method to map to which series every surface belongs to. This step is necessary to assign differenct tectonics
        such as unconformities or faults.


        Args:
            mapping_object (dict, :class:`pn.DataFrame`):
                * dict: keys are the series and values the surfaces belonging to that series

                * pn.DataFrame: Dataframe with surfaces as index and a column series with the correspondent series name
                  of each surface

        Returns:
             :class:`Surfaces`

        """

        # Updating surfaces['series'] categories
        self.df['series'].cat.set_categories(self.series.df.index, inplace=True)

        # TODO Fixing this. It is overriding the formations already mapped
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
                # TODO Test this
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
        self.reset_order_surfaces()
        self.sort_surfaces()
        self.set_basement()
        return self

# endregion

# region update_id

# endregion

    def add_surfaces_values(self, values_array: Union[np.ndarray, list], properties_names: list = np.empty(0)):
        """
        Add values to be interpolated for each surfaces
        Args:
            values_array (np.ndarray, list): array-like of the same length as number of surfaces. This functionality
            can be used to assign different geophysical properties to each layer
            properties_names (list): list of names for each values_array columns. This must be of same size as
            values_array axis 1. By default properties will take the column name: 'value_X'.

        Returns:
             :class:`Surfaces`:

        """
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

    def delete_surface_values(self, properties_names: Union[str, list]):
        """
        Delete a property or several properties column.
        Args:
            properties_names (str, list[str]): Name of the property to delete

        Returns:
             :class:`Surfaces`:

        """

        properties_names = np.asarray(properties_names)
        self.df.drop(properties_names, axis=1, inplace=True)
        return True

    def set_surfaces_values(self, values_array: Union[np.ndarray, list], properties_names: list = np.empty(0)):
        """
        Set values to be interpolated for each surfaces. This method will delete the previous values.
        Args:
            values_array (np.ndarray, list): array-like of the same length as number of surfaces. This functionality
            can be used to assign different geophysical properties to each layer
            properties_names (list): list of names for each values_array columns. This must be of same size as
            values_array axis 1. By default properties will take the column name: 'value_X'.

        Returns:
             :class:`Surfaces`:

        """
        # Check if there are values columns already
        old_prop_names = self.df.columns[~self.df.columns.isin(['surface', 'series', 'order_surfaces',
                                                                'id', 'isBasement', 'color'])]
        # Delete old
        self.delete_surface_values(old_prop_names)

        # Create new
        self.add_surfaces_values(values_array, properties_names)
        return self

    def modify_surface_values(self, idx, properties_names, values):
        """Method to modify values using loc of pandas.

        Args:
            idx (int, list[int]):
            properties_names (str, list[str]:
            values (float, np.ndarray):

        Returns:
             :class:`Surfaces`:

        """
        properties_names = np.atleast_1d(properties_names)
        assert ~np.isin(properties_names, ['surface', 'series', 'order_surfaces', 'id', 'isBasement', 'color']),\
            'only property names can be modified with this method'

        self.df.loc[idx, properties_names] = values
        return self


@setdoc_pro(Surfaces.__doc__)
class GeometricData(object):
    """
    Parent class of the objects which containing the input parameters: surface_points and orientations. This class
     contain the common methods for both types of data sets.

    Args:
        surfaces (:class:`Surfaces`): [s0]

    Attributes:
        surfaces (:class:`Surfaces`)
        df (:class:`pn.DataFrame`): Pandas DataFrame containing all the properties of each individual data point i.e.
        surface points and orientations
    """

    def __init__(self, surfaces: Surfaces):

        self.surfaces = surfaces
        self.df = pn.DataFrame()

    def __repr__(self):
        return self.df.to_string()

    def _repr_html_(self):
        return self.df.to_html()

    def init_dependent_properties(self):
        """Set the defaults values to the columns before gets mapped with the the :class:`Surfaces` attribute. This
        method will get invoked for example when we add a new point."""

        # series
        self.df['series'] = 'Default series'
        self.df['series'] = self.df['series'].astype('category', copy=True)
        self.df['series'].cat.set_categories(self.surfaces.df['series'].cat.categories, inplace=True)

        # id
        self.df['id'] = np.nan

        # order_series
        self.df['order_series'] = 1
        return self

    @staticmethod
    @setdoc(pn.read_csv.__doc__, indent=False)
    def read_data(file_path, **kwargs):
        """"""
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

    @setdoc_pro(Series.__doc__)
    def set_series_categories_from_series(self, series: Series):
        """set the series categorical columns with the series index of the passed :class:`Series`

        Args:
            series (:class:`Series`): [s0]
        """
        self.df['series'].cat.set_categories(series.df.index, inplace=True)
        return True

    def update_series_category(self):
        """Update the series categorical columns with the series categories of the :class:`Surfaces` attribute."""
        self.df['series'].cat.set_categories(self.surfaces.df['series'].cat.categories, inplace=True)

        return True

    @setdoc_pro(Surfaces.__doc__)
    def set_surface_categories_from_surfaces(self, surfaces: Surfaces):
        """set the series categorical columns with the series index of the passed :class:`Series`.

        Args:
            surfaces (:class:`Surfaces`): [s0]

        """

        self.df['surface'].cat.set_categories(surfaces.df['surface'], inplace=True)
        return True

    @setdoc_pro(Series.__doc__)
    def map_data_from_series(self, series, attribute: str, idx=None):
        """
        Map columns from the :class:`Series` data frame to a :class:`GeometricData` data frame.

        Args:
            series (:class:`Series`): [s0]
            attribute (str): column to be mapped from the :class:`Series` to the :class:`GeometricData`.
            idx (Optional[int, list[int]): If passed, list of indices of the :class:`GeometricData` that will be mapped.

        Returns:
            :class:GeometricData
        """
        if idx is None:
            idx = self.df.index

        idx = np.atleast_1d(idx)
        self.df.loc[idx, attribute] = self.df['series'].map(series.df[attribute])
        return self

    @setdoc_pro(Surfaces.__doc__)
    def map_data_from_surfaces(self, surfaces, attribute: str, idx=None):
        """
        Map columns from the :class:`Series` data frame to a :class:`GeometricData` data frame.
        Properties of surfaces: series, id, values.

        Args:
            surfaces (:class:`Surfaces`): [s0]
            attribute (str): column to be mapped from the :class:`Series` to the :class:`GeometricData`.
            idx (Optional[int, list[int]): If passed, list of indices of the :class:`GeometricData` that will be mapped.

        Returns:
            :class:GeometricData
        """

        if idx is None:
            idx = self.df.index
        idx = np.atleast_1d(idx)
        if attribute is 'series':
            if surfaces.df.loc[~surfaces.df['isBasement']]['series'].isna().sum() != 0:
                raise AttributeError('Surfaces does not have the correspondent series assigned. See'
                                     'Surfaces.map_series_from_series.')

        self.df.loc[idx, attribute] = self.df.loc[idx, 'surface'].map(surfaces.df.set_index('surface')[attribute])

    # def map_data_from_faults(self, faults, idx=None):
    #     """
    #     Method to map a df object into the data object on surfaces. Either if the surface is fault or not
    #     Args:
    #         faults (Faults):
    #
    #     Returns:
    #         pandas.core.frame.DataFrame: Data frame with the raw data
    #
    #     """
    #     if idx is None:
    #         idx = self.df.index
    #     idx = np.atleast_1d(idx)
    #     if any(self.df['series'].isna()):
    #         warnings.warn('Some points do not have series/fault')
    #
    #     self.df.loc[idx, 'isFault'] = self.df.loc[[idx], 'series'].map(faults.df['isFault'])


@setdoc_pro([Surfaces.__doc__, ds.coord, ds.surface_sp])
class SurfacePoints(GeometricData):
    """
    Data child with specific methods to manipulate interface data. It is initialize without arguments to give
    flexibility to the origin of the data.

    Args:
        surfaces (:class:`Surfaces`): [s0]
        coord (np.ndarray): [s1]
        surface (list[str]): [s2]


    Attributes:
          df (:class:`pn.core.frame.DataFrames`): Pandas data frame containing the necessary information respect
          the surface points of the model
    """

    def __init__(self, surfaces: Surfaces, coord=None, surface=None):

        super().__init__(surfaces)
        self._columns_i_all = ['X', 'Y', 'Z', 'surface', 'series', 'X_std', 'Y_std', 'Z_std',
                               'order_series', 'surface_number']

        self._columns_i_1 = ['X', 'Y', 'Z', 'X_r', 'Y_r', 'Z_r', 'surface', 'series', 'id',
                             'order_series', 'isFault', 'Smoothness']

        self._columns_rep = ['X', 'Y', 'Z', 'surface', 'series']
        self._columns_i_num = ['X', 'Y', 'Z', 'X_r', 'Y_r', 'Z_r']

        if (np.array(sys.version_info[:2]) <= np.array([3, 6])).all():
            self.df: pn.DataFrame

        self.set_surface_points(coord, surface)

    @setdoc_pro([ds.coord, ds.surface_sp])
    def set_surface_points(self, coord: np.ndarray = None, surface: list = None):
        """
        Set coordinates and surface columns on the df.

        Args:
            coord (np.ndarray): [s0]
            surface (list[str]): [s1]

        Returns:
            :class:`SurfacePoints`
        """
        self.df = pn.DataFrame(columns=['X', 'Y', 'Z', 'X_r', 'Y_r', 'Z_r', 'surface'], dtype=float)

        if coord is not None and surface is not None:
            self.df[['X', 'Y', 'Z']] = pn.DataFrame(coord)
            self.df['surface'] = surface

        self.df['surface'] = self.df['surface'].astype('category', copy=True)
        self.df['surface'].cat.set_categories(self.surfaces.df['surface'].values, inplace=True)

        # Choose types
        self.init_dependent_properties()

        # Add nugget columns
        self.df['smooth'] = 1e-8

        assert ~self.df['surface'].isna().any(), 'Some of the surface passed does not exist in the Formation' \
                                                 'object. %s' % self.df['surface'][self.df['surface'].isna()]

        return self

    @setdoc_pro([ds.x, ds.y, ds.z, ds.surface_sp, ds.idx_sp])
    def add_surface_points(self, x: Union[float, np.ndarray], y: Union[float, np.ndarray], z: Union[float, np.ndarray],
                           surface: Union[list, np.ndarray], idx: Union[int, list, np.ndarray] = None):
        """
        Add surface points.

        Args:
            x (float, np.ndarray): [s0]
            y (float, np.ndarray): [s1]
            z (float, np.ndarray): [s2]
            surface (list[str]): [s3]
            idx (Optional[int, list[int]): [s4]

        Returns:
            :class:`SurfacePoints`

        """

        # TODO: Add the option to pass the surface number

        if idx is None:
            idx = self.df.index.max()
            if idx is np.nan:
                idx = 0
            else:
                idx += 1

        coord_array = np.array([x, y, z])
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

        self.df.loc[idx, ['smooth']] = 1e-8

        self.map_data_from_surfaces(self.surfaces, 'series', idx=idx)
        self.map_data_from_surfaces(self.surfaces, 'id', idx=idx)
        self.map_data_from_series(self.surfaces.series, 'order_series', idx=idx)

        self.sort_table()
        return self, idx

    @setdoc_pro([ds.idx_sp])
    def del_surface_points(self, idx: Union[int, list, np.ndarray]):
        """
        Delete surface points
        Args:
            idx (int, list[int]): [s0]

        Returns:
            :class:`SurfacePoints`
        """
        self.df.drop(idx, inplace=True)
        return self

    @setdoc_pro([ds.idx_sp, ds.x, ds.y, ds.z, ds.surface_sp])
    def modify_surface_points(self, idx: Union[int, list, np.ndarray], **kwargs):
        """
         Allows modification of the x,y and/or z-coordinates of an interface at specified dataframe index.

         Args:
             idx (int, list, np.ndarray): [s0]
             **kwargs:
                * X: [s1]
                * Y: [s2]
                * Z: [s3]
                * surface: [s4]

         Returns:
            :class:`SurfacePoints`
         """
        idx = np.array(idx, ndmin=1)
        keys = list(kwargs.keys())
        is_surface = np.isin('surface', keys).all()

        # Check idx exist in the df
        assert np.isin(np.atleast_1d(idx), self.df.index).all(), 'Indices must exist in the dataframe to be modified.'

        # Check the properties are valid
        assert np.isin(list(kwargs.keys()), ['X', 'Y', 'Z', 'surface', 'smooth']).all(),\
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

        return self

    @setdoc_pro([ds.file_path, ds.debug, ds.inplace])
    def read_surface_points(self, file_path, debug=False, inplace=False,
                            kwargs_pandas: dict = None, **kwargs, ):
        """
        Read tabular using pandas tools and if inplace set it properly to the surface points object.

        Parameters:
            file_path (str, path object, or file-like object): [s0]
            debug (bool): [s1]
            inplace (bool): [s2]
            kwargs_pandas: kwargs for the panda function :func:`pn.read_csv`
            **kwargs:
                * update_surfaces (bool): If True add to the linked `Surfaces` object unique surface names read on
                  the csv file
                * coord_x_name (str): Name of the header on the csv for this attribute, e.g for coord_x. Default X
                * coord_y_name (str): Name of the header on the csv for this attribute. Default Y.
                * coord_z_name (str): Name of the header on the csv for this attribute. Default Z.
                * surface_name (str): Name of the header on the csv for this attribute. Default formation

        Returns:

        See Also:
            :meth:`GeometricData.read_data`
        """
        if 'sep' not in kwargs:
            kwargs['sep'] = ','

        coord_x_name = kwargs.get('coord_x_name', "X")
        coord_y_name = kwargs.get('coord_y_name', "Y")
        coord_z_name = kwargs.get('coord_z_name', "Z")
        surface_name = kwargs.get('surface_name', "formation")

        if kwargs_pandas is None:
            kwargs_pandas = {}

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
            assert {coord_x_name, coord_y_name, coord_z_name, surface_name}.issubset(table.columns), \
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
        """
        if self.df.shape[0] == 0:
            self.add_surface_points(0.00001, 0.00001, 0.00001, self.surfaces.df['surface'].iloc[0])
        return True

    def update_annotations(self):
        """
        Add a column in the Dataframes with latex names for each input_data paramenter.

        Returns:
            :class:`SurfacePoints`
        """
        point_num = self.df.groupby('id').cumcount()
        point_l = [r'${\bf{x}}_{\alpha \,{\bf{' + str(f) + '}},' + str(p) + '}$'
                   for p, f in zip(point_num, self.df['id'])]

        self.df['annotations'] = point_l
        return self


@setdoc_pro([Surfaces.__doc__, ds.coord_ori, ds.surface_sp, ds.pole_vector, ds.orientations])
class Orientations(GeometricData):
    """
    Data child with specific methods to manipulate orientation data. It is initialize without arguments to give
    flexibility to the origin of the data.

    Args:
        surfaces (:class:`Surfaces`): [s0]
        coord (np.ndarray): [s1]
        pole_vector (np.ndarray): [s3]
        orientation (np.ndarray): [s4]
        surface (list[str]): [s2]
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

    @setdoc_pro([ds.coord_ori, ds.surface_sp, ds.pole_vector, ds.orientations])
    def set_orientations(self, coord: np.ndarray = None, pole_vector: np.ndarray = None,
                         orientation: np.ndarray = None, surface: list = None):
        """
        Set coordinates, surface and orientation data.

        If both are passed pole vector has priority over orientation

        Args:
            coord (np.ndarray): [s0]
            pole_vector (np.ndarray): [s2]
            orientation (np.ndarray): [s3]
            surface (list[str]): [s1]

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

        self.init_dependent_properties()

        # Add nugget effect
        self.df['smooth'] = 0.01
        assert ~self.df['surface'].isna().any(), 'Some of the surface passed does not exist in the Formation' \
                                                 'object. %s' % self.df['surface'][self.df['surface'].isna()]

    @setdoc_pro([ds.x, ds.y, ds.z, ds.surface_sp, ds.pole_vector, ds.orientations, ds.idx_sp])
    def add_orientation(self, x, y, z, surface, pole_vector: Union[list, np.ndarray] = None,
                        orientation: Union[list, np.ndarray] = None, idx=None):
        """
        Add orientation.

        Args:
            x (float, np.ndarray): [s0]
            y (float, np.ndarray): [s1]
            z (float, np.ndarray): [s2]
            surface (list[str]): [s3]
            pole_vector (np.ndarray): [s4]
            orientation (np.ndarray): [s5]
            idx (Optional[int, list[int]): [s6]

        Returns:
            Orientations
        """
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
            self.df.loc[idx, ['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z']] = np.array([x, y, z, *pole_vector])
            self.df.loc[idx, 'surface'] = surface

            self.calculate_orientations(idx)

            if orientation is not None:
                warnings.warn('If pole_vector and orientation are passed pole_vector is used/')
        else:
            if orientation is not None:
                self.df.loc[idx, ['X', 'Y', 'Z', ]] = np.array([x, y, z])
                self.df.loc[idx, ['azimuth', 'dip', 'polarity']] = orientation
                self.df.loc[idx, 'surface'] = surface

                self.calculate_gradient(idx)
            else:
                raise AttributeError('At least pole_vector or orientation should have been passed to reach'
                                     'this point. Check previous condition')

        self.df.loc[idx, ['smooth']] = 0.01
        self.map_data_from_surfaces(self.surfaces, 'series', idx=idx)
        self.map_data_from_surfaces(self.surfaces, 'id', idx=idx)
        self.map_data_from_series(self.surfaces.series, 'order_series', idx=idx)

        self.sort_table()
        return self

    @setdoc_pro([ds.idx_sp])
    def del_orientation(self, idx):
        """
        Delete orientation

        Args:
            idx (int, list[int]): [s0]

        Returns:
            :class:`Orientations`
        """
        self.df.drop(idx, inplace=True)

    @setdoc_pro([ds.idx_sp, ds.surface_sp])
    def modify_orientations(self, idx, **kwargs):
        """
         Allows modification of any of an orientation column at a given index.

         Args:
             idx (int, list[int]): [s0]
             **kwargs:
                * X
                * Y
                * Z
                * G_x
                * G_y
                * G_z
                * dip
                * azimuth
                * polarity
                * surface (str): [s1]

         Returns:

         """

        idx = np.array(idx, ndmin=1)
        keys = list(kwargs.keys())
        is_surface = np.isin('surface', keys).all()

        # Check idx exist in the df
        assert np.isin(np.atleast_1d(idx), self.df.index).all(), 'Indices must exist in the dataframe to be modified.'

        # Check the properties are valid
        assert np.isin(list(kwargs.keys()), ['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', 'dip',
                                             'azimuth', 'polarity', 'surface', 'smooth']).all(),\
            'Properties must be one or more of the following: \'X\', \'Y\', \'Z\', \'G_x\', \'G_y\', \'G_z\', \'dip,\''\
            '\'azimuth\', \'polarity\', \'surface\''

        # stack properties values
        values = np.atleast_1d(list(kwargs.values()))

        # If we pass multiple index we need to transpose the numpy array
        if type(idx) is list or type(idx) is np.ndarray:
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
        return self

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
        return True

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

        return True

    @setdoc_pro([SurfacePoints.__doc__])
    def create_orientation_from_interface(self, surface_points: SurfacePoints, indices):
        # TODO test!!!!
        """
        Create and set orientations from at least 3 points categories_df

        Args:
            surface_points (:class:`SurfacePoints`): [s0]
            indices (list[int]): indices of the surface point used to generate the orientation. At least
             3 independent points will need to be passed.
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

    @setdoc_pro([ds.file_path, ds.debug, ds.inplace])
    def read_orientations(self, file_path, debug=False, inplace=True, kwargs_pandas: dict = None, **kwargs):
        """
        Read tabular using pandas tools and if inplace set it properly to the surface points object.

        Args:
            file_path (str, path object, or file-like object): [s0]
            debug (bool): [s1]
            inplace (bool): [s2]
            kwargs_pandas: kwargs for the panda function :func:`pn.read_csv`
            **kwargs:
                * update_surfaces (bool): If True add to the linked `Surfaces` object unique surface names read on
                  the csv file
                * coord_x_name (str): Name of the header on the csv for this attribute, e.g for coord_x. Default X
                * coord_y_name (str): Name of the header on the csv for this attribute. Default Y
                * coord_z_name (str): Name of the header on the csv for this attribute. Default Z
                * coord_x_name (str): Name of the header on the csv for this attribute. Default G_x
                * coord_y_name (str): Name of the header on the csv for this attribute. Default G_y
                * coord_z_name (str): Name of the header on the csv for this attribute. Default G_z
                * azimuth_name (str): Name of the header on the csv for this attribute. Default azimuth
                * dip_name     (str): Name of the header on the csv for this attribute. Default dip
                * polarity_name (str): Name of the header on the csv for this attribute. Default polarity
                * surface_name (str): Name of the header on the csv for this attribute. Default formation


        Returns:

        See Also:
            :meth:`GeometricData.read_data`
        """
        coord_x_name = kwargs.get('coord_x_name', "X")
        coord_y_name = kwargs.get('coord_y_name', "Y")
        coord_z_name = kwargs.get('coord_z_name', "Z")
        g_x_name = kwargs.get('G_x_name', 'G_x')
        g_y_name = kwargs.get('G_y_name', 'G_y')
        g_z_name = kwargs.get('G_z_name', 'G_z')
        azimuth_name = kwargs.get('azimuth_name', 'azimuth')
        dip_name = kwargs.get('dip_name', 'dip')
        polarity_name = kwargs.get('polarity_name', 'polarity')
        surface_name = kwargs.get('surface_name', "formation")

        if kwargs_pandas is None:
            kwargs_pandas = {}

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
            assert {coord_x_name, coord_y_name, coord_z_name, dip_name, azimuth_name,
                    polarity_name, surface_name}.issubset(table.columns), \
                "One or more columns do not match with the expected values " + str(table.columns)

            if inplace:
                # self.categories_df[table.columns] = table
                c = np.array(self._columns_o_1)
                orientations_read = table.assign(**dict.fromkeys(c[~np.in1d(c, table.columns)], np.nan))
                self.set_orientations(coord=orientations_read[[coord_x_name, coord_y_name, coord_z_name]],
                                      pole_vector=orientations_read[[g_x_name, g_y_name, g_z_name]].values,
                                      orientation=orientations_read[[azimuth_name, dip_name, polarity_name]].values,
                                      surface=orientations_read[surface_name])
            else:
                return table

    def update_annotations(self):
        """
        Add a column in the Dataframes with latex names for each input_data paramenter.

        Returns:

        """
        orientation_num = self.df.groupby('id').cumcount()
        foli_l = [r'${\bf{x}}_{\beta \,{\bf{' + str(f) + '}},' + str(p) + '}$'
                  for p, f in zip(orientation_num, self.df['id'])]

        self.df['annotations'] = foli_l
        return self

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
        elif normal[0] < 0 >= normal[1]:
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


@setdoc_pro([SurfacePoints.__doc__, Orientations.__doc__, Grid.__doc__])
class RescaledData(object):
    """
    Auxiliary class to rescale the coordinates between 0 and 1 to increase float stability.

    Attributes:
        df (:class:`pn.DataFrame`): Data frame containing the rescaling factor and centers
        surface_points (:class:`SurfacePoints`): [s0]
        orientations (:class:`Orientations`): [s1]
        grid (:class:`Grid`): [s2]

    Args:
        surface_points (:class:`SurfacePoints`):
        orientations (:class:`Orientations`):
        grid (:class:`Grid`):
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

    @setdoc_pro([ds.centers, ds.rescaling_factor])
    def modify_rescaling_parameters(self, attribute, value):
        """
        Modify the parameters used to rescale data

        Args:
            attribute (str): Attribute to be modified. It can be: centers, rescaling factor
            value (float, list[float]):
                * centers: [s0]
                * rescaling factor: [s1]

        Returns:

        """
        assert np.isin(attribute, self.df.columns).all(), 'Valid attributes are: ' + np.array2string(self.df.columns)

        if attribute == 'centers':
            try:
                assert value.shape[0] is 3

                self.df.loc['values', attribute] = value

            except AssertionError:
                print('centers length must be 3: XYZ')

        else:
            self.df.loc['values', attribute] = value

    @setdoc_pro([ds.centers, ds.rescaling_factor])
    def rescale_data(self, rescaling_factor=None, centers=None):
        """
        Rescale inplace: surface_points, orientations---adding columns in the categories_df---and grid---adding values_r
        attributes. The rescaled values will get stored on the linked objects.

        Args:
            rescaling_factor: [s1]
            centers: [s0]

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
            :attr:`SurfacePoints.df[['X_r', 'Y_r', 'Z_r']]`
        """
        return self.surface_points.df[['X_r', 'Y_r', 'Z_r']],

    def get_rescaled_orientations(self):
        """
        Get the rescaled coordinates. return an image of the interface and orientations categories_df with the X_r..
         columns.

        Returns:
            :attr:`Orientations.df[['X_r', 'Y_r', 'Z_r']]`
        """
        return self.orientations.df[['X_r', 'Y_r', 'Z_r']]

    @staticmethod
    @setdoc_pro([SurfacePoints.__doc__, Orientations.__doc__])
    def max_min_coord(surface_points=None, orientations=None):
        """
        Find the maximum and minimum location of any input data in each cartesian coordinate

        Args:
            surface_points (:class:`SurfacePoints`): [s0]
            orientations (:class:`Orientations`): [s1]

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

    @setdoc_pro([SurfacePoints.__doc__, Orientations.__doc__, ds.centers])
    def compute_data_center(self, surface_points=None, orientations=None,
                            max_coord=None, min_coord=None, inplace=True):
        """
        Calculate the center of the data once it is shifted between 0 and 1.

        Args:
            surface_points (:class:`SurfacePoints`): [s0]
            orientations (:class:`Orientations`): [s1]
            max_coord (float): Max XYZ coordinates of all GeometricData
            min_coord (float): Min XYZ coordinates of all GeometricData
            inplace (bool): if True modify the self.df rescaling factor attribute

        Returns:
            np.array: [s2]
        """

        if max_coord is None or min_coord is None:
            max_coord, min_coord = self.max_min_coord(surface_points, orientations)

        # Get the centers of every axis
        centers = ((max_coord + min_coord) / 2).astype(float).values
        if inplace is True:
            self.df.at['values', 'centers'] = centers
        return centers

    # def update_centers(self, surface_points=None, orientations=None, max_coord=None, min_coord=None):
    #     # TODO this should update the additional data
    #     self.compute_data_center(surface_points, orientations, max_coord, min_coord, inplace=True)

    @setdoc_pro([SurfacePoints.__doc__, Orientations.__doc__, ds.rescaling_factor])
    def compute_rescaling_factor(self, surface_points=None, orientations=None,
                                 max_coord=None, min_coord=None, inplace=True):
        """
        Calculate the rescaling factor of the data to keep all coordinates between 0 and 1

        Args:
            surface_points (:class:`SurfacePoints`): [s0]
            orientations (:class:`Orientations`): [s1]
            max_coord (float): Max XYZ coordinates of all GeometricData
            min_coord (float): Min XYZ coordinates of all GeometricData
            inplace (bool): if True modify the self.df rescaling factor attribute

        Returns:
            float: [s2]
        """

        if max_coord is None or min_coord is None:
            max_coord, min_coord = self.max_min_coord(surface_points, orientations)
        rescaling_factor_val = (2 * np.max(max_coord - min_coord))
        if inplace is True:
            self.df['rescaling factor'] = rescaling_factor_val
        return rescaling_factor_val

    # def update_rescaling_factor(self, surface_points=None, orientations=None,
    #                             max_coord=None, min_coord=None):
    #     self.compute_rescaling_factor(surface_points, orientations, max_coord, min_coord, inplace=True)

    @staticmethod
    @setdoc_pro([SurfacePoints.__doc__, compute_data_center.__doc__, compute_rescaling_factor.__doc__, ds.idx_sp])
    def rescale_surface_points(surface_points, rescaling_factor, centers, idx: list = None):
        """
        Rescale inplace: surface_points. The rescaled values will get stored on the linked objects.

        Args:
            surface_points (:class:`SurfacePoints`): [s0]
            rescaling_factor: [s2]
            centers: [s1]
            idx (int, list of int): [s3]

        Returns:

        """

        if idx is None:
            idx = surface_points.df.index

        # Change the coordinates of surface_points
        new_coord_surface_points = (surface_points.df.loc[idx, ['X', 'Y', 'Z']] -
                                    centers) / rescaling_factor + 0.5001

        new_coord_surface_points.rename(columns={"X": "X_r", "Y": "Y_r", "Z": 'Z_r'}, inplace=True)
        return new_coord_surface_points

    @setdoc_pro(ds.idx_sp)
    def set_rescaled_surface_points(self, idx: Union[list, np.ndarray] = None):
        """
        Set the rescaled coordinates into the surface_points categories_df

        Args:
            idx (int, list of int): [s0]

        Returns:

        """
        if idx is None:
            idx = self.surface_points.df.index

        self.surface_points.df.loc[idx, ['X_r', 'Y_r', 'Z_r']] = self.rescale_surface_points(
            self.surface_points, self.df.loc['values', 'rescaling factor'], self.df.loc['values', 'centers'], idx=idx)

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
    @setdoc_pro([Orientations.__doc__, compute_data_center.__doc__, compute_rescaling_factor.__doc__, ds.idx_sp])
    def rescale_orientations(orientations, rescaling_factor, centers, idx: list = None):
        """
        Rescale inplace: surface_points. The rescaled values will get stored on the linked objects.

        Args:
            orientations (:class:`Orientations`): [s0]
            rescaling_factor: [s2]
            centers: [s1]
            idx (int, list of int): [s3]

        Returns:

        """
        if idx is None:
            idx = orientations.df.index

        # Change the coordinates of orientations
        new_coord_orientations = (orientations.df.loc[idx, ['X', 'Y', 'Z']] -
                                  centers) / rescaling_factor + 0.5001

        new_coord_orientations.rename(columns={"X": "X_r", "Y": "Y_r", "Z": 'Z_r'}, inplace=True)

        return new_coord_orientations

    @setdoc_pro(ds.idx_sp)
    def set_rescaled_orientations(self, idx: Union[list, np.ndarray] = None):
        """
        Set the rescaled coordinates into the surface_points categories_df

        Args:
            idx (int, list of int): [s0]

        Returns:

        """
        if idx is None:
            idx = self.orientations.df.index

        self.orientations.df.loc[idx, ['X_r', 'Y_r', 'Z_r']] = self.rescale_orientations(
            self.orientations, self.df.loc['values', 'rescaling factor'], self.df.loc['values', 'centers'], idx=idx)
        return True

    @staticmethod
    def rescale_grid(grid, rescaling_factor, centers: pn.DataFrame):
        new_grid_extent = (grid.regular_grid.extent - np.repeat(centers, 2)) / rescaling_factor + 0.5001
        new_grid_values = (grid.values - centers) / rescaling_factor + 0.5001
        return new_grid_extent, new_grid_values

    def set_rescaled_grid(self):
        """
        Set the rescaled coordinates and extent into a grid object
        """

        self.grid.extent_r, self.grid.values_r = self.rescale_grid(self.grid, self.df.loc['values', 'rescaling factor'],
                                                                   self.df.loc['values', 'centers'])


@setdoc_pro([SurfacePoints.__doc__, Orientations.__doc__, Surfaces.__doc__, Faults.__doc__])
class Structure(object):
    """
    The structure_data class analyse the different lengths of subset in the interface and orientations categories_df
    to pass them to the theano function.

    Attributes:
        surface_points (:class:`SurfacePoints`): [s0]
        orientations (:class:`Orientations`): [s1]
        surfaces (:class:`Surfaces`): [s2]
        faults (:class:`Faults`): [s3]
        df (:class:`pn.DataFrame`):
            * len surfaces surface_points (list): length of each surface/fault in surface_points
            * len series surface_points (list) : length of each series in surface_points
            * len series orientations (list) : length of each series in orientations
            * number surfaces per series (list): number of surfaces per series
            * ...
    Args:
        surface_points (:class:`SurfacePoints`): [s0]
        orientations (:class:`Orientations`): [s1]
        surfaces (:class:`Surfaces`): [s2]
        faults (:class:`Faults`): [s3]

    """

    def __init__(self, surface_points: SurfacePoints, orientations: Orientations, surfaces: Surfaces, faults: Faults):

        self.surface_points = surface_points
        self.orientations = orientations
        self.surfaces = surfaces
        self.faults = faults

        df_ = pn.DataFrame(np.array(['False', 'False', -1, -1, -1, -1, -1, -1, -1],).reshape(1, -1),
                           index=['values'],
                           columns=['isLith', 'isFault',
                                    'number faults', 'number surfaces', 'number series',
                                    'number surfaces per series',
                                    'len surfaces surface_points', 'len series surface_points',
                                    'len series orientations'])

        self.df = df_.astype({'isLith': bool, 'isFault': bool, 'number faults': int,
                              'number surfaces': int, 'number series': int})

        self.update_structure_from_input()

    def __repr__(self):
        return self.df.T.to_string()

    def _repr_html_(self):
        return self.df.T.to_html()

    def update_structure_from_input(self):
        """
        Update all fields dependent on the linked Data objects.

        Returns:
            bool: True
        """
        self.set_length_surfaces_i()
        self.set_series_and_length_series_i()
        self.set_length_series_o()
        self.set_number_of_surfaces_per_series()
        self.set_number_of_faults()
        self.set_number_of_surfaces()
        self.set_is_lith_is_fault()
        return True

    def set_length_surfaces_i(self):
        """
        Set the length of each **surface** on `SurfacePoints` i.e. how many data points are for each surface

        Returns:
            :class:`pn.DataFrame`: df where Structural data is stored

        """
        # ==================
        # Extracting lengths
        # ==================
        # Array containing the size of every surface. SurfacePoints
        lssp = self.surface_points.df.groupby('id')['order_series'].count().values
        lssp_nonzero = lssp[np.nonzero(lssp)]

        self.df.at['values', 'len surfaces surface_points'] = lssp_nonzero

        return self.df

    def set_series_and_length_series_i(self):
        """
        Set the length of each **series** on `SurfacePoints` i.e. how many data points are for each series. Also
        sets the number of series itself.

        Returns:
            :class:`pn.DataFrame`: df where Structural data is stored

        """
        # Array containing the size of every series. SurfacePoints.
        len_series_i = self.surface_points.df['order_series'].value_counts(sort=False).values

        if len_series_i.shape[0] is 0:
            len_series_i = np.insert(len_series_i, 0, 0)

        self.df.at['values', 'len series surface_points'] = len_series_i
        self.df['number series'] = len(len_series_i)
        return self.df

    def set_length_series_o(self):
        """
        Set the length of each **series** on `Orientations` i.e. how many orientations are for each series.

        Returns:
            :class:`pn.DataFrame`: df where Structural data is stored

        """
        # Array containing the size of every series. orientations.
        self.df.at['values', 'len series orientations'] = self.orientations.df['order_series'].value_counts(
            sort=False).values
        return self.df

    def set_number_of_surfaces_per_series(self):
        """
        Set number of surfaces for each series

        Returns:
            :class:`pn.DataFrame`: df where Structural data is stored

        """
        self.df.at['values', 'number surfaces per series'] = self.surface_points.df.groupby('order_series').\
            surface.nunique().values
        return self.df

    def set_number_of_faults(self):
        """
        Set number of faults series. This method in gempy v2 is simply informative

        Returns:
            :class:`pn.DataFrame`: df where Structural data is stored

        """
        # Number of faults existing in the surface_points df
        self.df.at['values', 'number faults'] = self.faults.df['isFault'].sum()
        return self.df

    def set_number_of_surfaces(self):
        """
        Set the number of total surfaces

        Returns:
            :class:`pn.DataFrame`: df where Structural data is stored

        """
        # Number of surfaces existing in the surface_points df
        self.df.at['values', 'number surfaces'] = self.surface_points.df['surface'].nunique()

        return self.df

    def set_is_lith_is_fault(self):
        """
        Check if there is lithologies in the data and/or df. This method in gempy v2 is simply informative

        Returns:
            :class:`pn.DataFrame`: df where Structural data is stored
        """
        self.df['isLith'] = True if self.df.loc['values', 'number series'] >= self.df.loc['values', 'number faults']\
            else False
        self.df['isFault'] = True if self.df.loc['values', 'number faults'] > 0 else False

        return self.df


class Options(object):
    """The class options contains the auxiliary user editable flags mainly independent to the model.

     Attributes:
        df (:class:`pn.DataFrame`): df containing the flags. All fields are pandas categories allowing the user to
         change among those categories.

     """
    def __init__(self):
        df_ = pn.DataFrame(np.array(['float64', 'geology', 'fast_compile', 'cpu', None]).reshape(1, -1),
                           index=['values'],
                           columns=['dtype', 'output', 'theano_optimizer', 'device', 'verbosity'])

        self.df = df_.astype({'dtype': 'category', 'output': 'category',
                              'theano_optimizer': 'category', 'device': 'category',
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

    def modify_options(self, attribute, value):
        """
        Method to modify a given field

        Args:
            attribute (str): Name of the field to modify
            value: new value of the field. It will have to exist in the category in order for pandas to modify it.

        Returns:
            :class:`pn.DataFrame`: df where options data is stored
        """

        assert np.isin(attribute, self.df.columns).all(), 'Valid properties are: ' + np.array2string(self.df.columns)
        self.df.loc['values', attribute] = value
        return self.df

    def default_options(self):
        """
        Set default options.

        Returns:
            bool: True
        """
        self.df['dtype'] = 'float64'
        self.df['output'] = 'geology'
        self.df['theano_optimizer'] = 'fast_compile'
        self.df['device'] = 'cpu'
        return True


@setdoc_pro([Grid.__doc__, Structure.__doc__])
class KrigingParameters(object):
    """
    Class that stores and computes the default values for the kriging parameters used during the interpolation.
    The default values will be computed from the :class:`Grid` and :class:`Structure` linked objects

    Attributes:
        grid (:class:`Grid`): [s0]
        structure (:class:`Structure`): [s1]
        df (:class:`pn.DataFrame`): df containing the kriging parameters.

    Args:
        grid (:class:`Grid`): [s0]
        structure (:class:`Structure`): [s1]
    """

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

    def modify_kriging_parameters(self, attribute: str, value, **kwargs):
        """
        Method to modify a given field

        Args:
            attribute (str): Name of the field to modify
            value: new value of the field. It will have to exist in the category in order for pandas to modify it.
            kwargs:
                * u_grade_sep (str): If drift equations values are `str`, symbol that separates the values.

        Returns:
            :class:`pn.DataFrame`: df where options data is stored
        """

        u_grade_sep = kwargs.get('u_grade_sep', ',')

        assert np.isin(attribute, self.df.columns).all(), 'Valid properties are: ' + np.array2string(self.df.columns)

        if attribute == 'drift equations':
            if type(value) is str:
                value = np.fromstring(value[1:-1], sep=u_grade_sep, dtype=int)
            try:
                assert value.shape[0] is self.structure.df.loc['values', 'len series surface_points'].shape[0]
                self.df.at['values', attribute] = value

            except AssertionError:
                print('u_grade length must be the same as the number of series')

        else:
            self.df.loc['values', attribute] = value

    def str2int_u_grade(self, **kwargs):
        """
        Convert u_grade to ints

        Args:
            **kwargs:
                * u_grade_sep (str): If drift equations values are `str`, symbol that separates the values.

        Returns:

        """
        u_grade_sep = kwargs.get('u_grade_sep', ',')
        value = self.df.loc['values', 'drift equations']
        if type(value) is str:
            value = np.fromstring(value[1:-1], sep=u_grade_sep, dtype=int)
        try:
            assert value.shape[0] is self.structure.df.loc['values', 'len series surface_points'].shape[0]
            self.df.at['values', 'drift equations'] = value

        except AssertionError:
            print('u_grade length must be the same as the number of series')

        return self.df

    def set_default_range(self, extent=None):
        """
        Set default kriging_data range

        Args:
            extent (Optional[float, np.array]): extent used to compute the default range--i.e. largest diagonal. If None
             extent of the linked :class:`Grid` will be used.

        Returns:

        """
        if extent is None:
            extent = self.grid.regular_grid.extent
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
        """
        Set default covariance at 0.

        Args:
            range_var (Optional[float, np.array]): range used to compute the default c_0--i.e. largest diagonal. If None
             the already computed range will be used.

        Returns:

        """
        if range_var is None:
            range_var = self.df['range']

        self.df['$C_o$'] = range_var ** 2 / 14 / 3
        return self.df['$C_o$']

    def set_u_grade(self, u_grade: list = None):
        """
         Set default universal grade. Transform polynomial grades to number of equations

         Args:

             u_grade (list):

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

        # Transforming grade to number of equations
        n_universal_eq = np.zeros_like(u_grade)
        n_universal_eq[u_grade == 0] = 0
        n_universal_eq[u_grade == 1] = 3
        n_universal_eq[u_grade == 2] = 9

        self.df.at['values', 'drift equations'] = n_universal_eq
        return self.df['drift equations']


@setdoc_pro([SurfacePoints.__doc__, Orientations.__doc__, Grid.__doc__, Surfaces.__doc__, Faults.__doc__,
             RescaledData.__doc__, Structure.__doc__, KrigingParameters.__doc__, Options.__doc__])
class AdditionalData(object):
    """
    Container class that encapsulate :class:`Structure`, :class:`KrigingParameters`, :class:`Options` and
     rescaling parameters

    Args:
        surface_points (:class:`SurfacePoints`): [s0]
        orientations (:class:`Orientations`): [s1]
        grid (:class:`Grid`): [s2]
        faults (:class:`Faults`): [s4]
        surfaces (:class:`Surfaces`): [s3]
        rescaling (:class:`RescaledData`): [s5]

    Attributes:
        structure_data (:class:`Structure`): [s6]
        options (:class:`Options`): [s8]
        kriging_data (:class:`Structure`): [s7]
        rescaling_data (:class:`RescaledData`):

    """
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
        """
        Concatenate all linked data frames and transpose them for a nice visualization.

        Returns:
            pn.DataFrame: concatenated and transposed dataframe
        """
        concat_ = pn.concat([self.structure_data.df, self.options.df, self.kriging_data.df, self.rescaling_data.df],
                            axis=1, keys=['Structure', 'Options', 'Kriging', 'Rescaling'])
        return concat_.T

    def update_default_kriging(self):
        """
        Update default kriging values.
        """
        self.kriging_data.set_default_range()
        self.kriging_data.set_default_c_o()
        self.kriging_data.set_u_grade()
        self.kriging_data.df['nugget grad'] = 0.01
        self.kriging_data.df['nugget scalar'] = 1e-6

    def update_structure(self):
        """
        Update fields dependent on input data sucha as structure and universal kriging grade
        """
        self.structure_data.update_structure_from_input()
        self.kriging_data.set_u_grade()
