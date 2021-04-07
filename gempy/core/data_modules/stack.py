from typing import Union, Iterable

import numpy as np
import pandas as pn

from gempy.utils.meta import _setdoc_pro


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

        self._offset_faults = False

    def __repr__(self):
        return self.df.to_string()

    def _repr_html_(self):
        return self.df.to_html()

    def set_is_fault(self, series_fault: Union[str, list, np.ndarray] = None,
                     toggle=False, offset_faults=False):
        """
        Set a flag to the series that are faults.

        Args:
            series_fault(str, list[str]): Name of the series which are faults
            toggle (bool): if True, passing a name which is already True will set it False.
            offset_faults (bool): If True by default faults offset other faults

        Returns:
            :class:`gempy.core.data_modules.stack.Faults`

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

            self.set_default_faults_relations(offset_faults)
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

    def set_default_faults_relations(self, offset_faults:bool=None):
        if offset_faults is not None:
            self._offset_faults = offset_faults

        offset_faults = self._offset_faults

        try:
            # Update default fault relations
            for a_series in self.df.groupby('isFault').get_group(True).index:
                col_pos = self.faults_relations_df.columns.get_loc(a_series)
                # set the faults offset all younger
                self.faults_relations_df.iloc[col_pos, col_pos + 1:] = True

                if offset_faults is False:
                    # set the faults does not offset the younger faults
                    self.faults_relations_df.iloc[col_pos] = ~self.df['isFault'] & \
                                                             self.faults_relations_df.iloc[col_pos]
            return True

        except KeyError:
            return False

    def set_is_finite_fault(self, series_finite: Union[str, list, np.ndarray] = None, toggle=False):
        """
        Toggles given series' finite fault property.

        Args:
            series_finite (str, list[str]): Name of the series which are finite
            toggle (bool): if True, passing a name which is already True will set it False.

        Returns:
            :class:`gempy.core.data_modules.stack.Faults`
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
                self.df.loc[series_finite, 'isFinite'] = True

        return self

    def set_fault_relation(self, rel_matrix=None):
        """Method to set the df that offset a given sequence and therefore also another fault.

        Args:
            rel_matrix (numpy.array[bool]): 2D Boolean array with boolean logic.
             Rows affect (offset) columns

        Returns:
            :class:`gempy.core.data_modules.stack.Faults.faults_relations_df`

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


@_setdoc_pro(Faults.__doc__)
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
        self.df['isActive'] = False

    def __repr__(self):
        return self.df.to_string()

    def _repr_html_(self):
        return self.df.to_html()

    def reset_order_series(self):
        """
        Reset the column order series to monotonic ascendant values.
        """
        self.df.at[:, 'order_series'] = pn.RangeIndex(1, self.df.shape[0] + 1)

    @_setdoc_pro(reset_order_series.__doc__)
    def set_series_index(self, series_order: Union[list, np.ndarray], reset_order_series=True):
        """
        Rewrite the index of the series df

        Args:
            series_order (list): List with names and order of series.
            reset_order_series (bool): if true [s0]

        Returns:
             :class:`Series`: Series
        """
        if type(series_order) is list or type(series_order) is np.ndarray:
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
            try:
                self.df.loc[c] = [-1, 'Erosion', False, False, False]
            # This is in case someone use the old series
            except ValueError:
                self.df.loc[c] = [-1, 'Erosion', False]
                self.faults.df.loc[c, ['isFault', 'isFinite']] = [False, False]
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
            :class:`gempy.core.data_modules.stack.Stack`
        """
        self.df.loc[series_list, 'BottomRelation'] = bottom_relation

        if self.faults.df.loc[series_list, 'isFault'] is True:
            self.faults.set_is_fault(series_list, toggle=True)

        elif bottom_relation == 'Fault':
            self.faults.df.loc[series_list, 'isFault'] = True
        return self

    @_setdoc_pro(reset_order_series.__doc__)
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
        self.update_faults_index_rename()

        for c in series_list:
            # This is in case someone wants to run the old series
            try:
                self.df.loc[c] = [-1, 'Erosion', False, False, False]
            except ValueError:
                self.df.loc[c] = [-1, 'Erosion', False]
                self.faults.df.loc[c, ['isFault', 'isFinite']] = [False, False]
            self.faults.faults_relations_df.loc[c, c] = False

        self.faults.faults_relations_df.fillna(False, inplace=True)

        if reset_order_series is True:
            self.reset_order_series()
        return self

    @_setdoc_pro([reset_order_series.__doc__, pn.DataFrame.drop.__doc__])
    def delete_series(self, indices: Union[str, Iterable], reset_order_series=True):
        """[s1]

        Args:
            indices (str, list): name or list of names of the series to apply the functionality
            reset_order_series (bool): if true [s0]

        Returns:
            Series
        """
        self.df.drop(indices, inplace=True)
        # If we are using the Stack class it is just one element in memory
        try:
            self.faults.df.drop(indices, inplace=True)
        except KeyError:
            pass

        self.faults.faults_relations_df.drop(indices, axis=0, inplace=True)
        self.faults.faults_relations_df.drop(indices, axis=1, inplace=True)

        idx = self.df.index.remove_unused_categories()
        self.df.index = idx
        self.update_faults_index_rename()

        if reset_order_series is True:
            self.reset_order_series()
        return self

    @_setdoc_pro(pn.CategoricalIndex.rename_categories.__doc__)
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
        self.update_faults_index_rename()

        return self

    @_setdoc_pro([pn.CategoricalIndex.reorder_categories.__doc__, pn.CategoricalIndex.sort_values.__doc__])
    def reorder_series(self, new_categories: Union[list, np.ndarray]):
        """[s0] [s1]

        Args:
            new_categories (list): list with all series names in the desired order.

        Returns:
            Series
        """
        idx = self.df.index.reorder_categories(new_categories).sort_values()
        self.df = self.df.reindex(idx, copy=False)
        self.reset_order_series()
        self.update_faults_index_reorder()

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
        self.update_faults_index_reorder()

        return self

    def sort_series(self):
        self.df.sort_values(by='order_series', inplace=True)
        self.df.index = self.df.index.reorder_categories(self.df.index.to_numpy())

    def update_faults_index_rename(self):
        idx = self.df.index
        self.faults.df.index = idx
        self.faults.faults_relations_df.index = idx
        self.faults.faults_relations_df.columns = idx

        #  This is a hack for qgrid
        #  We need to add the qgrid special columns to categories
        self.faults.faults_relations_df.columns = self.faults.faults_relations_df.columns.add_categories(
            ['index', 'qgrid_unfiltered_index'])

    def update_faults_index_reorder(self):
        idx = self.df.index
        self.faults.df = self.faults.df.reindex(idx, copy=False)
        self.faults.faults_relations_df = self.faults.faults_relations_df.reindex(idx, axis=0)
        self.faults.faults_relations_df = self.faults.faults_relations_df.reindex(idx, axis=1)

        self.faults.faults_relations_df.columns = self.faults.faults_relations_df.columns.add_categories(
            ['index', 'qgrid_unfiltered_index'])
        self.faults.set_default_faults_relations()


class MockFault:
    pass


class Stack(Series, Faults):
    """Class that encapsulates all type of geological features. So far is Series and
          Faults

         Args:
             features_names (Iterable): Names of the features
             fault_features (Iterable): List of features that are faults
             rel_matrix:

    """
    def __init__(self, features_names: Iterable = None, fault_features: Iterable = None,
                 rel_matrix: Iterable = None):

        if features_names is None:
            features_names = ['Default series']

        # Set unique df
        df_ = pn.DataFrame(np.array([[1, np.nan, False, False, False]]),
                               index=pn.CategoricalIndex(features_names, ordered=False),
                               columns=['order_series', 'BottomRelation', 'isActive', 'isFault', 'isFinite'])

        self.df = df_.astype({'order_series': int,
                              'BottomRelation': 'category',
                              'isActive': bool,
                              'isFault': bool,
                              'isFinite': bool})

        self.df['order_series'] = self.df['order_series'].astype(int)
        self.df['BottomRelation'] = pn.Categorical(['Erosion'], categories=['Erosion', 'Onlap', 'Fault'])

        self.faults = self
        self.faults_relations_df = pn.DataFrame(index=pn.CategoricalIndex(['Default series']),
                                                columns=pn.CategoricalIndex(['Default series', '']), dtype='bool')

        self.set_is_fault(series_fault=fault_features)
        self.set_fault_relation(rel_matrix=rel_matrix)
        self.n_faults = 0
        self._offset_faults = False











