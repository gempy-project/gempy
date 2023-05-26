from typing import Union

import sys

import numpy as np
import pandas as pn

from .colors import Colors
from ..utils.meta import _setdoc_pro


class Surfaces(object):
    """
    Class that contains the surfaces of the model and the values of each of them.

    Args:
        surface_names (list or np.ndarray): list containing the names of the surfaces
        series (:class:`Series`): [s0]
        values_array (np.ndarray): 2D array with the values of each surface
        properties names (list or np.ndarray): list containing the names of each properties

    Attributes:
        df (:class:`pn.core.frame.DataFrames`): Pandas data frame containing the surface names mapped to series and
            the value used for each voxel in the final model.
        series (:class:`Series`)
        colors (:class:`Colors`)

    """

    def __init__(self, series, surface_names=None, values_array=None, properties_names=None):

        self._columns = ['surface', 'series', 'order_surfaces',
                         'isBasement', 'isFault', 'isActive', 'hasData', 'color',
                         'vertices', 'edges', 'sfai', 'id']

        self._columns_vis_drop = ['vertices', 'edges', 'sfai', 'isBasement', 'isFault',
                                  'isActive', 'hasData']
        self._n_properties = len(self._columns) - 1
        self.series = series
        self.colors = Colors(self)

        df_ = pn.DataFrame(columns=self._columns)
        self.df = df_.astype({'surface': str, 'series': 'category',
                              'order_surfaces': int,
                              'isBasement': bool, 'isFault': bool, 'isActive': bool, 'hasData': bool,
                              'color': bool, 'id': int, 'vertices': object, 'edges': object})

        if (np.array(sys.version_info[:2]) <= np.array([3, 6])).all():
            self.df: pn.DataFrame

        self.df['series'] = self.df['series'].cat.add_categories(['Default series'])
        if surface_names is not None:
            self.set_surfaces_names(surface_names)

        if values_array is not None:
            self.set_surfaces_values(values_array=values_array,
                                     properties_names=properties_names)

    def __repr__(self):
        c_ = self.df.columns[~(self.df.columns.isin(self._columns_vis_drop))]

        return self.df[c_].to_string()

    def _repr_html_(self):
        c_ = self.df.columns[~(self.df.columns.isin(self._columns_vis_drop))]

        return self.df[c_].style.applymap(self.background_color, subset=['color']).to_html()

    @property
    def properties_val(self):
        all_col = self.df.columns
        prop_cols = all_col.drop(self._columns)
        return prop_cols.insert(0, 'id')

    @property
    def basement(self):
        return self.df['surface'][self.df['isBasement']]

    def update_id(self, id_list: list = None):
        """
        Set id of the layers (1 based)
        Args:
            id_list (list):

        Returns:
             :class:`Surfaces`:

        """
        self.map_faults()
        if id_list is None:
            # This id is necessary for the faults
            id_unique = self.df.reset_index().index + 1

        self.df['id'] = id_unique

        return self

    def map_faults(self):
        self.df['isFault'] = self.df['series'].map(self.series.faults.df['isFault'])

    @staticmethod
    def background_color(value):
        if isinstance(value, str):
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
        if isinstance(surfaces_list, (list, np.ndarray)):
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
            :class:`gempy.core.data.Surfaces`
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

    @_setdoc_pro([update_id.__doc__, pn.DataFrame.drop.__doc__])
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

    def rename_surfaces(self, to_replace: Union[str, list, dict], **kwargs):
        """Replace values given in to_replace with value.

        Args:
            to_replace (str, regex, list, dict, Series, int, float, or None) –
             How to find the values that will be replaced.
            **kwargs:

        Returns:
            :class:`gempy.core.data.Surfaces`

        See Also:
            :any:`pandas.Series.replace`

        """
        if np.isin(to_replace, self.df['surface']).any():
            print('Two surfaces cannot have the same name.')
        else:
            self.df['surface'].replace(to_replace, inplace=True, **kwargs)
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
            :class:`gempy.core.data.Surfaces`

        """
        if series_name is None:
            series_name = self.df.loc[idx, 'series']

        group = self.df.groupby('series').get_group(series_name)['order_surfaces']
        assert np.isin(new_value, group), 'new_value must exist already in the order_surfaces group.'
        old_value = group[idx]
        self.df.loc[group.index.astype('int'), 'order_surfaces'] = group.replace([new_value, old_value], [old_value, new_value])
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
        Method to map which series every surface belongs to. This step is necessary to assign differenct tectonics
        such as unconformities or faults.


        Args:
            mapping_object (dict, :class:`pn.DataFrame`):
                * dict: keys are the series and values the surfaces belonging to that series

                * pn.DataFrame: Dataframe with surfaces as index and a column series with the corresponding series name
                  of each surface

        Returns:
             :class:`Surfaces`

        """

        # Updating surfaces['series'] categories
        self.df['series'] = self.df['series'].cat.set_categories(self.series.df.index)
        # TODO Fixing this. It is overriding the formations already mapped
        if mapping_object is not None:
            # If none is passed and series exists we will take the name of the first series as a default

            if type(mapping_object) is dict:

                s = []
                f = []
                for k, v in mapping_object.items():
                    for form in np.atleast_1d(v):
                        s.append(k)
                        f.append(form)

                new_series_mapping = pn.DataFrame(list(zip(pn.Categorical(s, self.series.df.index))),
                                                 index=f, columns=['series'])

            elif isinstance(mapping_object, pn.Categorical):
                # This condition is for the case we have surface on the index and in 'series' the category
                # TODO Test this
                new_series_mapping = mapping_object

            else:
                raise AttributeError(str(type(mapping_object)) + ' is not the right attribute type.')

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
        """Add values to be interpolated for each surfaces.

        Args:
            values_array (np.ndarray, list): array-like of the same length as number of surfaces. This functionality
             can be used to assign different geophysical properties to each layer
            properties_names (list): list of names for each values_array columns. This must be of the same size as
             values_array axis 1. By default properties will take the column name: 'value_X'.

        Returns:
            :class:`gempy.core.data.Surfaces`

        """
        values_array = np.atleast_2d(values_array)
        properties_names = np.atleast_1d(properties_names)
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
        """Delete a property or several properties column.

        Args:
            properties_names (str, list[str]): Name of the property to delete

        Returns:
             :class:`gempy.core.data.Surfaces`

        """

        properties_names = np.asarray(properties_names)
        self.df.drop(properties_names, axis=1, inplace=True)
        return True

    def set_surfaces_values(self, values_array: Union[np.ndarray, list], properties_names: list = np.empty(0)):
        """Set values to be interpolated for each surfaces. This method will delete the previous values.

        Args:
            values_array (np.ndarray, list): array-like of the same length as number of surfaces. This functionality
             can be used to assign different geophysical properties to each layer
            properties_names (list): list of names for each values_array columns. This must be of same size as
             values_array axis 1. By default properties will take the column name: 'value_X'.

        Returns:
             :class:`gempy.core.data.Surfaces`

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
            properties_names (str, list[str]):
            values (float, np.ndarray):

        Returns:
             :class:`gempy.core.data.Surfaces`

        """
        properties_names = np.atleast_1d(properties_names)
        assert ~np.isin(properties_names, ['surface', 'series', 'order_surfaces', 'id', 'isBasement', 'color']), \
            'only property names can be modified with this method'

        self.df.loc[idx, properties_names] = values
        return self
