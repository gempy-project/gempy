import numpy as np
import pandas as pd
import pandas as pn

from ..surfaces import Surfaces
from ...utils.meta import _setdoc_pro, _setdoc


@_setdoc_pro(Surfaces.__doc__)
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
        c_ = self._columns_rend
        return self.df[c_].to_string()

    def _repr_html_(self):
        c_ = self._columns_rend
        return self.df[c_].to_html()

    def init_dependent_properties(self):
        """Set the defaults values to the columns before gets mapped with the the :class:`Surfaces` attribute. This
        method will get invoked for example when we add a new point."""

        # series
        self.df['series'] = 'Default series'
        self.df['series'] = self.df['series'].astype('category', copy=True)

        self.df['series'] = self.df['series'].cat.set_categories(self.surfaces.df['series'].cat.categories)

        # id
        self.df['id'] = np.nan

        # order_series
        self.df['order_series'] = 1
        return self

    @staticmethod
    @_setdoc(pn.read_csv.__doc__, indent=False)
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

    # @_setdoc_pro(Series.__doc__)
    def set_series_categories_from_series(self, series):
        """set the series categorical columns with the series index of the passed :class:`Series`

        Args:
            series (:class:`Series`): [s0]
        """
        self.df['series'] = self.df['series'].cat.set_categories(series.df.index)
        return True

    def update_series_category(self):
        """Update the series categorical columns with the series categories of the :class:`Surfaces` attribute."""
        self.df['series'] = self.df['series'].cat.set_categories(self.surfaces.df['series'].cat.categories)

        return True

    @_setdoc_pro(Surfaces.__doc__)
    def set_surface_categories_from_surfaces(self, surfaces: Surfaces):
        """set the series categorical columns with the series index of the passed :class:`Series`.

        Args:
            surfaces (:class:`Surfaces`): [s0]

        """

        self.df['surface'] = self.df['surface'].cat.set_categories(surfaces.df['surface'])
        return True

    # @_setdoc_pro(Series.__doc__)
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
        if attribute in ['id', 'order_series']:
            self.df.loc[idx, attribute] = self.df['series'].map(series.df[attribute]).astype(int)

        else:
            self.df.loc[idx, attribute] = self.df['series'].map(series.df[attribute])

        if type(self.df['order_series'].dtype) is pn.CategoricalDtype:

            self.df['order_series'] = self.df['order_series'].cat.remove_unused_categories()
        return self

    @_setdoc_pro(Surfaces.__doc__)
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
        if attribute == 'series':
            if surfaces.df.loc[~surfaces.df['isBasement']]['series'].isna().sum() != 0:
                raise AttributeError('Surfaces does not have the correspondent series assigned. See'
                                     'Surfaces.map_series_from_series.')
            self.df.loc[idx, attribute] = self.df.loc[idx, 'surface'].map(surfaces.df.set_index('surface')[attribute])

        elif attribute in ['id', 'order_series']:
            self.df.loc[idx, attribute] = (self.df.loc[idx, 'surface'].map(surfaces.df.set_index('surface')[attribute])).astype(int)
        else:

            self.df.loc[idx, attribute] = self.df.loc[idx, 'surface'].map(surfaces.df.set_index('surface')[attribute])

    def _add_surface_to_list_from_new_surface_points_or_orientations(self, idx, surface: list | str):
        if type(surface) is str: surface = [surface]
        
        # Check if self.df['surface'] is a category
        if not isinstance(self.df['surface'].dtype, pd.CategoricalDtype):
            self.df['surface'] = self.df['surface'].astype('category', copy=True)
            self.df['surface'] = self.df['surface'].cat.set_categories(self.surfaces.df['surface'].values)

        # Check if elements in surface are categories in self.df['surface'] and if not add them
        # for s in surface:
        #     if s not in self.df['surface'].cat.categories:
        #         self.df['surface'] = self.df['surface'].cat.add_categories(s)

        if isinstance(idx, (np.int64, int)):
            self.df.loc[idx, 'surface'] = surface[0]
        elif type(idx) is list:
            self.df.loc[idx, 'surface'] = surface
