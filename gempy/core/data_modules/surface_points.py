import warnings
from typing import Union

import numpy as np
import pandas as pn

from .geometric_data import GeometricData
from ..surfaces import Surfaces
from ...utils import docstring as ds
from ...utils.meta import _setdoc_pro


@_setdoc_pro([Surfaces.__doc__, ds.coord, ds.surface_sp])
class SurfacePoints(GeometricData):
    """
    Data child with specific methods to manipulate interface data. It is initialize without arguments to give
    flexibility to the origin of the data.

    Args:
        surfa ces (:class:`Surfaces`): [s0]
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

        self._columns_i_1 = ['X', 'Y', 'Z', 'X_c', 'Y_c', 'Z_c', 'surface', 'series', 'id',
                             'order_series', 'isFault', 'smooth']

        self._columns_rep = ['X', 'Y', 'Z', 'surface', 'series']
        self._columns_i_num = ['X', 'Y', 'Z', 'X_c', 'Y_c', 'Z_c']
        self._columns_rend = ['X', 'Y', 'Z', 'smooth', 'surface']
        self.set_surface_points(coord, surface)

        self.df['order_series'] = self.df['order_series'].astype('int')
        self.df['id'] = self.df['id'].astype('int')

    @_setdoc_pro([ds.coord, ds.surface_sp])
    def set_surface_points(self, coord: np.ndarray = None, surface: list = None):
        """
        Set coordinates and surface columns on the df.

        Args:
            coord (np.ndarray): [s0]
            surface (list[str]): [s1]

        Returns:
            :class:`SurfacePoints`
        """
        self.df = pn.DataFrame(columns=['X', 'Y', 'Z', 'X_c', 'Y_c', 'Z_c', 'surface'], dtype=float)

        if coord is not None and surface is not None:
            self.df[['X', 'Y', 'Z']] = pn.DataFrame(coord)
            self.df['surface'] = surface

        self.df['surface'] = self.df['surface'].astype('category', copy=True)
        self.df['surface'] = self.df['surface'].cat.set_categories(self.surfaces.df['surface'].values)

        # Choose types
        self.init_dependent_properties()

        # Add nugget columns
        self.df['smooth'] = 2e-6

        assert ~self.df['surface'].isna().any(), 'Some of the surface passed does not exist in the Formation' \
                                                 'object. %s' % self.df['surface'][self.df['surface'].isna()]

        return self

    @_setdoc_pro([ds.x, ds.y, ds.z, ds.surface_sp, ds.idx_sp])
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
           :class:`gempy.core.data_modules.geometric_data.SurfacePoints`

        """
        max_idx = self.df.index.max()

        if idx is None:
            idx = max_idx
            if idx is np.nan:
                idx = 0
            else:
                idx += 1

        if max_idx is not np.nan:
            self.df.loc[idx] = self.df.loc[max_idx]

        coord_array = np.array([x, y, z])
        assert coord_array.ndim == 1, 'Adding an interface only works one by one.'

        try:
            if self.surfaces.df.groupby('isBasement').get_group(True)['surface'].isin(surface).any():
                warnings.warn('Surface Points for the basement will not be used. Maybe you are missing an extra'
                              'layer at the bottom of the pile.')

            self.df.loc[idx, ['X', 'Y', 'Z']] = coord_array.astype('float64')
            
            # Check if elements in surface are categories in self.df['surface'] and if not add them
            #if not self.df['surface'].cat.categories.isin(surface).all():

            self._add_surface_to_list_from_new_surface_points_or_orientations(idx, surface)

        except TypeError as error:  # ToDO test this
            self.del_surface_points(idx)
            raise TypeError('The surface passed does not exist in the pandas categories. This may imply that'
                  'does not exist in the surface object either.')

        self.df.loc[idx, ['smooth']] = 1e-6
        
        self.df['surface'] = self.df['surface'].astype('category', copy=True)
        self.df['surface'] = self.df['surface'].cat.set_categories(self.surfaces.df['surface'].values)

        self.df['series'] = self.df['series'].astype('category', copy=True)
        self.df['series'] = self.df['series'].cat.set_categories(self.surfaces.df['series'].cat.categories)

        self.map_data_from_surfaces(self.surfaces, 'series', idx=idx)
        self.map_data_from_surfaces(self.surfaces, 'id', idx=idx)
        self.map_data_from_series(self.surfaces.series, 'order_series', idx=idx)

        self.sort_table()
        return self, idx

    
    @_setdoc_pro([ds.idx_sp])
    def del_surface_points(self, idx: Union[int, list, np.ndarray]):
        """Delete surface points.

        Args:
            idx (int, list[int]): [s0]

        Returns:
            :class:`gempy.core.data_modules.geometric_data.SurfacePoints`

        """
        self.df.drop(idx, inplace=True)
        return self

    @_setdoc_pro([ds.idx_sp, ds.x, ds.y, ds.z, ds.surface_sp])
    def modify_surface_points(self, idx: Union[int, list, np.ndarray], **kwargs):
        """Allows modification of the x,y and/or z-coordinates of an interface at specified dataframe index.

         Args:
             idx (int, list, np.ndarray): [s0]
             **kwargs:
                * X: [s1]
                * Y: [s2]
                * Z: [s3]
                * surface: [s4]

         Returns:
            :class:`gempy.core.data_modules.geometric_data.SurfacePoints`

         """
        idx = np.array(idx, ndmin=1)
        try:
            surface_names = kwargs.pop('surface')
            self.df.loc[idx, ['surface']] = surface_names
            self.map_data_from_surfaces(self.surfaces, 'series', idx=idx)
            self.map_data_from_surfaces(self.surfaces, 'id', idx=idx)
            self.map_data_from_series(self.surfaces.series, 'order_series', idx=idx)
            self.sort_table()
        except KeyError:
            pass

        # keys = list(kwargs.keys())
    #    is_surface = np.isin('surface', keys).all()

        # Check idx exist in the df
        assert np.isin(np.atleast_1d(idx), self.df.index).all(), 'Indices must exist in the' \
                                                                 ' dataframe to be modified.'

        # Check the properties are valid
        assert np.isin(list(kwargs.keys()), ['X', 'Y', 'Z', 'surface', 'smooth']).all(),\
            'Properties must be one or more of the following: \'X\', \'Y\', \'Z\', ' '\'surface\''
        # stack properties values
        values = np.array(list(kwargs.values()))

        # If we pass multiple index we need to transpose the numpy array
        if type(idx) is list or type(idx) is np.ndarray:
            values = values.T

        # Selecting the properties passed to be modified
        if values.shape[0] == 1:
            values = np.repeat(values, idx.shape[0])

        self.df.loc[idx, list(kwargs.keys())] = values

        return self

    @_setdoc_pro([ds.file_path, ds.debug, ds.inplace])
    def read_surface_points(self, table_source, debug=False, inplace=False,
                            kwargs_pandas: dict = None, **kwargs, ):
        """
        Read tabular using pandas tools and if inplace set it properly to the surface points object.

        Parameters:
            table_source (str, path object, file-like object or direct pandas data frame): [s0]
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
        # TODO read by default either formation or surface

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

        if isinstance(table_source, pn.DataFrame):
            table = table_source
        else:
            table = pn.read_csv(table_source, **kwargs_pandas)

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
