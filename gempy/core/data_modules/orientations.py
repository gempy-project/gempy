import warnings
from typing import Union

import pandas as pn
import sys

import numpy as np

from ..surfaces import Surfaces
from .surface_points import SurfacePoints
from ..checkers import check_for_nans
from .geometric_data import GeometricData
from ...utils import docstring as ds
from ...utils.meta import _setdoc_pro


@_setdoc_pro([Surfaces.__doc__, ds.coord_ori, ds.surface_sp, ds.pole_vector, ds.orientations])
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
        self._columns_o_1 = ['X', 'Y', 'Z', 'X_c', 'Y_c', 'Z_c', 'G_x', 'G_y', 'G_z', 'dip', 'azimuth', 'polarity',
                             'surface', 'series', 'id', 'order_series', 'isFault']
        self._columns_o_num = ['X', 'Y', 'Z', 'X_c', 'Y_c', 'Z_c', 'G_x', 'G_y', 'G_z', 'dip', 'azimuth', 'polarity']
        self._columns_rend = ['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', 'smooth', 'surface']

        if (np.array(sys.version_info[:2]) <= np.array([3, 6])).all():
            self.df: pn.DataFrame

        self.set_orientations(coord, pole_vector, orientation, surface)

    @_setdoc_pro([ds.coord_ori, ds.surface_sp, ds.pole_vector, ds.orientations])
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
        self.df = pn.DataFrame(columns=['X', 'Y', 'Z', 'X_c', 'Y_c', 'Z_c', 'G_x', 'G_y', 'G_z', 'dip',
                                        'azimuth', 'polarity', 'surface'], dtype=float)

        self.df['surface'] = self.df['surface'].astype('category', copy=True)
        self.df['surface'] = self.df['surface'].cat.set_categories(self.surfaces.df['surface'].values)

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
        self.df['surface'] = self.df['surface'].cat.set_categories(self.surfaces.df['surface'].values)

        self.init_dependent_properties()

        # Add nugget effect
        self.df['smooth'] = 0.01
        assert ~self.df['surface'].isna().any(), 'Some of the surface passed does not exist in the Formation' \
                                                 'object. %s' % self.df['surface'][self.df['surface'].isna()]

    @_setdoc_pro([ds.x, ds.y, ds.z, ds.surface_sp, ds.pole_vector, ds.orientations, ds.idx_sp])
    def add_orientation(self, x, y, z, surface: list[str] | str, pole_vector: Union[list, tuple, np.ndarray] = None,
                        orientation: Union[list, np.ndarray] = None, idx=None):
        """
        Add orientation.

        Args:
            x (float, np.ndarray): [s0]
            y (float, np.ndarray): [s1]
            z (float, np.ndarray): [s2]
            surface (list[str], str): [s3]
            pole_vector (np.ndarray): [s4]
            orientation (np.ndarray): [s5]
            idx (Optional[int, list[int]): [s6]

        Returns:
            Orientations
        """
        if pole_vector is None and orientation is None:
            raise AttributeError('Either pole_vector or orientation must have a value. If both are passed pole_vector'
                                 'has preference')

        max_idx = self.df.index.max()

        if idx is None:
            idx = max_idx
            if idx is np.nan:
                idx = 0
            else:
                idx += 1

        if max_idx is not np.nan:
            self.df.loc[idx] = self.df.loc[max_idx]

        if pole_vector is not None:
            self.df.loc[idx, ['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z']] = np.array([x, y, z, *pole_vector], dtype=float)
            self.calculate_orientations(idx)

            if orientation is not None:
                warnings.warn('If pole_vector and orientation are passed pole_vector is used/')
        elif orientation is not None:
            self.df.loc[idx, ['X', 'Y', 'Z', ]] = np.array([x, y, z], dtype=float)
            self.df.loc[idx, ['azimuth', 'dip', 'polarity']] = np.array(orientation, dtype=float)
            self.calculate_gradient(idx)
        else:
            raise AttributeError('At least pole_vector or orientation should have been passed to reach'
                                 'this point. Check previous condition')
        
        self._add_surface_to_list_from_new_surface_points_or_orientations(idx=idx, surface=surface)
        # if type(idx) is int:
        #     self.df.loc[idx, 'surface'] = surface[0]
        # elif type(idx) is list:
        #     self.df.loc[idx, 'surface'] = surface
        
        self.df.loc[idx, ['smooth']] = 0.01
        
        # create new pandas categories from slef.df.['surface']
        self.df['surface'] = self.df['surface'].astype('category', copy=True)
        self.df['surface'] = self.df['surface'].cat.set_categories(self.surfaces.df['surface'].values)

        self.df['series'] = self.df['series'].astype('category', copy=True)
        self.df['series'] = self.df['series'].cat.set_categories(self.surfaces.df['series'].cat.categories)

        self.map_data_from_surfaces(self.surfaces, 'series', idx=idx)
        self.map_data_from_surfaces(self.surfaces, 'id', idx=idx)
        self.map_data_from_series(self.surfaces.series, 'order_series', idx=idx)

        self.sort_table()
        return self

    @_setdoc_pro()
    def del_orientation(self, idx):
        """Delete orientation

        Args:
            idx: [s_idx_sp]

        Returns:
            :class:`gempy.core.data_modules.geometric_data.Orientations`

        """
        self.df.drop(idx, inplace=True)

    @_setdoc_pro([ds.idx_sp, ds.surface_sp])
    def modify_orientations(self, idx, **kwargs):
        """Allows modification of any of an orientation column at a given index.

        Args:
            idx (int, list[int]): [s0]

        Keyword Args:
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
            :class:`gempy.core.data_modules.geometric_data.Orientations`

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

        # TODO Dep
        keys = list(kwargs.keys())

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

        if values.shape[0] == 1:
            values = np.repeat(values, idx.shape[0])

        # Selecting the properties passed to be modified
        self.df.loc[idx, list(kwargs.keys())] = values.astype('float64')

        if np.isin(list(kwargs.keys()), ['G_x', 'G_y', 'G_z']).any():
            self.calculate_orientations(idx)
        else:
            if np.isin(list(kwargs.keys()), ['azimuth', 'dip', 'polarity']).any():
                self.calculate_gradient(idx)
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

    @_setdoc_pro([SurfacePoints.__doc__])
    def create_orientation_from_surface_points(self, surface_points: SurfacePoints, indices):
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

    @_setdoc_pro([ds.file_path, ds.debug, ds.inplace])
    def read_orientations(self, table_source, debug=False, inplace=True, kwargs_pandas: dict = None, **kwargs):
        """
        Read tabular using pandas tools and if inplace set it properly to the surface points object.

        Args:
            table_source (str, path object, file-like object, or direct data frame): [s0]
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
            assert np.logical_or({coord_x_name, coord_y_name, coord_z_name, dip_name, azimuth_name,
                    polarity_name, surface_name}.issubset(table.columns),
                 {coord_x_name, coord_y_name, coord_z_name, g_x_name, g_y_name, g_z_name,
                  polarity_name, surface_name}.issubset(table.columns)), \
                "One or more columns do not match with the expected values, which are: \n" +\
                "- the locations of the measurement points '{}','{}' and '{}' \n".format(coord_x_name,coord_y_name,
                                                                                         coord_z_name)+ \
                "- EITHER '{}' (trend direction indicated by an angle between 0 and 360 with North at 0 AND " \
                "'{}' (inclination angle, measured from horizontal plane downwards, between 0 and 90 degrees) \n".format(
                azimuth_name, dip_name) +\
                "- OR the pole vectors of the orientation in a cartesian system '{}','{}' and '{}' \n".format(g_x_name,
                                                                                                              g_y_name,
                                                                                                              g_z_name)+\
                "- the '{}' of the orientation, can be normal (1) or reversed (-1) \n".format(polarity_name)+\
                "- the name of the surface: '{}'\n".format(surface_name)+\
                "Your headers are "+str(list(table.columns))

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
