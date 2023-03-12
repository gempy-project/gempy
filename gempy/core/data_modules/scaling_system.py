from typing import Union

import numpy as np
import pandas as pn

from ..grid import Grid
from .surface_points import SurfacePoints
from .orientations import Orientations
from ...utils import docstring as ds
from ...utils.meta import _setdoc_pro


@_setdoc_pro([SurfacePoints.__doc__, Orientations.__doc__, Grid.__doc__])
class ScalingSystem(object):
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

        self.axial_anisotropy = False
        self.max_coord = np.zeros(3)
        self.min_coord = np.zeros(3)
        self.axial_anisotropy_type = 'data'

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

    def toggle_axial_anisotropy(self, type='data'):
        self.axial_anisotropy_type = type
        self.axial_anisotropy = self.axial_anisotropy ^ True
        self.rescale_data()

    @_setdoc_pro([ds.centers, ds.rescaling_factor])
    def modify_rescaling_parameters(self, attribute, value):
        """
        Modify the parameters used to rescale data

        Args:
            attribute (str): Attribute to be modified. It can be: centers, rescaling factor
                * centers: [s0]
                * rescaling factor: [s1]
            value (float, list[float])


        Returns:
            :class:`gempy.core.data_modules.geometric_data.Rescaling`

        """
        assert np.isin(attribute, self.df.columns).all(), 'Valid attributes are: ' + np.array2string(self.df.columns)

        if attribute == 'centers':
            try:
                assert value.shape[0] == 3

                self.df.loc['values', attribute] = value

            except AssertionError:
                print('centers length must be 3: XYZ')

        else:
            self.df.loc['values', attribute] = value

        return self

    @_setdoc_pro([ds.centers, ds.rescaling_factor])
    def rescale_data(self,
                     rescaling_factor=None,
                     centers=None,
                     axial_anisotropy=None
                     ):
        """
        Rescale inplace: surface_points, orientations---adding columns in the categories_df---and grid---adding values_r
        attributes. The rescaled values will get stored on the linked objects.

        Args:
            rescaling_factor: [s1]
            centers: [s0]

        Returns:

        """

        xyz = self.concat_surface_points_orientations(self.surface_points.df[['X', 'Y', 'Z']],
                                                      self.orientations.df[['X', 'Y', 'Z']])

        # This is asking for XYZ parameters
        max_coord, min_coord = self.max_min_coord(xyz)

        if rescaling_factor is None:
            # This is asking for XYZ parameters
            self.df['rescaling factor'] = self.compute_rescaling_factor_for_0_1(max_coord=max_coord,
                                                                                min_coord=min_coord)
        else:
            self.df['rescaling factor'] = rescaling_factor
        if centers is None:
            # This is asking for XYZ parameters
            self.df.at['values', 'centers'] = self.compute_data_center(max_coord=max_coord,
                                                                       min_coord=min_coord)
        else:
            self.df.at['values', 'centers'] = centers

        self.set_rescaled_surface_points(axial_anisotropy=axial_anisotropy)
        self.set_rescaled_orientations(axial_anisotropy=axial_anisotropy)
        self.set_rescaled_grid(axial_anisotropy=axial_anisotropy)
        return True

    def compute_axial_anisotropy(self, type=None, extent=None):
        if type is None:
            type = self.axial_anisotropy_type

        if type == 'data':
            x1, y1, z1 = self.max_coord
            x0, y0, z0 = self.min_coord
        elif type == 'extent':
            if extent is None:
                extent = self.grid.regular_grid.extent

            x0, x1, y0, y1, z0, z1 = extent
        else:
            raise AttributeError('Type must be either data or extent')

        # Calculate average
        x_d = np.linalg.norm(x0-x1)
        y_d = np.linalg.norm(y0-y1)
        z_d = np.linalg.norm(z0-z1)
        mean_d = np.mean([x_d, y_d, z_d])
        return np.array([mean_d/x_d, mean_d/y_d, mean_d/z_d])

    def apply_axial_anisotropy(self, xyz, anisotropy):
        return xyz * anisotropy

    def get_rescaled_surface_points(self):
        """
        Get the rescaled coordinates. return an image of the interface and orientations categories_df with the X_r..
         columns

        Returns:
            :attr:`SurfacePoints.df[['X_c', 'Y_c', 'Z_c']]`
        """
        return self.surface_points.df[['X_c', 'Y_c', 'Z_c']]

    def get_rescaled_orientations(self):
        """
        Get the rescaled coordinates. return an image of the interface and orientations categories_df with the X_r..
         columns.

        Returns:
            :attr:`Orientations.df[['X_c', 'Y_c', 'Z_c']]`
        """
        return self.orientations.df[['X_c', 'Y_c', 'Z_c']]

    @staticmethod
    def concat_surface_points_orientations(surface_points_xyz=None, orientations_xyz=None) \
            -> pn.DataFrame:
        """
        Args:
            surface_points_xyz (:class:`pandas.DataFrame`): [s0]
            orientations_xyz (:class:`pandas.DataFrame`): [s1]
        Returns:

        """

        if surface_points_xyz is None and orientations_xyz is not None:
            df = orientations_xyz
        elif surface_points_xyz is not None and orientations_xyz is None:
            df = surface_points_xyz
        elif surface_points_xyz is not None and orientations_xyz is not None:
            df = pn.concat([orientations_xyz, surface_points_xyz], sort=False)
        else:
            raise AttributeError('You must pass at least one Data object')
        return df

    @_setdoc_pro([SurfacePoints.__doc__, Orientations.__doc__])
    def max_min_coord(self, df):
        """
        Find the maximum and minimum location of any input data in each cartesian coordinate

        Args:
            df

        Returns:
            tuple: max[XYZ], min[XYZ]
        """

        self.max_coord = df.max()[['X', 'Y', 'Z']]
        self.min_coord = df.min()[['X', 'Y', 'Z']]

        return self.max_coord, self.min_coord

    @_setdoc_pro([SurfacePoints.__doc__, Orientations.__doc__, ds.centers])
    def compute_data_center(self,
                            surface_points_xyz=None,
                            orientations_xyz=None,
                            max_coord=None, min_coord=None, inplace=True):
        """
        Calculate the center of the data once it is shifted between 0 and 1.

        Args:
            surface_points_xyz (:class:`pandas.DataFrame`): [s0]
            orientations_xyz (:class:`pandas.DataFrame`): [s1]
            max_coord (float): Max XYZ coordinates of all GeometricData
            min_coord (float): Min XYZ coordinates of all GeometricData
            inplace (bool): if True modify the self.df rescaling factor attribute

        Returns:
            np.array: [s2]
        """

        if max_coord is None or min_coord is None:
            max_coord, min_coord = self.max_min_coord(surface_points_xyz, orientations_xyz)

        # Get the centers of every axis
        centers = ((max_coord + min_coord) / 2).astype(float).values
        if inplace is True:
            self.df.at['values', 'centers'] = centers
        return centers

    @_setdoc_pro([SurfacePoints.__doc__, Orientations.__doc__, ds.rescaling_factor])
    def compute_rescaling_factor_for_0_1(self,
                                         surface_points_xyz=None,
                                         orientations_xyz=None,
                                         max_coord=None, min_coord=None,
                                         inplace=True):
        """
        Calculate the rescaling factor of the data to keep all coordinates between 0 and 1

        Args:
            surface_points_xyz (:class:`pandas.DataFrame`): [s0]
            orientations_xyz (:class:`pandas.DataFrame`): [s1]
            max_coord (float): Max XYZ coordinates of all GeometricData
            min_coord (float): Min XYZ coordinates of all GeometricData
            inplace (bool): if True modify the self.df rescaling factor attribute

        Returns:
            float: [s2]
        """

        if max_coord is None or min_coord is None:
            max_coord, min_coord = self.max_min_coord(surface_points_xyz, orientations_xyz)
        rescaling_factor_val = (2 * np.max(max_coord - min_coord))
        if inplace is True:
            self.df['rescaling factor'] = rescaling_factor_val
        return rescaling_factor_val

    @staticmethod
    @_setdoc_pro([SurfacePoints.__doc__, compute_data_center.__doc__,
                  compute_rescaling_factor_for_0_1.__doc__, ds.idx_sp])
    def rescale_surface_points(surface_points_xyz,
                               rescaling_factor,
                               centers=None,
                               idx: list = None):
        """
        Rescale inplace: surface_points. The rescaled values will get stored on the linked objects.

        Args:
            surface_points_xyz (:class:`pandas.DataFrame`): [s0]
            rescaling_factor: [s2]
            centers: [s1]
            idx (int, list of int): [s3]

        Returns:

        """

        if idx is None:
            idx = surface_points_xyz.index

        # Change the coordinates of surface_points
        new_coord_surface_points = (surface_points_xyz.loc[idx, ['X', 'Y', 'Z']] -
                                    centers) / rescaling_factor + 0.5001

        new_coord_surface_points.rename(columns={"X": "X_c", "Y": "Y_c", "Z": 'Z_c'},
                                        inplace=True)
        return new_coord_surface_points

    @_setdoc_pro(ds.idx_sp)
    def set_rescaled_surface_points(self,
                                    idx: Union[list, np.ndarray] = None,
                                    axial_anisotropy=None):
        """
        Set the rescaled coordinates into the surface_points categories_df

        Args:
            axial_anisotropy:
            idx (int, list of int): [s0]

        Returns:

        """
        if idx is None:
            idx = self.surface_points.df.index
        idx = np.atleast_1d(idx)

        if axial_anisotropy is None:
            axial_anisotropy = self.axial_anisotropy

        if axial_anisotropy is False:
            surface_points_xyz = self.surface_points.df
        else:
            axial_anisotropy_scale = self.compute_axial_anisotropy()
            surface_points_xyz = self.apply_axial_anisotropy(
                self.surface_points.df[['X', 'Y', 'Z']],
                axial_anisotropy_scale)

        self.surface_points.df.loc[idx, ['X_c', 'Y_c', 'Z_c']] = self.rescale_surface_points(
            surface_points_xyz, # This is asking for XYZ parameters
            self.df.loc['values', 'rescaling factor'],
            self.df.loc['values', 'centers'],
            idx=idx)

        return self.surface_points.df.loc[idx, ['X_c', 'Y_c', 'Z_c']]

    def rescale_data_point(self, data_points: np.ndarray, rescaling_factor=None, centers=None):
        """This method now is very similar to set_rescaled_surface_points passing an index

        Notes:
            So far is not used by any function
        """
        if rescaling_factor is None:
            rescaling_factor = self.df.loc['values', 'rescaling factor']
        if centers is None:
            centers = self.df.loc['values', 'centers']

        rescaled_data_point = (data_points - centers) / rescaling_factor + 0.5001

        return rescaled_data_point

    @staticmethod
    @_setdoc_pro([Orientations.__doc__, compute_data_center.__doc__, compute_rescaling_factor_for_0_1.__doc__, ds.idx_sp])
    def rescale_orientations(orientations_xyz, rescaling_factor, centers, idx: list = None):
        """
        Rescale inplace: surface_points. The rescaled values will get stored on the linked objects.

        Args:
            orientations_xyz (:class:`pandas.DataFrame`): [s0]
            rescaling_factor: [s2]
            centers: [s1]
            idx (int, list of int): [s3]

        Returns:

        """
        if idx is None:
            idx = orientations_xyz.index

        # Change the coordinates of orientations
        new_coord_orientations = (orientations_xyz.loc[idx, ['X', 'Y', 'Z']] -
                                  centers) / rescaling_factor + 0.5001

        new_coord_orientations.rename(columns={"X": "X_c", "Y": "Y_c", "Z": 'Z_c'}, inplace=True)

        return new_coord_orientations

    @_setdoc_pro(ds.idx_sp)
    def set_rescaled_orientations(self,
                                  idx: Union[list, np.ndarray] = None,
                                  axial_anisotropy=None
                                  ):
        """
        Set the rescaled coordinates into the surface_points categories_df

        Args:
            axial_anisotropy:
            idx (int, list of int): [s0]

        Returns:

        """
        if idx is None:
            idx = self.orientations.df.index
        idx = np.atleast_1d(idx)

        if axial_anisotropy is None:
            axial_anisotropy = self.axial_anisotropy

        if axial_anisotropy is False:
            orientations_xyz = self.orientations.df
        else:
            axial_anisotropy_scale = self.compute_axial_anisotropy()
            orientations_xyz = self.apply_axial_anisotropy(
                self.orientations.df[['X', 'Y', 'Z']],
                axial_anisotropy_scale)

        self.orientations.df.loc[idx, ['X_c', 'Y_c', 'Z_c']] = self.rescale_orientations(
            orientations_xyz,
            self.df.loc['values', 'rescaling factor'],
            self.df.loc['values', 'centers'],
            idx=idx
        )
        return self.orientations.df.loc[idx, ['X_c', 'Y_c', 'Z_c']]

    @staticmethod
    def rescale_grid(grid_extent, grid_values, rescaling_factor, centers: pn.DataFrame):
        new_grid_extent = (grid_extent - np.repeat(centers, 2)) / rescaling_factor + 0.5001
        new_grid_values = (grid_values - centers) / rescaling_factor + 0.5001
        return new_grid_extent, new_grid_values,

    def set_rescaled_grid(self, axial_anisotropy=None):
        """
        Set the rescaled coordinates and extent into a grid object
        """
        if axial_anisotropy is None:
            axial_anisotropy = self.axial_anisotropy

        # The grid has to be rescaled for having the model in scaled coordinates
        # between 0 and 1 but with the actual proportions
        self.grid.extent_r, self.grid.values_r = self.rescale_grid(
            self.grid.regular_grid.extent,
            self.grid.values,
            self.df.loc['values', 'rescaling factor'],
            self.df.loc['values', 'centers']
        )

        self.grid.regular_grid.extent_r, self.grid.regular_grid.values_r = self.grid.extent_r, self.grid.values_r

        # For the grid

        if axial_anisotropy is True:

            axial_anisotropy_scale = self.compute_axial_anisotropy()

            ani_grid_values = self.apply_axial_anisotropy(
                self.grid.values,
                axial_anisotropy_scale)

            axis_extended_l = self.apply_axial_anisotropy(
                self.grid.regular_grid.extent[[0, 2, 4]],
                axial_anisotropy_scale)

            axis_extended_r = self.apply_axial_anisotropy(
                self.grid.regular_grid.extent[[1, 3, 5]],
                axial_anisotropy_scale)

            ani_grid_extent = np.array([axis_extended_l[0],
                                        axis_extended_r[0],
                                        axis_extended_l[1],
                                        axis_extended_r[1],
                                        axis_extended_l[2],
                                        axis_extended_r[2]])

            self.grid.extent_c, self.grid.values_c = self.rescale_grid(
                ani_grid_extent,
                ani_grid_values,
                self.df.loc['values', 'rescaling factor'],
                self.df.loc['values', 'centers']
            )
        else:
            self.grid.values_c = self.grid.values_r
            self.grid.extent_c = self.grid.extent_r

        return self.grid.values_c
