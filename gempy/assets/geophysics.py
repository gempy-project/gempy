"""
    This file is part of gempy.

    gempy is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    gempy is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with gempy.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import theano
import theano.tensor as T
from gempy.core.grid_modules.grid_types import CenteredGrid


class GravityPreprocessing(CenteredGrid):
    def __init__(self, centered_grid: CenteredGrid = None):

        if centered_grid is None:
            super().__init__()
        elif isinstance(centered_grid, CenteredGrid):
            self.kernel_centers = centered_grid.kernel_centers
            self.kernel_dxyz_right = centered_grid.kernel_dxyz_right
            self.kernel_dxyz_left = centered_grid.kernel_dxyz_left
        self.tz = np.empty(0)

    def set_tz_kernel(self, scale=True, **kwargs):
        if self.kernel_centers.size == 0:
            self.set_centered_kernel(**kwargs)

        grid_values = self.kernel_centers

        s_gr_x = grid_values[:, 0]
        s_gr_y = grid_values[:, 1]
        s_gr_z = grid_values[:, 2]

        # getting the coordinates of the corners of the voxel...
        x_cor = np.stack((s_gr_x - self.kernel_dxyz_left[:, 0], s_gr_x + self.kernel_dxyz_right[:, 0]), axis=1)
        y_cor = np.stack((s_gr_y - self.kernel_dxyz_left[:, 1], s_gr_y + self.kernel_dxyz_right[:, 1]), axis=1)
        z_cor = np.stack((s_gr_z - self.kernel_dxyz_left[:, 2], s_gr_z + self.kernel_dxyz_right[:, 2]), axis=1)

        # ...and prepare them for a vectorial op
        x_matrix = np.repeat(x_cor, 4, axis=1)
        y_matrix = np.tile(np.repeat(y_cor, 2, axis=1), (1, 2))
        z_matrix = np.tile(z_cor, (1, 4))

        s_r = np.sqrt(x_matrix ** 2 + y_matrix ** 2 + z_matrix ** 2)

        # This is the vector that determines the sign of the corner of the voxel
        mu = np.array([1, -1, -1, 1, -1, 1, 1, -1])

        if scale is True:
            #
            G = 6.674e-3 # ugal     cm3⋅g−1⋅s−26.67408e-2 -- 1 m/s^2 to milligal = 100000 milligal
        else:
            from scipy.constants import G

        self.tz = (
            G *
            np.sum(- 1 *
                   mu * (
                           x_matrix * np.log(y_matrix + s_r) +
                           y_matrix * np.log(x_matrix + s_r) -
                           z_matrix * np.arctan(x_matrix * y_matrix / (z_matrix * s_r))),
                   axis=1))

        return self.tz


class MagneticsPreprocessing(CenteredGrid):
    """
    @Nilgün Güdük

    """
    def __init__(self, centered_grid: CenteredGrid = None):

        if centered_grid is None:
            super().__init__()
        elif isinstance(centered_grid, CenteredGrid):
            self.kernel_centers = centered_grid.kernel_centers
            self.kernel_dxyz_right = centered_grid.kernel_dxyz_right
            self.kernel_dxyz_left = centered_grid.kernel_dxyz_left
        self.V = np.empty(0)

    def set_Vs_kernel(self, **kwargs):
        if self.kernel_centers.size == 0:
            self.set_centered_kernel(**kwargs)

        grid_values = self.kernel_centers
        s_gr_x = grid_values[:, 0]
        s_gr_y = grid_values[:, 1]
        s_gr_z = -1 * grid_values[:, 2]  # talwani takes x-axis positive downwards, and gempy negative downwards

        # getting the coordinates of the corners of the voxel...
        x_cor = np.stack((s_gr_x - self.kernel_dxyz_left[:, 0], s_gr_x + self.kernel_dxyz_right[:, 0]), axis=1)
        y_cor = np.stack((s_gr_y - self.kernel_dxyz_left[:, 1], s_gr_y + self.kernel_dxyz_right[:, 1]), axis=1)
        z_cor = np.stack((s_gr_z + self.kernel_dxyz_left[:, 2], s_gr_z - self.kernel_dxyz_right[:, 2]), axis=1)
        # ...and prepare them for a vectorial op
        x_matrix = np.repeat(x_cor, 4, axis=1)
        y_matrix = np.tile(np.repeat(y_cor, 2, axis=1), (1, 2))
        z_matrix = np.tile(z_cor, (1, 4))

        R = np.sqrt(x_matrix ** 2 + y_matrix ** 2 + z_matrix ** 2)  # distance to each corner
        s = np.array([-1, 1, 1, -1, 1, -1, -1, 1])  # gives the sign of each corner: depends on your coordinate system

        # variables V1-6 represent integrals of volume for each voxel
        V1 = np.sum(-1 * s * np.arctan2((y_matrix * z_matrix), (x_matrix * R)), axis=1)
        V2 = np.sum(s * np.log(R + z_matrix), axis=1)
        V3 = np.sum(s * np.log(R + y_matrix), axis=1)
        V4 = np.sum(-1 * s * np.arctan2((x_matrix * z_matrix), (y_matrix * R)), axis=1)
        V5 = np.sum(s * np.log(R + x_matrix), axis=1)
        V6 = np.sum(-1 * s * np.arctan2((x_matrix * y_matrix), (z_matrix * R)), axis=1)

        # contains all the volume integrals (6 x n_kernelvalues)
        V = np.array([V1, V2, V3, V4, V5, V6])
        return V
