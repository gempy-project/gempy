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
from scipy.constants import G


class GravityPreprocessing(object):
    def __init__(self, interp_data, ai_extent, ai_resolution, ai_z=None, range_max=None):

        self.interp_data = interp_data
        self.ai_extent = np.array(ai_extent)
        self.ai_resolution = np.array(ai_resolution)
        self.model_grid = interp_data.geo_data_res.grid.values

        self.eu = self.compile_eu_f()

        if ai_z is None:
            ai_z = self.model_grid[:, 2].max()

        self.airborne_plane = self.set_airborne_plane(ai_z, self.ai_resolution)

        self.model_resolution = interp_data.geo_data_res.resolution[0] * \
                                interp_data.geo_data_res.resolution[1] * \
                                interp_data.geo_data_res.resolution[2]
        self.vox_size = self.set_vox_size()


        if range_max is None:
            self.range_max = self.default_range()
        else:
            self.range_max = range_max

        # Boolean array that select the voxels that affect each measurement. Size is measurement times resolution
        self.b_all = np.zeros((0, self.model_resolution), dtype=bool)

    def compute_gravity(self, n_chunck_o=25):
        # TODO this function sucks
        # Init
        i_0 = 0
        n_measurements = self.ai_resolution[0] * self.ai_resolution[1]
        loop_list = np.linspace(0, n_measurements, int(n_measurements/n_chunck_o)+1,
                                      endpoint=True, dtype=int)

        n_chunck_l = loop_list[1:] - loop_list[:-1]

        for e, i_1 in enumerate(loop_list[1:]):

            n_chunck = n_chunck_l[e]
            # print(i_0, i_1)
            # Select the number of measurements to compute in this iteration
            airborne_plane_s = self.airborne_plane[i_0:i_1]
            airborne_plane_s[:, 2] += 0.002
            dist = self.eu(airborne_plane_s, self.model_grid)

            # Boolean selection
            b = dist < self.range_max

            # Release memory
            del dist

            # Save selection
            self.b_all = np.vstack((self.b_all, b))

            # Compute cartesian distances from measurements to each voxel

            model_grid_rep = np.repeat(self.model_grid, n_chunck, axis=1)
            s_gr_x = (
                model_grid_rep[:, :n_chunck].T[b].reshape(n_chunck, -1) -
                airborne_plane_s[:, 0].reshape(n_chunck, -1)).astype('float')
            s_gr_y = (
                model_grid_rep[:, n_chunck:2*n_chunck].T[b].reshape(n_chunck, -1) -
                airborne_plane_s[:, 1].reshape(n_chunck, -1)).astype('float')
            s_gr_z = (
                model_grid_rep[:, 2*n_chunck:].T[b].reshape(n_chunck, -1) -
                airborne_plane_s[:, 2].reshape(n_chunck, -1)).astype('float')

            # getting the coordinates of the corners of the voxel...
            x_cor = np.stack((s_gr_x - self.vox_size[0], s_gr_x + self.vox_size[0]), axis=2)
            y_cor = np.stack((s_gr_y - self.vox_size[1], s_gr_y + self.vox_size[1]), axis=2)
            z_cor = np.stack((s_gr_z - self.vox_size[2], s_gr_z + self.vox_size[2]), axis=2)

            # ...and prepare them for a vectorial op
            x_matrix = np.repeat(x_cor, 4, axis=2)
            y_matrix = np.tile(np.repeat(y_cor, 2, axis=2), (1, 1, 2))
            z_matrix = np.tile(z_cor, (1, 1, 4))

            # Distances to each corner of the voxel
            s_r = np.sqrt(x_matrix ** 2 + y_matrix ** 2 + z_matrix ** 2)

            # This is the vector that determines the sign of the corner of the voxel
            mu = np.array([1, -1, -1, 1, -1, 1, 1, -1])

            # Component z of each voxel
            # We need to rescale it the volume so to the cube and multiply by the gravity constant for G
            tz = (
                 np.sum(- 1 *
                        #G *
                        mu * (
                x_matrix * np.log(y_matrix + s_r) +
                y_matrix * np.log(x_matrix + s_r) -
                z_matrix * np.arctan(x_matrix * y_matrix / (z_matrix * s_r))),
                        axis=2))

            # Stacking the precomputation
            if i_0 == 0:
                tz_all = tz

            else:
                tz_all = np.vstack((tz_all, tz))

            i_0 = i_1

        return tz_all, np.ravel(self.b_all)

    def default_range(self):
        # Max range to select voxels
        range_ = (self.model_grid[:, 2].max() - self.model_grid[:, 2].min()) * 0.9
        return range_

    @staticmethod
    def compile_eu_f():
        # Compile Theano function
        x_1 = T.matrix()
        x_2 = T.matrix()

        sqd = T.sqrt(T.maximum(
            (x_1 ** 2).sum(1).reshape((x_1.shape[0], 1)) +
            (x_2 ** 2).sum(1).reshape((1, x_2.shape[0])) -
            2 * x_1.dot(x_2.T), 0
        ))
        eu = theano.function([x_1, x_2], sqd, allow_input_downcast=True)
        return eu

    def set_airborne_plane(self, z, ai_resolution):

        # TODO Include all in the loop. At the moment I am tiling all grids and is useless
        # Rescale z
        z_res = (z-self.interp_data.centers[2])/self.interp_data.rescaling_factor + 0.5001
        ai_extent_rescaled = (self.ai_extent - np.repeat(self.interp_data.centers, 2)) / \
                              self.interp_data.rescaling_factor + 0.5001

        # Create xy meshgrid
        xy = np.meshgrid(np.linspace(ai_extent_rescaled.iloc[0], ai_extent_rescaled.iloc[1], self.ai_resolution[0]),
                         np.linspace(ai_extent_rescaled.iloc[2], ai_extent_rescaled.iloc[3], self.ai_resolution[1]))
        z = np.ones(self.ai_resolution[0]*self.ai_resolution[1])*z_res

        # Transformation
        xy_ravel = np.vstack(map(np.ravel, xy))
        airborne_plane = np.vstack((xy_ravel, z)).T.astype(self.interp_data.dtype)

        # Now we need to find what point of the grid are the closest to this grid and choose them. This is important in
        # order to obtain regular matrices when we set a maximum range of effect

        # First we compute the distance between the airborne plane to the grid and choose those closer
        i_0 = 0
        for i_1 in np.arange(25, self.ai_resolution[0] * self.ai_resolution[1] + 1 + 25, 25, dtype=int):

            d = self.eu(self.model_grid.astype('float'), airborne_plane[i_0:i_1])

            if i_0 == 0:
                ab_g = self.model_grid[np.argmin(d, axis=0)]
            else:
                ab_g = np.vstack((ab_g, self.model_grid[np.argmin(d, axis=0)]))

            i_0 = i_1

        return ab_g

    def set_vox_size(self):

        x_extent = self.interp_data.geo_data_res.extent[1] - self.interp_data.geo_data_res.extent[0]
        y_extent = self.interp_data.geo_data_res.extent[3] - self.interp_data.geo_data_res.extent[2]
        z_extent = self.interp_data.geo_data_res.extent[5] - self.interp_data.geo_data_res.extent[4]
        vox_size = np.array([x_extent, y_extent, z_extent]) / self.interp_data.geo_data_res.resolution
        return vox_size

