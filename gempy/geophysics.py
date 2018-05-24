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

#just testing
class Preprocessing(object):
    def __init__(self, interp_data, ai_extent, ai_resolution, ai_z=None, range_max=None):

        self.interp_data = interp_data
        self.ai_extent = np.array(ai_extent)
        self.ai_resolution = np.array(ai_resolution)
        self.model_grid = interp_data.geo_data_res.x_to_interp_given

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
        self.num = self.ai_resolution[0] * self.ai_resolution[1]

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

    def compute_magnetic(self, n_chunck_o=25):
        for i in range(self.num/n_chunck_o):
            airborne_plane_s = self.airborne_plane[i:(i+1)*n_chunck_o]
            dist = self.eu(airborne_plane_s, self.model_grid)
            b = dist < self.range_max
            del dist
            self.b_all = np.vstack((self.b_all, b))

            model_grid_rep = np.repeat(self.model_grid, n_chunck_o, axis=1)
            xn = model_grid_rep[:,:n_chunck_o][self.b_all[0:n_chunck_o].T].reshape(n_chunck_o,-1)
            yn = model_grid_rep[:,n_chunck_o:2*n_chunck_o][self.b_all[0:n_chunck_o].T].reshape(n_chunck_o,-1)
            zn = model_grid_rep[:,2*n_chunck_o:][self.b_all[0:n_chunck_o].T].reshape(n_chunck_o,-1)

            Xn = np.stack((xn - self.vox_size[0] / 2, xn + self.vox_size[0] / 2), axis=2)
            Yn = np.stack((yn - self.vox_size[1] / 2, yn + self.vox_size[1] / 2), axis=2)
            Zn = np.stack((zn - self.vox_size[2] / 2, zn + self.vox_size[2] / 2), axis=2)

            vx, vy, vz = get_V_mat(self, Xn, Yn, Zn, airborne_plane_s, n_chunck_o)
            # Stacking the precomputation
            if i == 0:
                vx_all = vx
                vy_all = vy
                vz_all = vz

            else:
                vx_all = np.vstack((vx_all, vx))
                vy_all = np.vstack((vy_all, vy))
                vz_all = np.vstack((vz_all, vz))

        return vx_all, vy_all, vz_all, np.ravel(self.b_all)

    def get_T_mat(self, Xn, Yn, Zn, rxLoc, n_chunck_o):
        eps = 1e-10  # add a small value to the locations to avoid /0

        nC = Xn.shape[1]

        # Pre-allocate space for 1D array
        Tx = np.zeros((n_chunck_o, 3 * nC))
        Ty = np.zeros((n_chunck_o, 3 * nC))
        Tz = np.zeros((n_chunck_o, 3 * nC))

        dz2 = np.repeat(rxLoc[:,2], nC).reshape(n_chunck_o,-1) - Zn[:, :, 0]
        dz1 = np.repeat(rxLoc[:,2], nC).reshape(n_chunck_o,-1) - Zn[:, :, 1]

        dy2 = Yn[:, :, 1] - np.repeat(rxLoc[:, 1], nC).reshape(n_chunck_o, -1)
        dy1 = Yn[:, :, 0] - np.repeat(rxLoc[:, 1], nC).reshape(n_chunck_o, -1)

        dx2 = Xn[:, :, 1] - np.repeat(rxLoc[:, 0], nC).reshape(n_chunck_o, -1)
        dx1 = Xn[:, :, 0] - np.repeat(rxLoc[:, 0], nC).reshape(n_chunck_o, -1)

        R1 = (dy2 ** 2 + dx2 ** 2) + eps
        R2 = (dy2 ** 2 + dx1 ** 2) + eps
        R3 = (dy1 ** 2 + dx2 ** 2) + eps
        R4 = (dy1 ** 2 + dx1 ** 2) + eps

        arg1 = np.sqrt(dz2 ** 2 + R2)
        arg2 = np.sqrt(dz2 ** 2 + R1)
        arg3 = np.sqrt(dz1 ** 2 + R1)
        arg4 = np.sqrt(dz1 ** 2 + R2)
        arg5 = np.sqrt(dz2 ** 2 + R3)
        arg6 = np.sqrt(dz2 ** 2 + R4)
        arg7 = np.sqrt(dz1 ** 2 + R4)
        arg8 = np.sqrt(dz1 ** 2 + R3)

        Tx[:, 0:nC] = np.arctan2(dy1 * dz2, (dx2 * arg5)) + \
                      - np.arctan2(dy2 * dz2, (dx2 * arg2)) + \
                      np.arctan2(dy2 * dz1, (dx2 * arg3)) + \
                      - np.arctan2(dy1 * dz1, (dx2 * arg8)) + \
                      np.arctan2(dy2 * dz2, (dx1 * arg1)) + \
                      - np.arctan2(dy1 * dz2, (dx1 * arg6)) + \
                      np.arctan2(dy1 * dz1, (dx1 * arg7)) + \
                      - np.arctan2(dy2 * dz1, (dx1 * arg4))

        Ty[0, 0:nC] = np.log((dz2 + arg2) / (dz1 + arg3)) + \
                      -np.log((dz2 + arg1) / (dz1 + arg4)) + \
                      np.log((dz2 + arg6) / (dz1 + arg7)) + \
                      -np.log((dz2 + arg5) / (dz1 + arg8))

        Ty[0, nC:2 * nC] = np.arctan2(dx1 * dz2, (dy2 * arg1)) + \
                           - np.arctan2(dx2 * dz2, (dy2 * arg2)) + \
                           np.arctan2(dx2 * dz1, (dy2 * arg3)) + \
                           - np.arctan2(dx1 * dz1, (dy2 * arg4)) + \
                           np.arctan2(dx2 * dz2, (dy1 * arg5)) + \
                           - np.arctan2(dx1 * dz2, (dy1 * arg6)) + \
                           np.arctan2(dx1 * dz1, (dy1 * arg7)) + \
                           - np.arctan2(dx2 * dz1, (dy1 * arg8))

        R1 = (dy2 ** 2 + dz1 ** 2) + eps
        R2 = (dy2 ** 2 + dz2 ** 2) + eps
        R3 = (dy1 ** 2 + dz1 ** 2) + eps
        R4 = (dy1 ** 2 + dz2 ** 2) + eps

        Ty[0, 2 * nC:] = np.log((dx1 + np.sqrt(dx1 ** 2 + R1)) /
                                (dx2 + np.sqrt(dx2 ** 2 + R1))) + \
                         -np.log((dx1 + np.sqrt(dx1 ** 2 + R2)) / (dx2 + np.sqrt(dx2 ** 2 + R2))) + \
                         np.log((dx1 + np.sqrt(dx1 ** 2 + R4)) / (dx2 + np.sqrt(dx2 ** 2 + R4))) + \
                         -np.log((dx1 + np.sqrt(dx1 ** 2 + R3)) / (dx2 + np.sqrt(dx2 ** 2 + R3)))

        R1 = (dx2 ** 2 + dz1 ** 2) + eps
        R2 = (dx2 ** 2 + dz2 ** 2) + eps
        R3 = (dx1 ** 2 + dz1 ** 2) + eps
        R4 = (dx1 ** 2 + dz2 ** 2) + eps

        Tx[0, 2 * nC:] = np.log((dy1 + np.sqrt(dy1 ** 2 + R1)) /
                                (dy2 + np.sqrt(dy2 ** 2 + R1))) + \
                         -np.log((dy1 + np.sqrt(dy1 ** 2 + R2)) / (dy2 + np.sqrt(dy2 ** 2 + R2))) + \
                         np.log((dy1 + np.sqrt(dy1 ** 2 + R4)) / (dy2 + np.sqrt(dy2 ** 2 + R4))) + \
                         -np.log((dy1 + np.sqrt(dy1 ** 2 + R3)) / (dy2 + np.sqrt(dy2 ** 2 + R3)))

        Tz[0, 2 * nC:] = -(Ty[0, nC:2 * nC] + Tx[0, 0:nC])
        Tz[0, nC:2 * nC] = Ty[0, 2 * nC:]
        Tx[0, nC:2 * nC] = Ty[0, 0:nC]
        Tz[0, 0:nC] = Tx[0, 2 * nC:]

        Tx = Tx / (4 * np.pi)
        Ty = Ty / (4 * np.pi)
        Tz = Tz / (4 * np.pi)

        return Tx, Ty, Tz

    def default_range(self):
        # Max range to select voxels
        range_ = (self.model_grid[:, 2].max() - self.model_grid[:, 2].min())
        return range_

    @staticmethod
    def compile_eu_f():
        # Compile Theano function
        x_1 = T.dmatrix()
        x_2 = T.dmatrix()

        sqd = T.sqrt(T.maximum(
            (x_1 ** 2).sum(1).reshape((x_1.shape[0], 1)) +
            (x_2 ** 2).sum(1).reshape((1, x_2.shape[0])) -
            2 * x_1.dot(x_2.T), 0
        ))
        eu = theano.function([x_1, x_2], sqd, allow_input_downcast=False)
        return eu

    def set_airborne_plane(self, z, ai_resolution):

        # TODO Include all in the loop. At the moment I am tiling all grids and is useless
        # Rescale z
        z_res = (z - self.interp_data.centers[2]) / self.interp_data.rescaling_factor + 0.5001
        ai_extent_rescaled = (self.ai_extent - np.repeat(self.interp_data.centers, 2)) / \
                             self.interp_data.rescaling_factor + 0.5001

        # Create xy meshgrid
        xy = np.meshgrid(np.linspace(ai_extent_rescaled.iloc[0], ai_extent_rescaled.iloc[1], self.ai_resolution[0]),
                         np.linspace(ai_extent_rescaled.iloc[2], ai_extent_rescaled.iloc[3], self.ai_resolution[1]))
        z = np.ones(self.ai_resolution[0] * self.ai_resolution[1]) * z_res

        # Transformation
        xy_ravel = np.vstack(map(np.ravel, xy))
        airborne_plane = np.vstack((xy_ravel, z)).T.astype(self.interp_data.dtype)

        return airborne_plane

    def set_vox_size(self):

        x_extent = self.interp_data.geo_data_res.extent[1] - self.interp_data.geo_data_res.extent[0]
        y_extent = self.interp_data.geo_data_res.extent[3] - self.interp_data.geo_data_res.extent[2]
        z_extent = self.interp_data.geo_data_res.extent[5] - self.interp_data.geo_data_res.extent[4]
        vox_size = np.array([x_extent, y_extent, z_extent]) / self.interp_data.geo_data_res.resolution
        return vox_size

