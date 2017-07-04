"""
bkkbab
"""

import numpy as np
import theano
import theano.tensor as T
from scipy.constants import G


class GeoPhysicsPreprocessing(object):

    # TODO geophysics grid different that modeling grid
    def __init__(self, interp_data, z, ai_extent, res_grav=[5, 5], n_cells=1000, grid=None):
        """

        Args:
            interp_data: Some model metadata such as rescaling factor or dtype
            z:
            res_grav: resolution of the gravity
            n_cells:
            grid: This is the model grid
        """
        self.interp_data = interp_data
        self.res_grav = res_grav
        self.z = z
        self.ai_extent = ai_extent

        self.compile_th_fun()
        self.n_cells = n_cells
        self.vox_size = self.set_vox_size()

        if not grid:
            self.grid = interp_data.data.grid.grid.astype(self.interp_data.dtype)
        else:
            self.grid = grid.astype(self.interp_data.dtype)

        self.airborne_plane = self.set_airborne_plane(z, res_grav)

        # self.closest_cells_index = self.set_closest_cells()
        # self.tz = self.z_decomposition()

    def looping_z_decomp(self, chunk_size):
        n_points_0 = 0
        final_tz = np.zeros((self.n_cells, 0))
        self.closest_cells_all = np.zeros((self.n_cells, 0), dtype=np.int)
        n_chunks = int(self.res_grav[0]*self.res_grav[1]/ chunk_size)

        for n_points_1 in np.linspace(chunk_size, self.res_grav[0]*self.res_grav[1], n_chunks,
                                      endpoint=True, dtype=int):
            self.n_measurements = n_points_1 - n_points_0
            self.airborne_plane_op = self.airborne_plane[n_points_0:n_points_1]
            self.closest_cells_all = np.hstack((self.closest_cells_all, self.set_closest_cells()))
            tz = self.z_decomposition()
            final_tz = np.hstack((final_tz, tz))
            n_points_0 = n_points_1

        return final_tz

    def set_airborne_plane(self, z, res_grav):

        # Rescale z
        z_res = (z-self.interp_data.centers[2])/self.interp_data.rescaling_factor + 0.5001
        ai_extent_rescaled = (self.ai_extent - np.repeat(self.interp_data.centers, 2)) / \
                              self.interp_data.rescaling_factor + 0.5001

        # Create xy meshgrid
        xy = np.meshgrid(np.linspace(ai_extent_rescaled.iloc[0], ai_extent_rescaled.iloc[1], res_grav[0]),
                         np.linspace(ai_extent_rescaled.iloc[2], ai_extent_rescaled.iloc[3], res_grav[1]))
        z = np.ones(res_grav[0]*res_grav[1])*z_res

        # Transformation
        xy_ravel = np.vstack(map(np.ravel, xy))
        airborne_plane = np.vstack((xy_ravel, z)).T.astype(self.interp_data.dtype)

        return airborne_plane

    def compile_th_fun(self):

        # Theano function
        x_1 = T.matrix()
        x_2 = T.matrix()

        sqd = T.sqrt(T.maximum(
            (x_1 ** 2).sum(1).reshape((x_1.shape[0], 1)) +
            (x_2 ** 2).sum(1).reshape((1, x_2.shape[0])) -
            2 * x_1.dot(x_2.T), 0
        ))
        self.eu = theano.function([x_1, x_2], sqd)

    def compute_distance(self):
        # if the resolution is too high is going to consume too much memory


        # Distance
        r = self.eu(self.grid, self.airborne_plane_op)

        return r

    def set_closest_cells(self):

        r = self.compute_distance()

        # This is a integer matrix at least
        self.closest_cells_index = np.argsort(r, axis=0)[:self.n_cells, :]

        # DEP?-- I need to make an auxiliary index for axis 1
        self._axis_1 = np.indices((self.n_cells, self.n_measurements))[1]

        # I think it is better to save it in memory since recompute distance can be too heavy
        self.selected_dist = r[self.closest_cells_index, self._axis_1]


        return self.closest_cells_index

    def select_grid(self):

        selected_grid_x = np.zeros((0, self.n_cells))
        selected_grid_y = np.zeros((0, self.n_cells))
        selected_grid_z = np.zeros((0, self.n_cells))

        # I am going to loop it in order to keep low memory (first loop in gempy?)
        for i in range(self.n_measurements):
            selected_grid_x = np.vstack((selected_grid_x, self.grid[:, 0][self.closest_cells_index[:, i]]))
            selected_grid_y = np.vstack((selected_grid_y, self.grid[:, 1][self.closest_cells_index[:, i]]))
            selected_grid_z = np.vstack((selected_grid_z, self.grid[:, 2][self.closest_cells_index[:, i]]))

        return selected_grid_x.T, selected_grid_y.T, selected_grid_z.T

    def set_vox_size(self):

        x_extent = self.interp_data.extent_rescaled.iloc[1] - self.interp_data.extent_rescaled.iloc[0]
        y_extent = self.interp_data.extent_rescaled.iloc[3] - self.interp_data.extent_rescaled.iloc[2]
        z_extent = self.interp_data.extent_rescaled.iloc[5] - self.interp_data.extent_rescaled.iloc[4]
        vox_size = np.array([x_extent, y_extent, z_extent]) / self.interp_data.data.resolution
        return vox_size

    def z_decomposition(self):

        s_gr_x, s_gr_y, s_gr_z = self.select_grid()
        s_r = np.repeat(np.expand_dims(self.selected_dist, axis=2), 8, axis=2)

        # x_cor = np.expand_dims(np.dstack((s_gr_x - self.vox_size[0], s_gr_x + self.vox_size[0])).T, axis=2)
        # y_cor = np.expand_dims(np.dstack((s_gr_y - self.vox_size[1], s_gr_y + self.vox_size[1])).T, axis=2)
        # z_cor = np.expand_dims(np.dstack((s_gr_z - self.vox_size[2], s_gr_z + self.vox_size[2])).T, axis=2)
        x_cor = np.stack((s_gr_x - self.vox_size[0], s_gr_x + self.vox_size[0]), axis=2)
        y_cor = np.stack((s_gr_y - self.vox_size[1], s_gr_y + self.vox_size[1]), axis=2)
        z_cor = np.stack((s_gr_z - self.vox_size[2], s_gr_z + self.vox_size[2]), axis=2)

        # Now we expand them in the 8 combinations. Equivalent to 3 nested loops
        #  see #TODO add paper
        x_matrix = np.repeat(x_cor, 4, axis=2)
        y_matrix = np.tile(np.repeat(y_cor, 2, axis=2), (1, 1, 2))
        z_matrix = np.tile(z_cor, (1, 1, 4))

        mu = np.array([1, -1, -1, 1, -1, 1, 1, -1])

        tz = np.sum(- G * mu * (
                x_matrix * np.log(y_matrix + s_r) +
                y_matrix * np.log(x_matrix + s_r) -
                z_matrix * np.arctan(x_matrix * y_matrix /
                                    (z_matrix * s_r))), axis=2)

        return tz

    # This has to be also a theano function
    def compute_gravity(self, block):

        block_matrix = np.tile(block, (1, self.res_grav[0] * self.res_grav[1]))
        block_matrix_sel = block_matrix[self.closest_cells_all,
                                        np.indices((self.n_cells, self.res_grav[0] * self.res_grav[1]))[1]]
        grav = block_matrix_sel

        return grav

