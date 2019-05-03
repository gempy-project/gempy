from gempy.utils.create_topography import Load_DEM_artificial, Load_DEM_GDAL
import numpy as np
import skimage
import matplotlib.pyplot as plt
from scipy.constants import G


class RegularGrid:
    def __init__(self, extent=None, resolution=None):
        self.grid_type = 'regurlar grid'
        self.resolution = np.ones((0, 3), dtype='int64')
        self.extent = np.empty(6, dtype='float64')
        self.values = np.empty((0, 3))
        self.mask_topo = np.empty((0,3), dtype=bool)
        if extent is not None and resolution is not None:
            self.set_regular_grid(extent, resolution)
            self.dx, self.dy, self.dz = self.get_dx_dy_dz()

    @staticmethod
    def create_regular_grid_3d(extent, resolution):
        """
        Method to create a 3D regular grid where is interpolated

        Args:
            extent (list):  [x_min, x_max, y_min, y_max, z_min, z_max]
            resolution (list): [nx, ny, nz].

        Returns:
            numpy.ndarray: Unraveled 3D numpy array where every row correspond to the xyz coordinates of a regular grid
        """

        dx, dy, dz = (extent[1] - extent[0]) / resolution[0], (extent[3] - extent[2]) / resolution[0], \
                     (extent[5] - extent[4]) / resolution[0]

        g = np.meshgrid(
            np.linspace(extent[0] + dx / 2, extent[1] - dx / 2, resolution[0], dtype="float64"),
            np.linspace(extent[2] + dy / 2, extent[3] - dy / 2, resolution[1], dtype="float64"),
            np.linspace(extent[4] + dz / 2, extent[5] - dz / 2, resolution[2], dtype="float64"), indexing="ij"
        )

        values = np.vstack(tuple(map(np.ravel, g))).T.astype("float64")
        return values

    def get_dx_dy_dz(self):
        dx = (self.extent[1] - self.extent[0]) / self.resolution[0]
        dy = (self.extent[3] - self.extent[2]) / self.resolution[1]
        dz = (self.extent[5] - self.extent[4]) / self.resolution[2]
        return dx, dy, dz

    def set_regular_grid(self, extent, resolution):
        """
        Set a regular grid into the values parameters for further computations
        Args:
             extent (list):  [x_min, x_max, y_min, y_max, z_min, z_max]
            resolution (list): [nx, ny, nz]
        """

        self.extent = np.asarray(extent, dtype='float64')
        self.resolution = np.asarray(resolution)
        self.values = self.create_regular_grid_3d(extent, resolution)
        self.length = self.values.shape[0]
        return self.values


class CustomGrid:
    def __init__(self, custom_grid: np.ndarray):
        self.values = np.zeros((0, 3))
        self.set_custom_grid(custom_grid)

    def set_custom_grid(self, custom_grid: np.ndarray):
        """
        Give the coordinates of an external generated grid

        Args:
            custom_grid (numpy.ndarray like): XYZ (in columns) of the desired coordinates

        Returns:
              numpy.ndarray: Unraveled 3D numpy array where every row correspond to the xyz coordinates of a regular
               grid
        """
        custom_grid = np.atleast_2d(custom_grid)
        assert type(custom_grid) is np.ndarray and custom_grid.shape[1] is 3, 'The shape of new grid must be (n,3)' \
                                                                              ' where n is the number of points of ' \
                                                                              'the grid'

        self.values = custom_grid
        self.length = self.values.shape[0]
        return self.values


class GravityGrid():
    def __init__(self):
       # Grid.__init__(self)
        self.grid_type = 'irregular_grid'
        self.values = np.empty((0, 3))
        self.kernel_dxyz_left = np.empty((0, 3))
        self.kernel_dxyz_right = np.empty((0, 3))
        self.tz = np.empty((0))

    @staticmethod
    def create_irregular_grid_kernel(resolution, radio):
        if radio is not list or radio is not np.ndarray:
            radio = np.repeat(radio, 3)

        g_ = []
        g_2 = []
        d_ = []
        for xyz in [0, 1, 2]:

            if xyz == 2:
                g_.append(np.geomspace(0.01, 1, int(resolution[xyz])))
                g_2.append((np.concatenate(([0], g_[xyz])) + 0.05) * - radio[xyz]*1.2)
            else:
                g_.append(np.geomspace(0.01, 1, int(resolution[xyz] / 2)))
                g_2.append(np.concatenate((-g_[xyz][::-1], [0], g_[xyz])) * radio[xyz])
            d_.append(np.diff(np.pad(g_2[xyz], 1, 'reflect', reflect_type='odd')))

        g = np.meshgrid(*g_2)
        d_left = np.meshgrid(d_[0][:-1]/2, d_[1][:-1]/2, d_[2][:-1]/2)
        d_right = np.meshgrid(d_[0][1:]/2, d_[1][1:]/2, d_[2][1:]/2)
        kernel_g = np.vstack(tuple(map(np.ravel, g))).T.astype("float64")
        kernel_d_left = np.vstack(tuple(map(np.ravel, d_left))).T.astype("float64")
        kernel_d_right = np.vstack(tuple(map(np.ravel, d_right))).T.astype("float64")
        #
        # g_x =
        # g_y = np.geomspace(0.01, 1, int(resolution[1] / 2))
        # g_z = np.geomspace(0.01, 1, int(resolution[2] / 2))
        # g_x2 = np.concatenate((-g_x[::-1], [0], g_x)) * radio[0]
        # g_y2 = np.concatenate((-g_y[::-1], [0], g_y)) * radio[1]
        # g_z2 = np.concatenate((-g_z[::-1], [0], g_z)) * radio[2]
        #
        #
        #
        # dx = np.gradient(g_x2, edge_order=2)
        # dy = np.gradient(g_y2, edge_order=2)
        # dz = np.gradient(g_z2, edge_order=2)
        #
        # g = np.meshgrid(g_x2, g_y2, g_z2)
        # kernel = np.vstack(tuple(map(np.ravel, g))).T.astype("float64")
        return kernel_g, kernel_d_left, kernel_d_right

    def set_irregular_kernel(self, resolution, radio):
        self.kernel_centers, self.kernel_dxyz_left, self.kernel_dxyz_right = self.create_irregular_grid_kernel(
            resolution, radio)

        return self.kernel_centers
    #
    # def set_airborne_plane(self, z, ai_resolution):
    #
    #     # TODO Include all in the loop. At the moment I am tiling all grids and is useless
    #     # Rescale z
    #     z_res = z  # (z-self.interp_data.centers[2])/self.interp_data.rescaling_factor + 0.5001
    #     ai_extent_rescaled = (self.ai_extent - np.repeat(self.interp_data.centers, 2)) / \
    #                          self.interp_data.rescaling_factor + 0.5001
    #
    #     # Create xy meshgrid
    #     xy = np.meshgrid(np.linspace(ai_extent_rescaled.iloc[0], ai_extent_rescaled.iloc[1], self.ai_resolution[0]),
    #                      np.linspace(ai_extent_rescaled.iloc[2], ai_extent_rescaled.iloc[3], self.ai_resolution[1]))
    #     z = np.ones(self.ai_resolution[0] * self.ai_resolution[1]) * z_res
    #
    #     # Transformation
    #     xy_ravel = np.vstack(map(np.ravel, xy))
    #     airborne_plane = np.vstack((xy_ravel, z)).T.astype(self.interp_data.dtype)


    def set_irregular_grid(self, centers, kernel_centers=None, **kwargs):
        self.values =np.empty((0, 3))
        if kernel_centers is None:
            kernel_centers = self.set_irregular_kernel(**kwargs)

        centers = np.atleast_2d(centers)
        for i in centers:
            self.values = np.vstack((self.values, i + kernel_centers))

        self.length = self.values.shape[0]

    def set_tz_kernel(self, **kwargs):
        if self.kernel_centers.size == 0:
            self.set_irregular_kernel(**kwargs)

        grid_values = self.kernel_centers
       # dx, dy, dz = dxdydz

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

        self.tz = (
            np.sum(- 1 *
                   G *
                   mu * (
                           x_matrix * np.log(y_matrix + s_r) +
                           y_matrix * np.log(x_matrix + s_r) -
                           z_matrix * np.arctan(x_matrix * y_matrix / (z_matrix * s_r))),
                   axis=1))

        return self.tz


class Topography:
    def __init__(self, regular_grid):
        self.regular_grid = regular_grid
        self.values = np.zeros((0, 3))

    def load_from_gdal(self, filepath):
        self.topo = Load_DEM_GDAL(filepath, self.regular_grid)
        self._create_init()
        self._fit2model()

    def load_random_hills(self, **kwargs):
        self.topo = Load_DEM_artificial(self.regular_grid, **kwargs)
        self._create_init()
        self._fit2model()

    def load_from_saved(self, filepath):
        #assert filepath ending is .npy
        assert filepath[-4:] == '.npy', 'The file must end on .npy'
        topo = np.load(filepath)
        self.values_3D = topo[0]
        self.extent = topo[1]
        self.resolution = topo[2]
        self._fit2model()

    def _create_init(self):
        self.values_3D = self.topo.values_3D
        self.extent = self.topo.extent
        self.resolution = self.topo.resolution

    def _fit2model(self):
        self.values = np.vstack((
            self.values_3D[:, :, 0].ravel(), self.values_3D[:, :, 1].ravel(),
            self.values_3D[:, :, 2].ravel())).T.astype("float64")

        if np.any(self.regular_grid.extent[:4] - self.extent) != 0:
            print('obacht')
            self._crop()

        if np.any(self.regular_grid.resolution[:2] - self.resolution) != 0:
            self._resize()
        else:
            self.values_3D_res = self.values_3D

        self.regular_grid.mask_topo = self._create_grid_mask()

    def _crop(self):
        pass

    def _resize(self):
        self.values_3D_res = skimage.transform.resize(self.values_3D,
                                                      (self.regular_grid.resolution[0], self.regular_grid.resolution[1]),
                                                      mode='constant',
                                                      anti_aliasing=False, preserve_range=True)

    def show(self):
        fig, ax = plt.subplots()
        CS= ax.contour(self.values_3D[:, :, 2], extent=(self.extent[:4]), colors='k', linestyles='solid')
        ax.clabel(CS, inline=1, fontsize=10, fmt='%d')
        CS2 = ax.contourf(self.values_3D[:, :, 2], extent=(self.extent[:4]), cmap='terrain')
        cbar = plt.colorbar(CS2)
        cbar.set_label('elevation')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Model topography')

    def save(self, filepath):
        np.save(filepath, np.array([self.values_3D, self.extent, self.resolution]))
        print('saved')


    def _create_grid_mask(self):
        ind = self._find_indices()
        gridz = self.regular_grid.values[:, 2].reshape(*self.regular_grid.resolution).copy()
        for x in range(self.regular_grid.resolution[0]):
            for y in range(self.regular_grid.resolution[1]):
                z = ind[x, y]
                gridz[x, y, z:] = 99999
        mask = (gridz == 99999)
        return mask.swapaxes(0,1)# np.multiply(np.full(self.regular_grid.values.shape, True).T, mask.ravel()).T

    def _find_indices(self):
        zs = np.linspace(self.regular_grid.extent[4], self.regular_grid.extent[5], self.regular_grid.resolution[2])
        dz = (zs[-1] - zs[0]) / len(zs)
        return ((self.values_3D_res[:, :, 2] - zs[0]) / dz + 1).astype(int)

    def _line_in_section(self, direction='y', cell_number=0):
        # todo use slice2D of plotting class for this
        if np.any(self.resolution - self.regular_grid.resolution[:2]) != 0:
            cell_number_res = (self.values_3D.shape[:2] / self.regular_grid.resolution[:2] * cell_number).astype(int)
            cell_number = cell_number_res[0] if direction == 'x' else cell_number_res[1]
        if direction == 'x':
            topoline = self.values_3D[:, cell_number, :][:, [1, 2]].astype(int)
        elif direction == 'y':
            topoline = self.values_3D[cell_number, :, :][:, [0, 2]].astype(int)
        else:
            raise NotImplementedError
        return topoline
