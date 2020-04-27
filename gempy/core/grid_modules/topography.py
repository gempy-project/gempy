import numpy as np

class Topography:
    """
    Object to include topography in the model.
    """
    def __init__(self, regular_grid):
        self.regular_grid = regular_grid
        self.values = np.zeros((0, 3))

        self.topo = None
        self.values_3D = np.zeros((0, 0, 0))
        self.extent = None
        self.resolution = None

        self.type = None

    def load_from_gdal(self, filepath):
        self.topo = Load_DEM_GDAL(filepath, self.regular_grid)
        self._create_init()
        self._fit2model()
        self.type = 'real'

    def load_random_hills(self, **kwargs):
        self.topo = LoadDEMArtificial(self.regular_grid, **kwargs)
        self._create_init()
        self._fit2model()
        self.type = 'artificial'

    def load_from_saved(self, filepath):
        assert filepath[-4:] == '.npy', 'The file must end on .npy'
        topo = np.load(filepath, allow_pickle=True)
        self.values_3D = topo[0]
        self.extent = topo[1]
        self.resolution = topo[2]
        self._fit2model()
        self.type = 'real'

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
            # todo if grid extent bigger fill missing values with nans for chloe
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
        from gempy.plot.helpers import add_colorbar
        if self.type == 'artificial':
            fig, ax = plt.subplots()
            CS= ax.contour(self.values_3D[:, :, 2], extent=(self.extent[:4]), colors='k', linestyles='solid')
            ax.clabel(CS, inline=1, fontsize=10, fmt='%d')
            CS2 = ax.contourf(self.values_3D[:, :, 2], extent=(self.extent[:4]), cmap='terrain')
            add_colorbar(axes=ax, label='elevation [m]', cs=CS2)
        else:
            im = plt.imshow(np.flipud(self.values_3D[:,:,2]), extent=(self.extent[:4]))
            add_colorbar(im=im, label='elevation [m]')
        plt.axis('scaled')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Model topography')

    def save(self, filepath):
        """
        Save the topography file in a numpy array which can be loaded later, to avoid the gdal process.
        Args:
            filepath (str): path where the array should be stored.

        Returns:

        """
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
        return mask# np.multiply(np.full(self.regular_grid.values.shape, True).T, mask.ravel()).T

    def _find_indices(self):
        zs = np.linspace(self.regular_grid.extent[4], self.regular_grid.extent[5], self.regular_grid.resolution[2])
        dz = (zs[-1] - zs[0]) / len(zs)
        return ((self.values_3D_res[:, :, 2] - zs[0]) / dz + 1).astype(int)

    def interpolate_zvals_at_xy(self, xy, method='interp2d'):
        """
        Interpolates DEM values on a defined section

        Args:
            :param xy: x (EW) and y (NS) coordinates of the profile
            :param method: interpolation method, 'interp2d' for cubic scipy.interpolate.interp2d
                                             'spline' for scipy.interpolate.RectBivariateSpline
        Returns:
            :return: z values, i.e. topography along the profile
        """
        xj = self.values_3D[:, :, 0][0, :]
        yj = self.values_3D[:, :, 1][:, 0]
        zj = self.values_3D[:, :, 2]

        if method == 'interp2d':
            f = interpolate.interp2d(xj, yj, zj, kind='cubic')
            zi = f(xy[:, 0], xy[:, 1])
            if xy[:, 0][0] <= xy[:, 0][-1] and xy[:, 1][0] <= xy[:, 1][-1]:
                return np.diag(zi)
            else:
                return np.flipud(zi).diagonal()
        else:
            assert xy[:, 0][0] <= xy[:, 0][-1], 'The xy values of the first point must be smaller than second.' \
                                               'Please use interp2d as method argument. Will be fixed.'
            assert xy[:, 1][0] <= xy[:, 1][-1], 'The xy values of the first point must be smaller than second.' \
                                               'Please use interp2d as method argument. Will be fixed.'
            f = interpolate.RectBivariateSpline(xj, yj, zj)
            zi = f(xy[:, 0], xy[:, 1])
            return np.flipud(zi).diagonal()