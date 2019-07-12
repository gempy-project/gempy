from gempy.utils.create_topography import Load_DEM_artificial, Load_DEM_GDAL
import numpy as np
import skimage
import matplotlib.pyplot as plt
from scipy.constants import G
from scipy import interpolate
from gempy.utils.meta import setdoc, setdoc_pro
import gempy.utils.docstring as ds
from typing import Optional
import pandas as pn

class RegularGrid:
    """
    Class with the methods and properties to manage 3D regular grids where the model will be interpolated.

    Args:
        extent (np.ndarray):  [x_min, x_max, y_min, y_max, z_min, z_max]
        resolution (np.ndarray): [nx, ny, nz]

    Attributes:
        extent (np.ndarray):  [x_min, x_max, y_min, y_max, z_min, z_max]
        resolution (np.ndarray): [nx, ny, nz]
        values (np.ndarray): XYZ coordinates
        mask_topo (np.ndarray): TODO @elisa fill
        dx (float): size of the cells on x
        dy (float): size of the cells on y
        dz (float): size of the cells on z

    """
    def __init__(self, extent=None, resolution=None):
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


class Sections:
    def __init__(self, regular_grid, section_dict):
        #todo tidy up
        self.regular_grid = regular_grid
        self.section_dict = section_dict
        self.names = np.array(list(self.section_dict.keys()))

        self.points = []
        self.resolution = []
        self.length = [0]
        self.dist = []
        self.get_section_params()
        self.calculate_distance()
        self.values = []
        self.compute_section_coordinates()

        self.extent = None

    def _repr_html_(self):
        return pn.DataFrame.from_dict(self.section_dict, orient='index', columns=['start', 'stop', 'resolution']).to_html()

    def show(self):
        pass

    def get_section_params(self):
        for i, section in enumerate(self.names):
            points = [self.section_dict[section][0], self.section_dict[section][1]]
            assert points[0] != points[1], 'The start and end points of the section must not be identical.'
            self.points.append(points)
            self.resolution.append(self.section_dict[section][2])
            self.length.append(self.section_dict[section][2][0] * self.section_dict[section][2][1])
        self.length = np.array(self.length).cumsum()

    def calculate_distance(self):
        self.coordinates = np.array(self.points).ravel().reshape(-1, 4) #axis are x1,y1,x2,y2
        self.dist = np.sqrt(np.diff(self.coordinates[:, [0, 2]])**2 + np.diff(self.coordinates[:, [1, 3]])**2)

    def compute_section_coordinates(self):
        for i in range(len(self.names)):
            xy = self.calculate_line_coordinates_2points(self.points[i][0], self.points[i][1], self.resolution[i][0],
                                                         self.resolution[i][0]) #two times xy resolution is correct
            zaxis = np.linspace(self.regular_grid.extent[4], self.regular_grid.extent[5], self.resolution[i][1],
                                     dtype="float64")
            X, Z = np.meshgrid(xy[:, 0], zaxis, indexing='ij')
            Y, _ = np.meshgrid(xy[:, 1], zaxis, indexing='ij')
            xyz = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
            if i == 0:
                self.values = xyz
            else:
                self.values = np.vstack((self.values, xyz))

    def calculate_line_coordinates_2points(self, p1, p2, resx, resy):
        x0 = p1[0]
        x1 = p2[0]
        y0 = p1[1]
        y1 = p2[1]

        dx = np.abs((x1 - x0) / resx)
        dy = np.abs((y1 - y0) / resy)

        if x0 == x1:  # slope is infinite
            # for cases where phi == -np.pi/2 or phi == np.pi/2
            xi = x0 * np.ones(resy)
            yj = np.linspace(y0, y1, resy)
        else:
            # calculate support points between two points
            phi = np.arctan2(y1 - y0, x1 - x0)  # angle of line with x-axis
            if np.pi / 2 < phi <= np.pi: #shift all values to first or fourth quadrant
                phi -= np.pi
            elif -np.pi <= phi < -np.pi / 2:
                phi += np.pi  # shift values in first or fourth quadrant so that cosine is positive
            else:
                pass
            ds = np.abs(dx * np.cos(phi)) + np.abs(dy * np.sin(phi))  # support point spacing
            # abs needed for cases where phi == -1/4 pi or 3/4 pi
            if x0 > x1:
                n_points = np.ceil((x0 - x1) / (ds * np.cos(phi)))
            else:
                n_points = np.ceil((x1 - x0) / (ds * np.cos(phi)))
            xi = np.linspace(x0, x1, int(n_points))
            m = (y1 - y0) / (x1 - x0)  # slope of line
            yj = m * (xi - x0) + y0 * np.ones(xi.shape)  # calculate yvalues with line equation
        return np.vstack((xi, yj)).T


    def get_section_args(self, section_name: str):
        where = np.where(self.names == section_name)[0][0]
        return self.length[where], self.length[where+1]

    def get_section_grid(self, section_name: str):
        l0, l1 = self.get_section_args(section_name)
        return self.values[l0:l1]


class CustomGrid:
    """Object that contains arbitrary XYZ coordinates.

    Args:
        custom_grid (numpy.ndarray like): XYZ (in columns) of the desired coordinates

    Attributes:
        values (np.ndarray): XYZ coordinates
    """
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


class CenteredGrid:
    """
    Logarithmic spaced grid.
    """

    def __init__(self, centers=None, radio=None, resolution=None):
        self.grid_type = 'centered_grid'
        self.values = np.empty((0, 3))
        self.length = self.values.shape[0]
        self.kernel_centers = np.empty((0, 3))
        self.kernel_dxyz_left = np.empty((0, 3))
        self.kernel_dxyz_right = np.empty((0, 3))
        self.tz = np.empty(0)

        if centers is not None and radio is not None:
            if resolution is None:
                resolution = [10, 10, 20]

            self.set_centered_grid(centers=centers, radio=radio, resolution=resolution)

    @staticmethod
    @setdoc_pro(ds.resolution)
    def create_irregular_grid_kernel(resolution, radio):
        """
        Create an isometric grid kernel (centered at 0)

        Args:
            resolution: [s0]
            radio (float): Maximum distance of the kernel

        Returns:
            tuple: center of the voxel, left edge of each voxel (for xyz), right edge of each voxel (for xyz).
        """

        if radio is not list or radio is not np.ndarray:
            radio = np.repeat(radio, 3)

        g_ = []
        g_2 = []
        d_ = []
        for xyz in [0, 1, 2]:

            if xyz == 2:
                # Make the grid only negative for the z axis

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

        return kernel_g, kernel_d_left, kernel_d_right

    @setdoc_pro(ds.resolution)
    def set_centered_kernel(self, resolution, radio):
        """
        Set a centered

        Args:
            resolution: [s0]
            radio (float): Maximum distance of the kernel

        Returns:

        """
        self.kernel_centers, self.kernel_dxyz_left, self.kernel_dxyz_right = self.create_irregular_grid_kernel(
            resolution, radio)

        return self.kernel_centers

    @setdoc_pro(ds.resolution)
    def set_centered_grid(self, centers, kernel_centers=None, **kwargs):
        """
        Main method of the class, set the XYZ values around centers using a kernel.

        Args:
            centers (np.array): XYZ array with the centers of where we want to create a grid around
            kernel_centers (Optional[np.array]): center of the voxels of a desired kernel.
            **kwargs:
                * resolution: [s0]
                * radio (float): Maximum distance of the kernel
        Returns:

        """

        self.values = np.empty((0, 3))
        centers = np.atleast_2d(centers)

        if kernel_centers is None:
            kernel_centers = self.set_centered_kernel(**kwargs)

        assert centers.shape[1] == 3, 'Centers must be a numpy array that contains the coordinates XYZ'

        for i in centers:
            self.values = np.vstack((self.values, i + kernel_centers))

        self.length = self.values.shape[0]

    def set_tz_kernel(self, **kwargs):
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
    """
    TODO @Elisa
    """
    def __init__(self, regular_grid):
        self.regular_grid = regular_grid
        self.values = np.zeros((0, 3))

        self.topo = None
        # TODO @Elisa: values 3D is a 3D numpy array isnt it
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
        self.topo = Load_DEM_artificial(self.regular_grid, **kwargs)
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
        return mask.swapaxes(0, 1)# np.multiply(np.full(self.regular_grid.values.shape, True).T, mask.ravel()).T

    def _find_indices(self):
        zs = np.linspace(self.regular_grid.extent[4], self.regular_grid.extent[5], self.regular_grid.resolution[2])
        dz = (zs[-1] - zs[0]) / len(zs)
        return ((self.values_3D_res[:, :, 2] - zs[0]) / dz + 1).astype(int)

    def interpolate_zvals_at_xy(self, xy):
        assert xy[:, 0][0] <= xy[:, 0][-1], 'At the moment, the xy values of the first point must be smaller than second' \
                                            '(fix soon)'
        assert xy[:, 1][0] <= xy[:, 1][-1], 'At the moment, the xy values of the first point must be smaller than second' \
                                            '(fix soon)'
        xj = self.values_3D[:, :, 0][0, :]
        yj = self.values_3D[:, :, 1][:, 0]
        zj = self.values_3D[:, :, 2].T
        f = interpolate.RectBivariateSpline(xj, yj, zj)
        zi = f(xy[:, 0], xy[:, 1])
        return np.diag(zi)

    def _line_in_section(self, direction='y', cell_number=1):
        # todo delete after replacing it with the other function

        x = self.values_3D_res[:, :, 0]
        y = self.values_3D_res[:, :, 1]
        z = self.values_3D_res[:, :, 2]

        if direction == 'y':
            a = x[cell_number, :]
            b = y[cell_number, :]
            c = z[cell_number, :]
            assert len(np.unique(b)) == 1
            topoline = np.dstack((a, c)).reshape(-1, 2).astype(int)

        elif direction == 'x':
            a = x[:, cell_number]
            b = y[:, cell_number]
            c = z[:, cell_number]
            assert len(np.unique(a)) == 1
            topoline = np.dstack((b, c)).reshape(-1, 2).astype(int)

        elif direction == "z":
            raise NotImplementedError

        return topoline
