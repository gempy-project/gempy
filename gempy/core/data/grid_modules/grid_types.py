from typing import Optional

import numpy as np

from gempy.core.data.core_utils import calculate_line_coordinates_2points
from gempy.optional_dependencies import require_pandas


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
        mask_topo (np.ndarray, dtype=bool): same shape as values. Values above the topography are False
        dx (float): size of the cells on x
        dy (float): size of the cells on y
        dz (float): size of the cells on z

    """
    resolution: np.ndarray
    extent: np.ndarray
    extent_r: np.ndarray
    values: np.ndarray
    values_r: np.ndarray
    mask_topo: np.ndarray
    x: Optional[np.ndarray]
    y: Optional[np.ndarray]
    z: Optional[np.ndarray]
    dx: float
    dy: float
    dz: float
    
    _cached_topography: "Topography" = None

    def __init__(self, extent=None, resolution=None, **kwargs):
        # @ formatter:off
        self.resolution = np.ones((0, 3), dtype='int64')
        self.extent = np.zeros(6, dtype='float64')
        self.extent_r = np.zeros(6, dtype='float64')
        self.values = np.zeros((0, 3))
        self.values_r = np.zeros((0, 3))
        self.mask_topo = np.zeros((0, 3), dtype=bool)
        self.x = None
        self.y = None
        self.z = None

        if extent is not None and len(resolution) > 0:
            self.set_regular_grid(extent, resolution)
            self.dx, self.dy, self.dz = self.get_dx_dy_dz()

        # @ formatter:on

    @property
    def bounding_box(self) -> np.ndarray:
        extents = self.extent
        # Define 3D points of the bounding box corners based on extents
        bounding_box_points = np.array([[extents[0], extents[2], extents[4]],  # min x, min y, min z
                                        [extents[0], extents[2], extents[5]],  # min x, min y, max z
                                        [extents[0], extents[3], extents[4]],  # min x, max y, min z
                                        [extents[0], extents[3], extents[5]],  # min x, max y, max z
                                        [extents[1], extents[2], extents[4]],  # max x, min y, min z
                                        [extents[1], extents[2], extents[5]],  # max x, min y, max z
                                        [extents[1], extents[3], extents[4]],  # max x, max y, min z
                                        [extents[1], extents[3], extents[5]]])  # max x, max y, max z
        return bounding_box_points

    def set_coord(self, extent, resolution):
        dx = (extent[1] - extent[0]) / resolution[0]
        dy = (extent[3] - extent[2]) / resolution[1]
        dz = (extent[5] - extent[4]) / resolution[2]

        self.x = np.linspace(extent[0] + dx / 2, extent[1] - dx / 2, resolution[0], dtype="float64")
        self.y = np.linspace(extent[2] + dy / 2, extent[3] - dy / 2, resolution[1], dtype="float64")
        self.z = np.linspace(extent[4] + dz / 2, extent[5] - dz / 2, resolution[2], dtype="float64")

        return self.x, self.y, self.z

    def create_regular_grid_3d(self, extent, resolution):
        """
        Method to create a 3D regular grid where is interpolated

        Args:
            extent (list):  [x_min, x_max, y_min, y_max, z_min, z_max]
            resolution (list): [nx, ny, nz].

        Returns:
            numpy.ndarray: Unraveled 3D numpy array where every row correspond to the xyz coordinates of a regular grid

        """

        coords = self.set_coord(extent, resolution)
        g = np.meshgrid(*coords, indexing="ij")
        values = np.vstack(tuple(map(np.ravel, g))).T.astype("float64")
        return values

    def get_dx_dy_dz(self, rescale=False):
        if rescale is True:
            dx = (self.extent_r[1] - self.extent_r[0]) / self.resolution[0]
            dy = (self.extent_r[3] - self.extent_r[2]) / self.resolution[1]
            dz = (self.extent_r[5] - self.extent_r[4]) / self.resolution[2]
        else:
            dx = (self.extent[1] - self.extent[0]) / self.resolution[0]
            dy = (self.extent[3] - self.extent[2]) / self.resolution[1]
            dz = (self.extent[5] - self.extent[4]) / self.resolution[2]
        return dx, dy, dz

    def set_regular_grid(self, extent, resolution):
        """
        Set a regular grid into the values parameters for further computations
        Args:
             extent (list, np.ndarry):  [x_min, x_max, y_min, y_max, z_min, z_max]
            resolution (list, np.ndarray): [nx, ny, nz]
        """
        # * Check extent and resolution are not the same
        extent_equal = np.array_equal(extent, self.extent)
        resolution_equal = np.array_equal(resolution, self.resolution)

        if extent_equal and resolution_equal:
            return self.values

        self.extent = np.asarray(extent, dtype='float64')
        self.resolution = np.asarray(resolution)
        self.values = self.create_regular_grid_3d(extent, resolution)
        self.length = self.values.shape[0]
        self.dx, self.dy, self.dz = self.get_dx_dy_dz()
        return self.values

    @property
    def values_vtk_format(self):
        extent = self.extent
        resolution = self.resolution + 1

        x = np.linspace(extent[0], extent[1], resolution[0], dtype="float64")
        y = np.linspace(extent[2], extent[3], resolution[1], dtype="float64")
        z = np.linspace(extent[4], extent[5], resolution[2], dtype="float64")
        xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")
        g = np.vstack((xv.ravel(), yv.ravel(), zv.ravel())).T

        return g


class Sections:
    """
    Object that creates a grid of cross sections between two points.

    Args:
        regular_grid: Model.grid.regular_grid
        section_dict: {'section name': ([p1_x, p1_y], [p2_x, p2_y], [xyres, zres])}
    """

    def __init__(self, regular_grid=None, z_ext=None, section_dict=None):
        pd = require_pandas()
        if regular_grid is not None:
            self.z_ext = regular_grid.extent[4:]
        else:
            self.z_ext = z_ext

        self.section_dict = section_dict
        self.names = []
        self.points = []
        self.resolution = []
        self.length = [0]
        self.dist = []
        self.df = pd.DataFrame()
        self.df['dist'] = self.dist
        self.values = np.empty((0, 3))
        self.extent = None

        if section_dict is not None:
            self.set_sections(section_dict)

    def _repr_html_(self):
        return self.df.to_html()

    def __repr__(self):
        return self.df.to_string()

    def show(self):
        pass

    def set_sections(self, section_dict, regular_grid=None, z_ext=None):
        pd = require_pandas()
        self.section_dict = section_dict
        if regular_grid is not None:
            self.z_ext = regular_grid.extent[4:]

        self.names = np.array(list(self.section_dict.keys()))

        self.get_section_params()
        self.calculate_all_distances()
        self.df = pd.DataFrame.from_dict(self.section_dict, orient='index', columns=['start', 'stop', 'resolution'])
        self.df['dist'] = self.dist

        self.compute_section_coordinates()

    def get_section_params(self):
        self.points = []
        self.resolution = []
        self.length = [0]

        for i, section in enumerate(self.names):
            points = [self.section_dict[section][0], self.section_dict[section][1]]
            assert points[0] != points[
                1], 'The start and end points of the section must not be identical.'

            self.points.append(points)
            self.resolution.append(self.section_dict[section][2])
            self.length = np.append(self.length, self.section_dict[section][2][0] *
                                    self.section_dict[section][2][1])
        self.length = np.array(self.length).cumsum()

    def calculate_all_distances(self):
        self.coordinates = np.array(self.points).ravel().reshape(-1,
                                                                 4)  # axis are x1,y1,x2,y2
        self.dist = np.sqrt(np.diff(self.coordinates[:, [0, 2]]) ** 2 + np.diff(
            self.coordinates[:, [1, 3]]) ** 2)

    def compute_section_coordinates(self):
        for i in range(len(self.names)):
            xy = calculate_line_coordinates_2points(self.coordinates[i, :2],
                                                         self.coordinates[i, 2:],
                                                         self.resolution[i][0])
            zaxis = np.linspace(self.z_ext[0], self.z_ext[1], self.resolution[i][1],
                                dtype="float64")
            X, Z = np.meshgrid(xy[:, 0], zaxis, indexing='ij')
            Y, _ = np.meshgrid(xy[:, 1], zaxis, indexing='ij')
            xyz = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
            if i == 0:
                self.values = xyz
            else:
                self.values = np.vstack((self.values, xyz))

    def generate_axis_coord(self):
        for i, name in enumerate(self.names):
            xy = calculate_line_coordinates_2points(
                self.coordinates[i, :2],
                self.coordinates[i, 2:],
                self.resolution[i][0]
            )
            yield name, xy

    
    def get_section_args(self, section_name: str):
        where = np.where(self.names == section_name)[0][0]
        return self.length[where], self.length[where + 1]

    def get_section_grid(self, section_name: str):
        l0, l1 = self.get_section_args(section_name)
        return self.values[l0:l1]

    

class CustomGrid:
    """Object that contains arbitrary XYZ coordinates.

    Args:
        xyx_coords (numpy.ndarray like): XYZ (in columns) of the desired coordinates

    Attributes:
        values (np.ndarray): XYZ coordinates
    """

    def __init__(self, xyx_coords: np.ndarray):
        self.values = np.zeros((0, 3))
        self.set_custom_grid(xyx_coords)

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
        assert type(custom_grid) is np.ndarray and custom_grid.shape[1] == 3, \
            'The shape of new grid must be (n,3)  where n is the number of' \
            ' points of the grid'

        self.values = custom_grid
        self.length = self.values.shape[0]
        return self.values

