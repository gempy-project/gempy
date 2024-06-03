import dataclasses

from typing import Optional, Sequence

import numpy as np

from ..core_utils import calculate_line_coordinates_2points
from ....optional_dependencies import require_pandas
from gempy_engine.core.data.transforms import Transform, TransformOpsOrder


@dataclasses.dataclass
class RegularGrid:
    """
    Class with the methods and properties to manage 3D regular grids where the model will be interpolated.

    """
    resolution: np.ndarray
    extent: np.ndarray #: this is the orthogonal extent. If the grid is rotated, the extent will be different
    values: np.ndarray
    mask_topo: np.ndarray
    x: Optional[np.ndarray]
    y: Optional[np.ndarray]
    z: Optional[np.ndarray]
    _transform: Transform  #: If a transform exists, it will be applied to the grid

    def __init__(self, extent: np.ndarray, resolution: np.ndarray, transform: Optional[Transform]=None):
        self.resolution = np.ones((0, 3), dtype='int64')
        self.extent = np.zeros(6, dtype='float64')
        self.values = np.zeros((0, 3))
        self.mask_topo = np.zeros((0, 3), dtype=bool)

        self.set_regular_grid(extent, resolution, transform)

    def _create_regular_grid_3d(self):
        coords = self.x_coord, self.y_coord, self.z_coord

        g = np.meshgrid(*coords, indexing="ij")
        values = np.vstack(tuple(map(np.ravel, g))).T.astype("float64")

        # Transform the values
        self.values = self.transform.apply_with_pivot(
            points=values, 
            pivot=np.array([self.extent[0], self.extent[2], self.extent[4]])
        )


    def set_regular_grid(self, extent: Sequence[float], resolution: Sequence[int], transform: Optional[Transform] = None):
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
        self.transform = transform
        self._create_regular_grid_3d()
        
    @property
    def transform(self) -> Transform:
        return self._transform
    
    @transform.setter
    def transform(self, value: Transform):
        if value is None:
            self._transform = Transform.init_neutral()
        self._transform = value

    @classmethod
    def from_corners_box(cls, pivot: tuple, point2: tuple, point3: tuple, zmin, zmax, resolution: np.ndarray):
        # Define the coordinates of the three points
        x1, y1 = pivot
        x2, y2 = point2
        x3, y3 = point3

        # Calculate the vectors along the sides of the rectangle
        v1 = np.array([x2 - x1, y2 - y1, 0])
        v2 = np.array([x3 - x1, y3 - y1, 0])

        # Check if the points are collinear
        cross = np.cross(v1[:2], v2[:2])
        if np.isclose(cross, 0):
            raise ValueError("The points are collinear")

        # Check if v1 and v2 are perpendicular
        if not np.isclose(np.dot(v1, v2), 0):
            # Compute the closest valid p3
            v1_norm = v1 / np.linalg.norm(v1)
            v2_projected = v2 - np.dot(v2, v1_norm) * v1_norm
            p3_corrected = np.array([x1, y1, 0]) + v2_projected
            x3, y3 = p3_corrected[:2]
            raise ValueError(f"The provided points are not perpendicular. Suggested corrected point3: ({x3}, {y3})")
        
        # Calculate the orthonormal basis
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        v3_norm = np.cross(v1_norm, v2_norm)

        # Create the rotation matrix
        rotation_matrix = np.array([
                [v1_norm[0], v2_norm[0], v3_norm[0], 0],
                [v1_norm[1], v2_norm[1], v3_norm[1], 0],
                [v1_norm[2], v2_norm[2], v3_norm[2], 0],
                [0, 0, 0, 1]
        ])

        # rotation_matrix_degrees = np.degrees(rotation_matrix)
        transform = Transform.from_matrix(rotation_matrix)

        # Calculate the extents in the new coordinate system
        extent_x = np.linalg.norm(v1)
        extent_y = np.linalg.norm(v2)

        # We transform the origin point1 to the new coordinates
        vector = np.array([[x1, y1, zmin]])
        origin_transformed = transform.apply_with_pivot(vector, pivot=vector[0])[0]

        xmin, ymin, zmin = origin_transformed
        xmax, ymax, zmax = xmin + extent_x, ymin + extent_y, zmin + zmax - zmin

        extent = np.array([xmin, xmax, ymin, ymax, zmin, zmax], dtype='float64')

        grid = cls(extent=extent, resolution=resolution, transform=transform)
        return grid
    

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

    @property
    def x_coord(self):
        return np.linspace(self.extent[0] + self.dx / 2, self.extent[1] - self.dx / 2, self.resolution[0], dtype="float64")

    @property
    def y_coord(self):
        return np.linspace(self.extent[2] + self.dy / 2, self.extent[3] - self.dy / 2, self.resolution[1], dtype="float64")

    @property
    def z_coord(self):
        return np.linspace(self.extent[4] + self.dz / 2, self.extent[5] - self.dz / 2, self.resolution[2], dtype="float64")

    @property
    def dx_dy_dz(self):
        dx = (self.extent[1] - self.extent[0]) / self.resolution[0]
        dy = (self.extent[3] - self.extent[2]) / self.resolution[1]
        dz = (self.extent[5] - self.extent[4]) / self.resolution[2]
        return dx, dy, dz

    @property
    def dx(self):
        return (self.extent[1] - self.extent[0]) / self.resolution[0]

    @property
    def dy(self):
        return (self.extent[3] - self.extent[2]) / self.resolution[1]

    @property
    def dz(self):
        return (self.extent[5] - self.extent[4]) / self.resolution[2]

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
