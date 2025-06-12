import dataclasses
from pydantic import model_validator, Field

from typing import Optional, Sequence, Annotated

import numpy as np

from ..encoders.converters import numpy_array_short_validator
from .... import optional_dependencies
from gempy_engine.core.data.transforms import Transform


@dataclasses.dataclass
class RegularGrid:
    """
    Class with the methods and properties to manage 3D regular grids where the model will be interpolated.

    """
    resolution: Annotated[np.ndarray, numpy_array_short_validator] = dataclasses.field(default_factory=lambda: np.ones((0, 3), dtype='int64'))
    extent: Annotated[np.ndarray, numpy_array_short_validator] = dataclasses.field(default_factory=lambda: np.zeros(6, dtype='float64'))  #: this is the ORTHOGONAL extent. If the grid is rotated, the extent will be different
    values: Annotated[np.ndarray, Field(exclude=True)] = dataclasses.field(default_factory=lambda: np.zeros((0, 3)))
    mask_topo: Annotated[np.ndarray, Field(exclude=True)] = dataclasses.field(default_factory=lambda: np.zeros((0, 3), dtype=bool))
    _transform: Transform | None = None  #: If a transform exists, it will be applied to the grid

    def __init__(self, extent: np.ndarray, resolution: np.ndarray, transform: Optional[Transform] = None):
        self.resolution = np.ones((0, 3), dtype='int64')
        self.extent = np.zeros(6, dtype='float64')
        self.values = np.zeros((0, 3))
        self.mask_topo = np.zeros((0, 3), dtype=bool)

        self.set_regular_grid(extent, resolution, transform)


    @model_validator(mode="after")
    def _validate_regular_grid(self):
        self._create_regular_grid_3d()
        return self


    def _create_regular_grid_3d(self):
        coords = self.x_coord, self.y_coord, self.z_coord

        g = np.meshgrid(*coords, indexing="ij")
        values = np.vstack(tuple(map(np.ravel, g))).T.astype("float64")

        # Transform the values
        if self.transform is not None:
            self.values = self.transform.apply_inverse_with_pivot(
                points=values,
                pivot=np.array([self.extent[0], self.extent[2], self.extent[4]])
            )
        else:
            self.values = values

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
        if self._transform is None:
            return Transform.init_neutral()
        return self._transform

    @transform.setter
    def transform(self, value: Transform):
        self._transform = value

    @classmethod
    def from_corners_box(cls, pivot: tuple, point_x_axis: tuple, distance_point3: float,
                         zmin: float, zmax: float, resolution: np.ndarray, plot: bool = True):
        """Always rotate around the z axis towards the positive x axis. 
         The distance_point3 is the distance from the pivot to the point3."""

        def _calculate_rotated_box_val(v1, v2):
            # Check if the points are collinear
            # noinspection PyUnreachableCode
            cross = np.cross(v1[:2], v2[:2])
            if np.isclose(cross, 0):
                raise ValueError("The points are collinear")
            # Check if v1 and v2 are perpendicular
            if not np.isclose(np.dot(v1, v2), 0, atol=1e-4):
                # Compute the closest valid p3
                v1_norm = v1 / np.linalg.norm(v1)
                v2_projected = v2 - np.dot(v2, v1_norm) * v1_norm
                p3_corrected = np.array([x1, y1, 0]) + v2_projected
                x3, y3 = p3_corrected[:2]
                raise ValueError(f"The provided points are not perpendicular. Suggested corrected point3: ({x3}, {y3})")
            # Calculate the orthonormal basis
            v1_norm = v1 / np.linalg.norm(v1)
            v2_norm = v2 / np.linalg.norm(v2)
            # noinspection PyUnreachableCode
            # v3_norm = np.cross(v1_norm, v2_norm)
            v3_norm = np.array([0, 0, 1])  # ! We assume rotation is always around z axis
            return v1_norm, v2_norm, v3_norm

        # Define the coordinates of the three points
        x1, y1 = pivot
        x2, y2 = point_x_axis

        dx, dy = point_x_axis[0] - pivot[0], point_x_axis[1] - pivot[1]
        x3, y3 = (pivot[0] - dy * distance_point3 / np.hypot(dx, dy), pivot[1] + dx * distance_point3 / np.hypot(dx, dy))

        # Calculate the vectors along the sides of the rectangle
        v1 = np.array([x2 - x1, y2 - y1, 0])
        v2 = np.array([x3 - x1, y3 - y1, 0])

        v1_norm, v2_norm, v3_norm = _calculate_rotated_box_val(v1, v2)

        # Create the rotation matrix
        rotation_matrix = np.array([
                [v1_norm[0], v2_norm[0], v3_norm[0], 0],
                [v1_norm[1], v2_norm[1], v3_norm[1], 0],
                [v1_norm[2], v2_norm[2], v3_norm[2], 0],
                [0, 0, 0, 1]
        ])

        inverted_rotation_matrix = np.linalg.inv(rotation_matrix)
        # Calculate the extents in the new coordinate system
        extent_x = np.linalg.norm(v1)
        extent_y = np.linalg.norm(v2)

        # We transform the origin point1 to the new coordinates
        origin_ = [x1, y1, zmin]

        xmin, ymin, zmin = origin_
        xmax, ymax, zmax = xmin + extent_x, ymin + extent_y, zmin + zmax - zmin

        extent = np.array([xmin, xmax, ymin, ymax, zmin, zmax], dtype='float64')

        transform = Transform.from_matrix(inverted_rotation_matrix)
        transform.cached_pivot = origin_
        grid = cls(extent=extent, resolution=resolution, transform=transform)

        if plot:
            cls.plot_rotation(
                regular_grid=grid,
                pivot=pivot,
                point_x_axis=point_x_axis,
                point_y_axis=(x3, y3)
            )
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
    def values_vtk_format(self) -> np.ndarray:
        return self.get_values_vtk_format()

    def get_values_vtk_format(self, orthogonal: bool = False) -> np.ndarray:
        extent = self.extent
        resolution = self.resolution + 1

        x = np.linspace(extent[0], extent[1], resolution[0], dtype="float64")
        y = np.linspace(extent[2], extent[3], resolution[1], dtype="float64")
        z = np.linspace(extent[4], extent[5], resolution[2], dtype="float64")
        xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")
        g = np.vstack((xv.ravel(), yv.ravel(), zv.ravel())).T

        # Transform the values
        if self.transform is not None and orthogonal is False:
            g = self.transform.apply_inverse_with_pivot(
                points=g,
                pivot=np.array([self.extent[0], self.extent[2], self.extent[4]])
            )

        return g

    @staticmethod
    def plot_rotation(regular_grid, pivot, point_x_axis, point_y_axis):
        plt = optional_dependencies.require_matplotlib()
        x1, y1 = pivot
        x2, y2 = point_x_axis
        x3, y3 = point_y_axis

        # Plot the original corners
        plt.scatter([x1, x2, x3], [y1, y2, y3], c='r')
        # Plot lines with distance labels
        plt.plot([x1, x2], [y1, y2], 'r')
        plt.text((x1 + x2) / 2, (y1 + y2) / 2, f'{np.linalg.norm([x1 - x2, y1 - y2]):.2f}', color='r')
        plt.plot([x1, x3], [y1, y3], 'r')
        plt.text((x1 + x3) / 2, (y1 + y3) / 2, f'{np.linalg.norm([x1 - x3, y1 - y3]):.2f}', color='r')
        # Plot the transformed corners
        transformed_extent = regular_grid.extent
        bounding_box = regular_grid.bounding_box
        plt.scatter(bounding_box[:, 0], bounding_box[:, 1], c='b')
        # Plot lines with distance labels
        plt.plot([bounding_box[2, 0], bounding_box[0, 0]], [bounding_box[2, 1], bounding_box[0, 1]], 'b')
        plt.text((bounding_box[2, 0] + bounding_box[0, 0]) / 2, (bounding_box[2, 1] + bounding_box[0, 1]) / 2, f'{np.linalg.norm(bounding_box[2] - bounding_box[0]):.2f}', color='b')
        plt.plot([bounding_box[0, 0], bounding_box[4, 0]], [bounding_box[0, 1], bounding_box[4, 1]], 'b')
        plt.text((bounding_box[0, 0] + bounding_box[4, 0]) / 2, (bounding_box[0, 1] + bounding_box[4, 1]) / 2, f'{np.linalg.norm(bounding_box[0] - bounding_box[4]):.2f}', color='b')
        # Plot the values of the grid
        values = regular_grid.values
        plt.scatter(values[:, 0], values[:, 1], c='g')
        plt.show()


