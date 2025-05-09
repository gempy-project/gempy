import warnings
from typing import Optional

import numpy as np

from .grid_types import RegularGrid
from ....modules.grids.create_topography import _LoadDEMArtificial

from ....optional_dependencies import require_skimage


class Topography:
    """
    Object to include topography in the model.

    Notes:
        This always assumes that the topography we pass fits perfectly the extent

    """

    def __init__(self, regular_grid: RegularGrid, values_2d: Optional[np.ndarray] = None):

        self._mask_topo = None
        self._regular_grid = regular_grid

        # Values (n, 3)
        self.values = np.zeros((0, 3))

        # Values (n, n, 3)
        self.values_2d = np.zeros((0, 0, 3))

        # Shape original
        self.raster_shape = tuple()

        # Topography Resolution
        self.resolution = np.zeros((0, 3))

        # Source for the
        self.source = None

        # Coords
        self._x = None
        self._y = None

        if values_2d is not None:
            self.set_values(values_2d)

    @classmethod
    def from_subsurface_structured_data(cls, structured_data: 'subsurface.StructuredData', regular_grid: RegularGrid):
        """Creates a topography object from a subsurface structured data object

        Args:
            structured_data (subsurface.StructuredData): Structured data object

        Returns:
            :class:`gempy.core.grid_modules.topography.Topography`

        """

        # Generate meshgrid for x and y coordinates
        ds = structured_data.data
        x_coordinates = ds['x']
        y_coordinates = ds['y']
        height_values = ds['topography']

        return cls.from_arrays(regular_grid, x_coordinates, y_coordinates, height_values)

    @classmethod
    def from_unstructured_mesh(cls, regular_grid, xyz_vertices):
        """Creates a topography object from an unstructured mesh of XYZ vertices.

        Args:
            regular_grid (RegularGrid): The regular grid object.
            xyz_vertices (numpy.ndarray): Array of XYZ vertices of the unstructured mesh.

        Returns:
            :class:`gempy.core.grid_modules.topography.Topography`
        """
        # Perform Delaunay triangulation on the vertices

        # Generate the regular grid points
        from scipy.interpolate import griddata
        
        x_regular, y_regular = np.meshgrid(
            np.linspace(regular_grid.extent[0], regular_grid.extent[1], regular_grid.resolution[0]),
            np.linspace(regular_grid.extent[2], regular_grid.extent[3], regular_grid.resolution[1]),
            indexing='ij'
        )

        # Interpolate the z-values onto the regular grid
        z_regular = griddata(
            points=xyz_vertices[:, :2],
            values=xyz_vertices[:, 2],
            xi=(x_regular, y_regular),
            method='nearest',
            fill_value=np.nan  # You can choose a different fill value or method
        )

        # Reshape the grid for compatibility with existing structure
        values_2d = np.stack((x_regular, y_regular, z_regular), axis=-1)

        return cls(regular_grid=regular_grid, values_2d=values_2d)
    
    
    @classmethod
    def from_arrays(cls, regular_grid, x_coordinates, y_coordinates, height_values,):
        x_vals, y_vals = np.meshgrid(x_coordinates, y_coordinates, indexing='ij')
        # Reshape arrays for stacking
        x_vals = x_vals[:, :, np.newaxis]  # shape (73, 34, 1)
        y_vals = y_vals[:, :, np.newaxis]  # shape (73, 34, 1)
        topography_vals = height_values.values[:, :, np.newaxis]  # shape (73, 34, 1)
        # Stack along the last dimension
        result = np.concatenate([x_vals, y_vals, topography_vals], axis=2)  # shape (73, 34, 3)
        return cls(regular_grid=regular_grid, values_2d=result)

    @property
    def extent(self):
        return self._regular_grid.extent

    @property
    def regular_grid_resolution(self):
        return self._regular_grid.resolution

    @property
    def x(self):
        if self._x is not None:
            return self._x
        else:
            val = self.values[:, 0].copy()
            val.sort()
            return np.unique(val)

    @property
    def y(self):
        if self._y is not None:
            return self._y
        else:
            val = self.values[:, 1].copy()
            val.sort()
            return np.unique(val)

    def set_values(self, values_2d: np.ndarray):
        """General method to set topography

        Args:
            values_2d (numpy.ndarray[float,float, 3]): array with the XYZ values
             in 2D

        Returns:
            :class:`gempy.core.grid_modules.topography.Topography`


        """
        # Original topography data
        self.values_2d = values_2d
        self.resolution = values_2d.shape[:2]

        # n,3 array
        self.values = values_2d.reshape((-1, 3), order='C')
        return self

    @property
    def topography_mask(self):
        """This method takes a topography grid of the same extent as the regular
         grid and creates a mask of voxels

        """

        # * Check if the topography is the same as the cached one and if so, return the cached mask
        if self._mask_topo is not None:
            return self._mask_topo

        # interpolate topography values to the regular grid
        skimage = require_skimage()
        regular_grid_topo = skimage.transform.resize(
            image=self.values_2d,
            output_shape=(self.regular_grid_resolution[0], self.regular_grid_resolution[1]),
            mode='constant',
            anti_aliasing=False,
            preserve_range=True
        )

        # Adjust the topography to be lower by half a voxel height
        # Assumes your voxel heights are uniform and can be calculated as the total height divided by resolution
        voxel_height = self._regular_grid.dz * 2
        regular_grid_topo = regular_grid_topo - voxel_height

        # Reshape the Z values of the regular grid to 3d
        values_3d = self._regular_grid.values[:, 2].reshape(self.regular_grid_resolution)
        if regular_grid_topo.ndim == 3:
            regular_grid_topo_z = regular_grid_topo[:, :, [2]]
        elif regular_grid_topo.ndim == 2:
            regular_grid_topo_z = regular_grid_topo
        else:
            raise ValueError()
        mask = np.greater(values_3d[:, :, :], regular_grid_topo_z)

        self._mask_topo = mask
        return self._mask_topo

    def resize_topo(self):
        skimage = require_skimage()
        regular_grid_topo = skimage.transform.resize(
            self.values_2d,
            (self.regular_grid_resolution[0], self.regular_grid_resolution[1]),
            mode='constant',
            anti_aliasing=False, preserve_range=True)

        return regular_grid_topo

    def load_random_hills(self, **kwargs):
        warnings.warn('This function is deprecated. Use load_from_random instead', DeprecationWarning)
        if 'extent' in kwargs:
            self.extent = kwargs.pop('extent')

        if 'resolution' in kwargs:
            self.regular_grid_resolution = kwargs.pop('resolution')

        dem = _LoadDEMArtificial(extent=self.extent,
                                 resolution=self.regular_grid_resolution, **kwargs)

        self._x, self._y = dem.x, dem.y
        self.set_values(dem.get_values())

    def save(self, path):
        np.save(path, self.values_2d)

    def load(self, path):
        self.set_values(np.load(path))
        self._x, self._y = None, None
        return self.values

    def load_from_saved(self, *args, **kwargs):
        self.load(*args, **kwargs)
