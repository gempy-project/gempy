import dataclasses

import warnings
from pydantic import Field
from typing import Optional, Tuple

import numpy as np

from .regular_grid import RegularGrid

from ....optional_dependencies import require_skimage, require_scipy
from dataclasses import field, dataclass
from ..encoders.converters import short_array_type


@dataclass
class Topography:

    """
      Object to include topography in the model.
      Notes:
          This always assumes that the topography we pass fits perfectly the extent.
      """

    _regular_grid: RegularGrid
    values_2d: np.ndarray = Field(exclude=True, default_factory=lambda: np.zeros((0, 0, 3)))
    source: Optional[str] = None

    # Fields managed internally
    values: short_array_type = field(init=False, default_factory=lambda: np.zeros((0, 3)))
    resolution: Tuple[int, int] = Field(init=True, default=(0, 0))
    raster_shape: Tuple[int, ...] = field(init=False, default_factory=tuple)
    _mask_topo: Optional[np.ndarray] = field(init=False, default=None, repr=False)
    _x: Optional[np.ndarray] = field(init=False, default=None, repr=False)
    _y: Optional[np.ndarray] = field(init=False, default=None, repr=False)

    def __post_init__(self):
        # if a non-empty array was provided, initialize the flattened values
        if self.values_2d.size:
            self.set_values(self.values_2d)


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

        return cls(_regular_grid=regular_grid, values_2d=values_2d)
    
    
    @classmethod
    def from_arrays(cls, regular_grid, x_coordinates, y_coordinates, height_values,):
        x_vals, y_vals = np.meshgrid(x_coordinates, y_coordinates, indexing='ij')
        # Reshape arrays for stacking
        x_vals = x_vals[:, :, np.newaxis]  # shape (73, 34, 1)
        y_vals = y_vals[:, :, np.newaxis]  # shape (73, 34, 1)
        topography_vals = height_values.values[:, :, np.newaxis]  # shape (73, 34, 1)
        # Stack along the last dimension
        result = np.concatenate([x_vals, y_vals, topography_vals], axis=2)  # shape (73, 34, 3)
        return cls(_regular_grid=regular_grid, values_2d=result)

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

    def set_values2d(self, values: np.ndarray) -> "Topography":
        """
        Reconstruct the 2D topography (shape = resolution + [3]) from
        a flat Nx3 array (or from self.values if none is provided).
        """
        # default to the already-flattened XYZ array
        
        # compute expected size
        nx, ny = self.resolution
        expected = nx * ny * 3
        if values.size != expected:
            raise ValueError(
                f"Cannot reshape array of size {values.size} into shape {(nx, ny, 3)}."
            )

        # reshape in C-order to (nx, ny, 3)
        self.set_values(
            values_2d=values.reshape(nx, ny, 3, order="C")
        )

        # invalidate any cached mask
        self._mask_topo = None
        return self

    def set_values2d_(self, values: np.ndarray):
        resolution = (60, 60)
        self.values_2d = values.reshape(*resolution, 3)

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


class _LoadDEMArtificial:  # * Cannot think of a good reason to be a class

    def __init__(self, grid=None, fd=2.0, extent=None, resolution=None, d_z=None):
        """Class to create a random topography based on a fractal grid algorithm.

        Args:
            fd:         fractal dimension, defaults to 2.0
            d_z:        maximum height difference. If none, last 20% of the model in z direction
            extent:     extent in xy direction. If none, geo_model.grid.extent
            resolution: desired resolution of the topography array. If none, geo_model.grid.resolution
        """
        self.values_2d = np.array([])
        self.resolution = grid.resolution[:2] if resolution is None else resolution

        assert all(np.asarray(self.resolution) >= 2), 'The regular grid needs to be at least of size 2 on all directions.'
        self.extent = grid.extent if extent is None else extent

        if d_z is None:
            self.d_z = np.array([self.extent[5] - (self.extent[5] - self.extent[4]) * 1 / 5, self.extent[5]])
            print(self.d_z)
        else:
            self.d_z = d_z

        topo = self.fractalGrid(fd, n=self.resolution.max())
        topo = np.interp(topo, (topo.min(), topo.max()), self.d_z)

        self.dem_zval = topo[:self.resolution[0], :self.resolution[1]]  # crop fractal grid with resolution
        self.create_topo_array()

    @staticmethod
    def fractalGrid(fd, n=256):
        """
        Modified after https://github.com/samthiele/pycompass/blob/master/examples/3_Synthetic%20Examples.ipynb

        Generate isotropic fractal surface image using
        spectral synthesis method [1, p.]
        References:
        1. Yuval Fisher, Michael McGuire,
        The Science of Fractal Images, 1988

        (cf. http://shortrecipes.blogspot.com.au/2008/11/python-isotropic-fractal-surface.html)
        **Arguments**:
         -fd = the fractal dimension
         -N = the size of the fractal surface/image

        """
        h = 1 - (fd - 2)
        # X = np.zeros((N, N), complex)
        a = np.zeros((n, n), complex)
        powerr = -(h + 1.0) / 2.0

        for i in range(int(n / 2) + 1):
            for j in range(int(n / 2) + 1):
                phase = 2 * np.pi * np.random.rand()

                if i != 0 or j != 0:
                    rad = (i * i + j * j) ** powerr * np.random.normal()
                else:
                    rad = 0.0

                a[i, j] = complex(rad * np.cos(phase), rad * np.sin(phase))

                if i == 0:
                    i0 = 0
                else:
                    i0 = n - i

                if j == 0:
                    j0 = 0
                else:
                    j0 = n - j

                a[i0, j0] = complex(rad * np.cos(phase), -rad * np.sin(phase))

                a.imag[int(n / 2)][0] = 0.0
                a.imag[0, int(n / 2)] = 0.0
                a.imag[int(n / 2)][int(n / 2)] = 0.0

        for i in range(1, int(n / 2)):
            for j in range(1, int(n / 2)):
                phase = 2 * np.pi * np.random.rand()
                rad = (i * i + j * j) ** powerr * np.random.normal()
                a[i, n - j] = complex(rad * np.cos(phase), rad * np.sin(phase))
                a[n - i, j] = complex(rad * np.cos(phase), -rad * np.sin(phase))

        scipy = require_scipy()
        itemp = scipy.fftpack.ifft2(a)
        itemp = itemp - itemp.min()

        return itemp.real / itemp.real.max()

    def create_topo_array(self):
        """for masking the lith block"""
        x = np.linspace(self.extent[0], self.extent[1], self.resolution[0])
        y = np.linspace(self.extent[2], self.extent[3], self.resolution[1])
        self.x = x
        self.y = y
        xx, yy = np.meshgrid(x, y, indexing='ij')
        self.values_2d = np.dstack([xx, yy, self.dem_zval])

    def get_values(self):
        return self.values_2d