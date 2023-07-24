import warnings
from typing import Optional

import numpy as np

from gempy.core.grid_modules.grid_types import RegularGrid
from gempy.modules.grids.create_topography import _LoadDEMArtificial, LoadDEMGDAL
import skimage


class Topography:
    """
    Object to include topography in the model.

    Notes:
        This always assumes that the topography we pass fits perfectly the extent

    """

    def __init__(self, regular_grid: RegularGrid, values_2d: Optional[np.ndarray] = None):

        self.extent = regular_grid.extent[:]
        self.regular_grid_resolution = regular_grid.resolution[:]

        # Values (n, 3)
        self.values = np.zeros((0, 3))

        # Values (n, n, 3)
        self.values_2d = np.zeros((0, 3))

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

    @property
    def x(self):
        if self._x is not None:
            return self._x
        else:
            val = self.values[:, 0]
            val.sort()
            return np.unique(val)

    @property
    def y(self):
        if self._y is not None:
            return self._y
        else:
            val = self.values[:, 1]
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

    def crop_topography(self, extent):
        """Crop the topography to a given extent.

        This may be useful for example to mask the regular grid.

        Args:
            extent:

        Returns:

        """
        raise NotImplementedError

    def resize_topo(self):
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

    def load_from_gdal(self, filepath):
        dem = LoadDEMGDAL(filepath, extent=self.extent)
        self._x, self._y = None, None
        self.set_values(dem.get_values())
        # self.source = 'gdal'

    def save(self, path):
        np.save(path, self.values_2d)

    def load(self, path):
        self.set_values(np.load(path))
        self._x, self._y = None, None
        return self.values

    def load_from_saved(self, *args, **kwargs):
        self.load(*args, **kwargs)
