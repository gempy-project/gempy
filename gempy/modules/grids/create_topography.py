"""
This file is part of gempy.

Created on 16.04.2019

@author: Elisa Heim
"""
from typing import Optional

import numpy as np

from gempy.optional_dependencies import require_scipy


def create_random_topography(extent: np.array, resolution: np.array, dz: Optional[np.array] = None,
                             fractal_dimension: Optional[float] = 2.0) -> np.array:
    dem = _LoadDEMArtificial(
        extent=extent,
        resolution=resolution,
        d_z=dz,
        fd=fractal_dimension
    )

    return dem.get_values()


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
