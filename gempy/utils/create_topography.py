"""
This file is part of gempy.

Created on 16.04.2019

@author: Elisa Heim
"""


import numpy as np
import scipy
import pandas as pn

# you can not import libraries inside a class?!
try:
    import gdal
    GDAL_IMPORT = True
except ImportError:
    GDAL_IMPORT = False


class Load_DEM_GDAL():
    '''Class to include height elevation data (e.g. DEMs) with the geological grid '''

    def __init__(self, path_dem, grid=None):
        '''
        Args:
            path_dem: path where dem is stored. file format: GDAL raster formats
            if grid: cropped to geomodel extent
        '''
        if GDAL_IMPORT == False:
            raise ImportError('Gdal package is not installed. No support for raster formats.')
        self.dem = gdal.Open(path_dem)
        self.dem_zval = self.dem.ReadAsArray()
        self._get_raster_dimensions()
        print(self.extent, self.resolution)

        if grid is not None:
            self.grid = grid
            self.crop2grid()
        else:
            print('pass geo_model to directly crop the DEM to the grid extent')
            print('depending on the size of the raster, this can take forever')
        print(self.extent, self.resolution)
        self.convert2xyz()

    def _get_raster_dimensions(self):
        '''returns dtm.extent and dtm.resolution'''
        ulx, xres, xskew, uly, yskew, yres = self.dem.GetGeoTransform()
        z = self.dem_zval
        if np.any(np.array([xskew, yskew])) != 0:
            print('Obacht! DEM is not north-oriented.')
        lrx = ulx + (self.dem.RasterXSize * xres)
        lry = uly + (self.dem.RasterYSize * yres)
        self.resolution = np.array([(uly - lry) / (-yres), (lrx - ulx) / xres]).astype(int)
        self.extent = np.array([ulx, lrx, lry, uly]).astype(int)
        self.d_z = np.array([z.min(), z.max()])

    def crop2grid(self):
        '''
        evtl in anderer Klasse weil xyz kann gecroppt werden
            output_path:
        Returns:
        '''
        cornerpoints_geo = self._get_cornerpoints(self.grid.extent)
        cornerpoints_dtm = self._get_cornerpoints(self.extent)

        if np.any(cornerpoints_geo[:2] - cornerpoints_dtm[:2]) != 0:
            path_dest = '_cropped_DEM.tif'
            new_bounds = (self.grid.extent[[0, 2, 1, 3]])
            gdal.Warp(path_dest, self.dem, options=gdal.WarpOptions(
                options=['outputBounds'], outputBounds=new_bounds))

            self.dem = gdal.Open(path_dest)
            self.dem_zval = self.dem.ReadAsArray()
            self._get_raster_dimensions()
        print('Cropped raster to geo_model.grid.extent.')

    def convert2xyz(self):
        '''
        Returns: array with the x,y,z coordinates of the topography  [0]: shape(a,b,3), [1]: shape(a*b,3)
        '''
        path_dest = '_topo.xyz'
        print('storing converted file...')
        shape = self.dem_zval.shape
        gdal.Translate(path_dest, self.dem, options=gdal.TranslateOptions(options=['format'], format="XYZ"))

        xyz = pn.read_csv(path_dest, header=None, sep=' ').values
        # print(xyz.shape)
        x = np.flip(xyz[:, 0].reshape(shape), axis=0)
        y = np.flip(xyz[:, 1].reshape(shape), axis=0)
        z = np.flip(xyz[:, 2].reshape(shape), axis=0)

        self.values_3D = np.dstack([x, y, z])

    def resize(self):
        pass

        # return xyz, xyz_box

    def _get_cornerpoints(self, extent):
        upleft = ([extent[0], extent[3]])
        lowleft = ([extent[0], extent[2]])
        upright = ([extent[1], extent[3]])
        lowright = ([extent[1], extent[2]])
        return np.array([upleft, lowleft, upright, lowright])


class Load_DEM_artificial():
    def __init__(self, grid, fd=2.0, extent=None, resolution=None, d_z=None):
        """resolution:np 2D array with extent in X and Y direction"""
        self.grid = grid

        self.resolution = grid.resolution[:2] if resolution is None else resolution
        self.extent = self.grid.extent[:4] if extent is None else extent

        if d_z is None:
            self.d_z = np.array(
                [self.grid.extent[5] - (self.grid.extent[5] - self.grid.extent[4]) * 1 / 5,
                 self.grid.extent[5]])
        else:
            self.d_z = d_z

        topo = self.fractalGrid(fd, N=self.resolution.max())
        topo = np.interp(topo, (topo.min(), topo.max()), (self.d_z))

        self.dem_zval = topo[:self.resolution[0], :self.resolution[1]]  # crop fractal grid with resolution

        self.create_topo_array()

    def fractalGrid(self, fd, N=256):
        '''
        Copied of https://github.com/samthiele/pycompass/blob/master/examples/3_Synthetic%20Examples.ipynb

        Generate isotropic fractal surface image using
        spectral synthesis method [1, p.]
        References:
        1. Yuval Fisher, Michael McGuire,
        The Science of Fractal Images, 1988

        (cf. http://shortrecipes.blogspot.com.au/2008/11/python-isotropic-fractal-surface.html)
        **Arguments**:
         -fd = the fractal dimension
         -N = the size of the fractal surface/image

        '''
        H = 1 - (fd - 2);
        X = np.zeros((N, N), complex)
        A = np.zeros((N, N), complex)
        powerr = -(H + 1.0) / 2.0

        for i in range(int(N / 2) + 1):
            for j in range(int(N / 2) + 1):
                phase = 2 * np.pi * np.random.rand()

                if i is not 0 or j is not 0:
                    rad = (i * i + j * j) ** powerr * np.random.normal()
                else:
                    rad = 0.0

                A[i, j] = complex(rad * np.cos(phase), rad * np.sin(phase))

                if i is 0:
                    i0 = 0
                else:
                    i0 = N - i

                if j is 0:
                    j0 = 0
                else:
                    j0 = N - j

                A[i0, j0] = complex(rad * np.cos(phase), -rad * np.sin(phase))

                A.imag[int(N / 2)][0] = 0.0
                A.imag[0, int(N / 2)] = 0.0
                A.imag[int(N / 2)][int(N / 2)] = 0.0

        for i in range(1, int(N / 2)):
            for j in range(1, int(N / 2)):
                phase = 2 * np.pi * np.random.rand()
                rad = (i * i + j * j) ** powerr * np.random.normal()
                A[i, N - j] = complex(rad * np.cos(phase), rad * np.sin(phase))
                A[N - i, j] = complex(rad * np.cos(phase), -rad * np.sin(phase))

        itemp = scipy.fftpack.ifft2(A)
        itemp = itemp - itemp.min()

        return itemp.real / itemp.real.max()

    def create_topo_array(self):
        '''for masking the lith block'''
        x = np.linspace(self.grid.values[:, 0].min(), self.grid.values[:, 0].max(), self.resolution[1])
        y = np.linspace(self.grid.values[:, 1].min(), self.grid.values[:, 1].max(), self.resolution[0])
        xx, yy = np.meshgrid(x, y, indexing='ij')
        self.values_3D = np.dstack([xx.T, yy.T, self.dem_zval])

