"""
This file is part of gempy.

Created on 16.04.2019

@author: Elisa Heim
"""


import numpy as np
from scipy import fftpack
import pandas as pn
try:
    import gdal
    GDAL_IMPORT = True
except ImportError:
    GDAL_IMPORT = False
import matplotlib.pyplot as plt


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

        if isinstance(self.dem, type(None)):
            raise AttributeError('Raster file could not be opened. Check if the filepath is correct. If yes,'
                                 'check if your file fits the requirements of GDALs raster file formats.')

        try:
            self.dem_zval = self.dem.ReadAsArray()
        except AttributeError:
            print('Filepath seems to be wrong.')
            raise

        self._get_raster_dimensions()

        if grid is not None:
            self.grid = grid
            self.crop2grid()
        else:
            print('pass geo_model to automatically crop the DEM to the grid extent')
        print('depending on the size of the raster, this can take a while...')
        self.convert2xyz()

    def _get_raster_dimensions(self):
        '''calculates DEM extent, resolution, and max. z extent (d_z)'''
        ulx, xres, xskew, uly, yskew, yres = self.dem.GetGeoTransform()
        z = self.dem_zval
        if np.any(np.array([xskew, yskew])) != 0:
            print('DEM is not north-oriented.')
        lrx = ulx + (self.dem.RasterXSize * xres)
        lry = uly + (self.dem.RasterYSize * yres)
        self.resolution = np.array([(uly - lry) / (-yres), (lrx - ulx) / xres]).astype(int)
        self.extent = np.array([ulx, lrx, lry, uly]).astype(int)
        self.d_z = np.array([z.min(), z.max()])

    def info(self):
        ulx, xres, xskew, uly, yskew, yres = self.dem.GetGeoTransform()
        print('raster extent:  {}\n raster resolution: {}\n Pixel X size {}, Pixel Y size {}'.format(
            self.extent, self.resolution, xres, yres))
        plt.imshow(self.dem_zval, extent=self.extent)  # plot raster as image
        plt.colorbar()

    def crop2grid(self):
        '''
        Crops raster to extent of the geomodel grid.
        '''
        cornerpoints_geo = self._get_cornerpoints(self.grid.extent)
        cornerpoints_dtm = self._get_cornerpoints(self.extent)

        #self.check()

        if np.any(cornerpoints_geo[:2] - cornerpoints_dtm[:2]) != 0:
            path_dest = '_cropped_DEM.tif'
            new_bounds = (self.grid.extent[[0, 2, 1, 3]])
            gdal.Warp(path_dest, self.dem, options=gdal.WarpOptions(
                options=['outputBounds'], outputBounds=new_bounds))

            self.dem = gdal.Open(path_dest)
            self.dem_zval = self.dem.ReadAsArray()
            self._get_raster_dimensions()
        print('Cropped raster to geo_model.grid.extent.')

    def check(self, test=False):
        #todo make this usable
        test = np.logical_and.reduce((self.grid.extent[0] <= self.extent[0],
                                      self.grid.extent[1] >= self.extent[1],
                                      self.grid.extent[2] <= self.extent[2],
                                      self.grid.extent[3] >= self.extent[3]))
        if test:
            cornerpoints_geo = self._get_cornerpoints(self.grid.extent)
            cornerpoints_dtm = self._get_cornerpoints(self.extent)
            plt.scatter(cornerpoints_geo[:, 0], cornerpoints_geo[:, 1], label='grid extent')
            plt.scatter(cornerpoints_dtm[:, 0], cornerpoints_dtm[:, 1], label='raster extent')
            plt.legend(frameon=True, loc='upper left')
            raise AssertionError('The model extent is too different from the raster extent.')

    def convert2xyz(self):
        '''
        Translates the gdal raster object to a numpy array of xyz coordinates.
        '''
        path_dest = 'topo.xyz'
        print('storing converted file...')
        shape = self.dem_zval.shape
        if len(shape) == 3:
            shape = shape[1:]
        gdal.Translate(path_dest, self.dem, options=gdal.TranslateOptions(options=['format'], format="XYZ"))

        xyz = pn.read_csv(path_dest, header=None, sep=' ').values
        x = np.flipud(xyz[:, 0].reshape(shape))
        y = np.flipud(xyz[:, 1].reshape(shape))
        z = np.flipud(xyz[:, 2].reshape(shape))

        self.values_3D = np.dstack([x, y, z])

    def _resize(self, resx, resy):

        #self.values_3D_res = skimage.transform.resize(self.values_3D,
        #                                              (resx, resy),
         #                                             mode='constant',
         #                                             anti_aliasing=False, preserve_range=True)
        #self.resolution_res = np.array([resx, resy])
        pass

    def resample(self, new_xres, new_yres, save_path):
        """
        Decrease the pixel size of the raster.
        Args:
            new_xres (int): desired resolution in x-direction
            new_yres (int): desired resolution in y-direction
            save_path (str): filepath to where the output file should be stored

        Returns: Nothing, it writes a raster file with decreased resolution.

        """
        props = self.dem.GetGeoTransform()
        print('current pixel xsize:', props[1], 'current pixel ysize:', -props[-1])
        options = gdal.WarpOptions(options=['tr'], xRes=new_xres, yRes=new_yres)
        newfile = gdal.Warp(save_path, self.dem, options=options)
        newprops = newfile.GetGeoTransform()
        print('new pixel xsize:', newprops[1], 'new pixel ysize:', -newprops[-1])
        print('file saved in ' + save_path)

    def _get_cornerpoints(self, extent):
        """
        Get the coordinates of the bounding box.
        Args:
            extent: np.array([xmin, xmax, ymin, ymax)]

        Returns: np.ndarray with corner coordinates

        """
        upleft = ([extent[0], extent[3]])
        lowleft = ([extent[0], extent[2]])
        upright = ([extent[1], extent[3]])
        lowright = ([extent[1], extent[2]])
        return np.array([upleft, lowleft, upright, lowright])


class Load_DEM_artificial():

    def __init__(self, grid, fd=2.0, extent=None, resolution=None, d_z=None):
        """
        Class to create a random topography based on a fractal grid algorithm.
        Args:
            fd:         fractal dimension, defaults to 2.0
            d_z:        maximum height difference. If none, last 20% of the model in z direction
            extent:     extent in xy direction. If none, geo_model.grid.extent
            resolution: desired resolution of the topography array. If none, geo_model.grid.resolution
        """
        self.grid = grid

        self.resolution = grid.resolution[:2] if resolution is None else resolution

        assert all(np.asarray(self.resolution) >= 2), 'The regular grid needs to be at least of size 2 on all ' \
                                                     'directions.'
        self.extent = self.grid.extent[:4] if extent is None else extent

        if d_z is None:
            self.d_z = np.array(
                [self.grid.extent[5] - (self.grid.extent[5] - self.grid.extent[4]) * 1 / 5,
                 self.grid.extent[5]])
            print(self.d_z)
        else:
            self.d_z = d_z

        topo = self.fractalGrid(fd, N=self.resolution.max())
        topo = np.interp(topo, (topo.min(), topo.max()), self.d_z)

        self.dem_zval = topo[:self.resolution[1], :self.resolution[0]]  # crop fractal grid with resolution
        self.create_topo_array()

    def fractalGrid(self, fd, N=256):
        '''
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

        '''
        H = 1 - (fd - 2)
        #X = np.zeros((N, N), complex)
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

        itemp = fftpack.ifft2(A)
        itemp = itemp - itemp.min()

        return itemp.real / itemp.real.max()

    def create_topo_array(self):
        '''for masking the lith block'''
        x = np.linspace(self.grid.values[:, 0].min(), self.grid.values[:, 0].max(), self.resolution[1])
        y = np.linspace(self.grid.values[:, 1].min(), self.grid.values[:, 1].max(), self.resolution[0])
        xx, yy = np.meshgrid(x, y, indexing='ij')
        self.values_3D = np.dstack([xx.T, yy.T, self.dem_zval.T])

