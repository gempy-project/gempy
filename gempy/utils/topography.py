"""
    This file is part of gempy.

    gempy is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    gempy is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with gempy.  If not, see <http://www.gnu.org/licenses/>.

    @author: Elisa Heim
"""

try:
    import gdal
except ImportError:
    import warnings
    warnings.warn("gdal package is not installed. No support for raster functions")

import numpy as np
import matplotlib.pyplot as plt

import gempy as gp
import pandas as pn

import skimage

class DEM():
    '''Class to include height elevation data (e.g. DEMs) with the geological model '''

    def __init__(self, path_dem, geodata=None, interpdata=None, output_path=None):
        '''
        Args:
            path_dem: path where dem is stored. file format: GDAL raster formats
            geodata: geo_data object
            output_path: path to a folder. Must be defined for gdal to perform modifications on the raster
        '''
        if path_dem:
            self.dem = gdal.Open(path_dem)
        else:
            print('Define path to raster file')

        self.dem_zval = self.dem.ReadAsArray()
        self.raster_extent, self.raster_resolution = self._get_raster_dimensions()

        if geodata:
            self.geo_data = geodata
            self.extent_match = self.compare_extent()

        if self.extent_match == False:  # crop dem and update values
            if output_path:
                self.dem = self.cropDEM2geodata(output_path)
                self.dem_zval = self.dem.ReadAsArray()
                self.raster_extent, self.raster_resolution = self._get_raster_dimensions()
                self.extent_match = self.compare_extent()
            else:
                print('extents of DEM and geodata do not match. To see a comparison, use self.show(compare=True). '
                      'For automatic cropping, define an output path')

        self.grid_info = self._get_grid_info()  # daraus tabelle machen für übersicht
        self.dem_resized = skimage.transform.resize(self.dem_zval,
                                                    (self.geo_data.resolution[0], self.geo_data.resolution[1]),
                                                    preserve_range=True)

        if output_path:  # create file only when extents match
            self.surface_coordinates = self.convertDEM2xyz(output_path)
        else:
            print('for the full spectrum of plotting with topography, please define an output path')
        if interpdata:
            self.interp_data = interpdata

    def compare_extent(self):
        '''
        Returns:
        '''
        if self.geo_data:
            cornerpoints_geo = self._get_cornerpoints(self.geo_data.extent)
            cornerpoints_dtm = self._get_cornerpoints(self.raster_extent)
            if np.any(cornerpoints_geo[:2] - cornerpoints_dtm[:2]) != 0:
                # print('Extent of geo_data and DEM do not match. Use function cropDEMtogeodata to crop')
                return False
            else:
                # print('geodata and dem extents match')
                return True

    def show(self, compare=False,plot_data=False, **kwargs):
        '''

        Args:
            compare:
            plot_data:
            **kwargs: gp.plotting.plot_data kwargs

        Returns:

        '''
        #Todo add legend that shows elevation
        print('Raster extent:', self.raster_extent,
              '\nRaster resolution:', self.raster_resolution)
        if plot_data:
            gp.plotting.plot_data(self.geo_data, direction='z', data_type='all', **kwargs)
        plt.imshow(self.dem_zval, extent=(self.raster_extent[:4]))
        if compare == True:
            if self.geo_data:
                cornerpoints_geo = self._get_cornerpoints(self.geo_data.extent)
                cornerpoints_dtm = self._get_cornerpoints(self.raster_extent)
                if self.extent_match == False:
                    plt.plot(cornerpoints_geo[:, 0], cornerpoints_geo[:, 1], 'ro', markersize=7,
                             label='Geo_data extent')
                    plt.plot(cornerpoints_dtm[:, 0], cornerpoints_dtm[:, 1], 'gX', markersize=11, label='DTM extent')
                    plt.legend(loc=0, fancybox=True, shadow=True)
                else:
                    print('geodata and dem extents match')

    def cropDEM2geodata(self, output_path):
        '''
        Args:
            output_path:
        Returns:
        '''
        path_dest = output_path + '_cropped_DEM.tif'
        print('Extents of geo_data and DEM do not match. DEM is cropped and stored as', path_dest)
        new_bounds = (
        self.geo_data.extent[0], self.geo_data.extent[2], self.geo_data.extent[1], self.geo_data.extent[3])
        # destName = "C:\\Users\\elisa\\Documents\\git\\MSc\\GempyTopography\\cropped_DTM.tif"
        gdal.Warp(path_dest, self.dem, options=gdal.WarpOptions(
            options=['outputBounds'], outputBounds=new_bounds))
        # cropped_dem = gdal.Open(path_dest)
        # return topography(path_dest, self.geo_data, output_path)
        return gdal.Open(path_dest)

    def convertDEM2xyz(self, output_path):
        '''
        Args:
            output_path:
        Returns: array with the x,y,z coordinates of the topography  [0]: shape(a,b,3), [1]: shape(a*b,3)
        '''

        path_dest = output_path + '_gempytopo.xyz'
        shape = self.dem_zval.shape
        gdal.Translate(path_dest, self.dem, options=gdal.TranslateOptions(options=['format'], format="XYZ"))
        xyz = pn.read_csv(path_dest, header=None, sep=' ').values
        #xyz_box = np.dstack([xyz[:, 0].reshape(shape), xyz[:, 1].reshape(shape), xyz[:, 2].reshape(shape)])
        x = xyz[:, 0].reshape(shape)
        y = xyz[:, 1].reshape(shape)
        z = xyz[:, 2].reshape(shape)
        x = np.flip(x, axis=0)
        y = np.flip(y, axis=0)
        z = np.flip(z, axis=0)
        xyz_box = np.dstack([x,y,z])
        return xyz, xyz_box

    def calculate_geomap(self, interpdata = None, plot=True):
        '''
        Args:
            interpdata:
            plot:
        Returns:
        '''
        if interpdata:
            geomap, fault = gp.compute_model_at(self.surface_coordinates[0], interpdata)
        else:
            geomap, fault = gp.compute_model_at(self.surface_coordinates[0], self.interp_data)
        geomap = geomap[0].reshape(self.dem_zval.shape)  # resolution of topo gives much better map
        geomap = np.flip(geomap, axis=0) #to match the orientation of the other plotting options
        if plot:
            plt.imshow(geomap, origin="lower", cmap=gp.plotting.colors.cmap, norm=gp.plotting.colors.norm)  # set extent
            plt.title("Geological map", fontsize=15)
        return geomap

    def _slice(self, direction, extent, cell_number=25):
        '''
        Args:
            direction:
            extent:
            cell_number:
        Returns:
        '''
        surface_dem = self.surface_coordinates[1]
        x = surface_dem[:, :, 0]
        y = surface_dem[:, :, 1]
        z = surface_dem[:, :, 2]

        if direction == 'y':
            a = x[cell_number, :]
            b = y[cell_number, :]
            c = z[cell_number, :]
            assert len(np.unique(b)) == 1
            topoline = np.dstack((a, c)).reshape(-1, 2).astype(int)
            upleft = np.array([extent[0], extent[3]])
            upright = np.array([extent[1], extent[3]])
            topolinebox = np.append(topoline, (upright, upleft), axis=0)

        elif direction == 'x':
            a = x[:, cell_number]
            b = y[:, cell_number]
            c = z[:, cell_number]
            assert len(np.unique(a)) == 1
            topoline = np.dstack((b, c)).reshape(-1, 2).astype(int)
            upleft = np.array([extent[0], extent[3]])
            upright = np.array([extent[1], extent[3]])
            topolinebox = np.append(topoline, (upright, upleft), axis=0)

        elif direction == "z":
            print('not implemented')

        return topolinebox

    def _get_raster_dimensions(self):
        '''returns dtm.extent and dtm.resolution'''
        ulx, xres, xskew, uly, yskew, yres = self.dem.GetGeoTransform()
        z = self.dem_zval
        if np.any(np.array([xskew, yskew])) != 0:
            print('Obacht! DEM is not north-oriented.')
        lrx = ulx + (self.dem.RasterXSize * xres)
        lry = uly + (self.dem.RasterYSize * yres)
        res = np.array([(uly - lry) / (-yres), (lrx - ulx) / xres]).astype(int)
        return np.array([ulx, lrx, lry, uly, z.min(), z.max()]).astype(int), res

    def _get_cornerpoints(self, extent):
        upleft = ([extent[0], extent[3]])
        lowleft = ([extent[0], extent[2]])
        upright = ([extent[1], extent[3]])
        lowright = ([extent[1], extent[2]])
        return np.array([upleft, lowleft, upright, lowright])

    def _get_grid_info(self):
        ext = self.geo_data.extent
        xres, yres, zres = self.geo_data.resolution[0], self.geo_data.resolution[1], self.geo_data.resolution[2]
        # ext = self.raster_extent
        # xres,yres,zres= self.raster_extent[0],self.raster_extent[1],25
        dx = (ext[1] - ext[0]) / xres
        dy = (ext[3] - ext[2]) / yres
        dz = (ext[5] - ext[4]) / zres
        # return np.array([ext[0],ext[1],xres,dx,ext[2],ext[3],yres,dy,ext[4],ext[5],zres,dz]).reshape(3,4).astype(int)
        return np.array([ext[0], ext[1], xres, dx, ext[2], ext[3], yres, dy, ext[4], ext[5], zres, dz]).astype(int)

