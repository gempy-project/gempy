'''gdal functions for gempy from elisa
useful for cropping model extent to topography, and projecting 2D maps & x-secs with topography'''

import numpy as np
import gdal
from copy import copy
import matplotlib.pyplot as plt
import pandas as pn


def import_dtm(path_dtm):
    '''returns: osgeo.gdal.Dataset'''
    #import_data_csv
    if path_dtm:
        dtm = gdal.Open(path_dtm)
    return dtm

def compare_extent(dtm, geo_data, show=True):
    dtm_extent, dtm_resolution = gdal2geodata_extent(dtm)
    cornerpoints_geo = get_cornerpoints(geo_data.extent)
    cornerpoints_dtm = get_cornerpoints(dtm_extent)
    
    if show:
        plt.style.use('bmh')
        plt.plot(cornerpoints_geo[:,0], cornerpoints_geo[:,1], 'ro', markersize = 12, label = 'Geo_data extent')
        plt.plot(cornerpoints_dtm[:,0], cornerpoints_dtm[:,1], 'gX',markersize = 11, label = 'DTM extent')
        plt.title('Extent comparison')
        plt.legend(loc=0, fancybox=True, shadow=True)
        plt.show()
        
    if np.any(cornerpoints_geo[:2]-cornerpoints_dtm[:2]) != 0:
        print('Extent of geo_data and DTM do not match. Use function cropDTMtogeodata to crop')
    else:
        print('Extent of geo_data and DTM match. You may continue!')
    
def gdal2geodata_extent(dtm, nanval=0):
    '''can return dtm.extent and dtm.resolution'''
    ulx, xres, xskew, uly, yskew, yres  = dtm.GetGeoTransform() #res means pixel size
    dtma = dtm.ReadAsArray()
    if np.any(np.array([xskew,yskew]))!= 0:
        #xskew = 0 if north-oriented
        print('Obacht! DTM is not north-oriented. Stop.')
    #lower right x coord = upper left x coord + (number of raster cells in x direction * width of raster cells in x dirction)
    lrx = ulx + (dtm.RasterXSize * xres)  
    lry = uly + (dtm.RasterYSize * yres)
    if dtma.min() == nanval:  #filter if there is a numeric value for nans
        print('Min z val is equal to nan val - making a copy of the array as floats with np.nan to find true min')
        dtmf = dtma.astype(float)       #make a copy array with floats
        dtmf[dtmf[:]==nanval]=np.nan      #replace nan vals with np.nan
        zmin = np.nanmin(dtmf)          #get min value excluding nans
    else: zmin = dtma.min()              #if no nan conflicts, proceed as normal
    extent = np.array([ulx, lrx, lry, uly, zmin, dtma.max()]).astype(int)
    res = np.array([abs((uly-lry)/yres),abs((lrx-ulx)/xres)]).astype(int)   #this should be x,y not y,x. Also here res means size of each cell (how is this different than RasterXSize?)
    return extent, res

def get_cornerpoints(extent):
    upleft = ([extent[0], extent[3]]) 
    lowleft = ([extent[0], extent[2]]) 
    upright = ([extent[1], extent[3]]) 
    lowright = ([extent[1], extent[2]]) 
    return np.array([upleft,lowleft,upright,lowright])

def cropDTM2geodata(path_dest, dtm, geo_data):
    new_bounds = (geo_data.extent[0], geo_data.extent[2], geo_data.extent[1], geo_data.extent[3])
    #destName = "C:\\Users\\elisa\\Documents\\git\\MSc\\GempyTopography\\cropped_DTM.tif"
    gdal.Warp(path_dest, dtm, options = gdal.WarpOptions(
        options = ['outputBounds'], outputBounds=new_bounds))
    return gdal.Open(path_dest)

def tif2xyz(path_dest, dtm):
    '''returns array with the x,y,z coordinates of the topography.'''
    shape = dtm.ReadAsArray().shape
    #print(shape)
    gdal.Translate(path_dest, dtm, options=gdal.TranslateOptions(options = ['format'],format = "XYZ"))
    xyz = pn.read_csv(path_dest, header=None, sep = ' ').as_matrix()
    return np.dstack([xyz[:,0].reshape(shape),xyz[:,1].reshape(shape),xyz[:,2].reshape(shape)])

def height_ind(dtm, zs):
    '''calculate 'indices': array with shape (ext1, ext2) where the values
    are similar if the elevation is similar.'''  
    #zs = np.linspace(geo_data.extent[4], geo_data.extent[5], lb.shape[2])
    #dz = geo_data.extent[5] - geo_data.extent[4])/geo_data.resolution[2]
    # dz is number of points in z direction
    dz = (zs[-1] - zs[0]) / len(zs)
    #2. substract minimum value (here -1000) from each value of dtm and divide it by dz
    dtm_v = (dtm - zs[0]) / dz
    return dtm_v.astype(int)


def calculate_geomap(lb, dtm, geo_data, plot=True):
    zs = np.linspace(geo_data.extent[4], geo_data.extent[5], lb.shape[2])
    indices = height_ind(dtm,zs).T   
    geomap = np.zeros((lb.shape[0],lb.shape[1]))
    for x in range(lb.shape[0]):
        for y in range(lb.shape[1]):
            geomap[x,y] = lb[x,y,indices[x,y]]   
    if plot:
        plt.imshow(geomap.T, origin="upper", cmap=gp.plotting.colors.cmap, norm=gp.plotting.colors.norm)
        plt.title("Geological map")    
    return geomap.T  

def extend_lithblock(lb, factor):
    fertig2 = []
    for i in range(0,lb.shape[2]):
        lb_sub=lb[:,:,i]
        fertig = []
        for j in range(0, lb.shape[0]):
            y = np.repeat(lb_sub[j,:], factor)
            fertig = np.append(fertig, [y]*factor)
        fertig = fertig.reshape(lb.shape[0]*factor, lb.shape[1]*factor)
        fertig2.append(fertig)
        fertig2.append(fertig)
        fertig2.append(fertig)
        fertig2.append(fertig)
    return np.dstack(fertig2)

def mask_lith_block_above_topo(lb, geo_data, dtm):
    '''hier wird allen Werten die oberhalb der topographie 
    liegen ein minus eins zugewiesen und das dann maskiert'''
    zs = np.linspace(geo_data.extent[4], geo_data.extent[5], lb.shape[2])
    indices = height_ind(dtm,zs).T

    geoblock = copy(lb)
    for x in range(lb.shape[0]):
        for y in range(lb.shape[1]):
            z = indices[x,y]
            geoblock[x,y,z:] = -1
    
    return np.ma.masked_where(geoblock < 0, geoblock)