'''Mapping module for GemPy.

Adds functions to crop the model based on the land surface or irregularly-shaped boundaries, and to plot custom diagonal cross-sections.
Based on code from Elisa.

See accompanying notebook for examples of how to use.'''

#Imports:
import sys, os
import numpy as np
import pandas as pn
import matplotlib.pyplot as plt
import matplotlib
import gdal
from copy import copy

sys.path.append("../../..")   #optional: if gempy has been downloaded from GitHub rather than installed normally, look for it in the folders above the current folder
import gempy as gp

#############################################################################
def importDEM(filename, show=True):
    '''Import DEM from a tif file using gdal package.
    Return a dem object, and xyz extent and resolution.
    (this can be used to set the model extent)
    NOTE: vertical (z) resolution can't be extracted from the raster!
    
    filename: string indicating the filename (must be a rectangular tif)
    show:     option to show a plot of the DEM or not.
    
    xmin:     minimum x value (same for ymin, zmin)
    xmax:     maximum x value (same for ymax, zmax)
    xres:     x resolution, aka number of columns, aka number of cells along x axis (NOT pixel width)
    etc.
    '''
    
    dem = gdal.Open(filename)    #DEM must be rectangular tif 
    dema = dem.ReadAsArray()     #copy of DEM as a numpy array (defaults to integers)
    dema = dema.astype(float)    #convert integer array to float array
    dema[dema==0] = np.nan       #replace zeros with NaNs (have to convert array to float first)

    ulx, pixelwidthx, xskew, uly, yskew, pixelheighty = dem.GetGeoTransform() #get resolution and coordinate info (for some reason the order of skew and pixel size is flipped for y axis?!)
    ncol = dem.RasterXSize            #number of columns (aka number of cells along x axis)
    nrow = dem.RasterYSize            #number of rows (aka number of cells along y axis)
    lrx = ulx + (ncol * pixelwidthx)  #lower right x coord = upper left x coord + (width of raster cells in x direction * number of raster cells in x direction)
    lry = uly + (nrow * pixelheighty)

    #Get min and max elevations (z):
    #note: gdal's built-in GetRasterBand and GetStatistics return an incorrect zmin (WHY?!)
    zmin = np.nanmin(dema)
    zmax = np.nanmax(dema)
    
    #Assign useful names:
    xmin = ulx
    xmax = lrx
    xres = ncol
    dx =   abs(pixelwidthx)
    ymin = lry
    ymax = uly
    dy =   abs(pixelheighty)
    yres = nrow
    zres = 'na'     #can't be extracted from raster

    #Print results & display raster:
    if show==True:
        print('Raster dimensions: \nxmin: {:<12} xmax: {:<12} xres: {} \nymin: {:<12} ymax: {:<12} yres: {} \nzmin: {:<12} zmax: {:<12} zres: {}'.format(
            xmin,xmax,xres,ymin,ymax,yres,zmin,zmax,zres))
        plt.imshow(dema, extent=(xmin,xmax,ymin,ymax), vmin=zmin, vmax=zmax) #plot raster as image
        #print(gdal.Info(dem))  #for more detailed file info, uncomment this line
        
    return dem,dema,xmin,xmax,xres,dx,ymin,ymax,yres,dy,zmin,zmax,zres



#############################################################################
def get_surflith(dem, dema, interp_data, grid_info, output_filename='DEMxyz.csv'):
    '''Reshape DEM and use it to compute the lithology values of the GemPy model at the land surface.
    Returns an array of lith values at the surface z elevation for each xy point.
    
    dem:                dem object returned by importDEM() or dem = gdal.Open(filename)
    dema:               dem array returned by importDEM() or dema = dem.ReadAsArray()
    interp_data:        interpolated data returned by gempy.InterpolatorData()
    grid_info:          [xmin,xmax,xres,dx,ymin,ymax,yres,dy,zmin,zmax,zres,dz] array of model grid and resolution info from importDEM()
    output_filename:    string to name gdal's output csv file (can be a throw-away - not used again)
    
    returns:
    surflith:           an array of lith values at the surface z elevation for each xy point, dim (yres,xres)'''
    
    #Get required grid info: 
    ##grid_info = [xmin,xmax,xres,dx,ymin,ymax,yres,dy,zmin,zmax,zres,dz]
    xres = grid_info[2]
    yres = grid_info[6]
    
    #Get an array with xyz values from the DEM:
    #can this be streamlined to avoid having to export and re-import?
    translate_options = gdal.TranslateOptions(options = ['format'],format = "XYZ")  #set options for gdal.Translate()
    gdal.Translate(output_filename, dem, options=translate_options)  #convert dem to a csv with one column of points, each with an xyz value
    xyz = pn.read_csv(output_filename, header=None, sep = ' ')       #read xyz csv with pandas
    demlist = xyz.values  #convert to np array of (x,y,z) values with dim (ncol*nrow, 3)
    
    #Get and format the geologic data:
    surfall, fault2 = gp.compute_model_at(demlist, interp_data) #compute the model values at the locations specified (aka the land surface) (why is fault a required output?)
    surflith = surfall[0].reshape(dema.shape) #reshape lith block (a list) to an array with same dimensions as dem (yres,xres,zres) (note: xres*yres must equal length of lith)
    #now we have a discretized array with the same resolution as the dem, with a value for the lithology at the surface elevation for each xy point
    surflith[np.isnan(dema)] = np.nan       #crop to dem boundary (insert nan values everywhere where the dem has no data) 
    surflist = surflith.reshape(xres*yres)  #reshape cropped data to list format (with dimensions xres*yres)
    
    return surflith, surflist 



#############################################################################
def crop2elevation(lith, dema, grid_info):
    '''Discretizes lith block into an array matching the model dimensions (yres,xres,zres), i.e. (nrow,ncol,nlay), 
    then crops off values above the land surface and replaces them with np.nan.
    
    IMPORTANT: lith returned by gempy.compute_model() is two arrays (each with dim: model extent) with a formation number assigned to each cell, and the orientation of that formation at each cell
    The format of lith seems to change if fault is present or not, so need to index differently to get the right slice:
    fault present:    lith[0]
    no fault present: lith[0][0]
    
    lith:       array of lithological unit values of dimensions (slice of lith block array returned by gempy.compute_model() - either lith[0] if fault present, or lith[0][0] if no fault present)
    dema:       array of elevation values generated from DEM (use importDEM(), or dem = gdal.Open(filename) & dema = dem.ReadAsArray())
    grid_info:  [xmin,xmax,xres,dx,ymin,ymax,yres,dy,zmin,zmax,zres,dz] array of model grid and resolution info from importDEM()
    
    returns:
    lithzcrop:  elevation-cropped array of lithologic unit indices of dimensions (nrow,ncol,nlay), i.e. (yres,xres,zres).'''

    #Get required grid info: 
    ##grid_info = [xmin,xmax,xres,dx,ymin,ymax,yres,dy,zmin,zmax,zres,dz]
    xres = grid_info[2]
    yres = grid_info[6]
    zmin = grid_info[8]
    zmax = grid_info[9]
    zres = grid_info[10]
    dz =   grid_info[11]

    #Get lith array into a shape that matches the model dimensions:
    lith2 = lith.reshape([xres,yres,zres]) #reshape lith block (a list) to an array with same dimensions as model (xres,yres,zres) (note: length of lith mst equal mxres*myres*mzres) 
    lith3 = np.transpose(lith2,(1,0,2)) #transpose swaps order of axes (so here, I am flipping x and y (aka 0 and 1))
    lith4 = np.flip(lith3,axis=0)       #transpose didn't correctly map y (north-south) axis, so need to flip just that axis

    #Convert the DEM to an array of vertical cell indices:
    #i.e. how many cells (aka layers) up from the base of the model is each (x,y) location?
    zvals = np.linspace(zmin, zmax, zres)  #create linearly-spaced z values within model range
    zind = (dema - zmin) / dz              #calculate the cell index of each point in the dem array using the cell height (i.e. how many cells/layers up it is from the base)
    zind = zind.astype(int)                #convert to integers for use as vertical indices

    #Remove the model values above the land surface:
    lithzcrop = copy(lith4)                 #make a copy to avoid messing up original (is this necessary?)
    for row in range(yres):                 #loop over rows (y axis)
        for col in range(xres):             #loop over columns (x axis)
            z = zind[row,col]               #get z index at current row and col
            lithzcrop[row,col,z:] = np.nan  #assign nan to all cells greater than z of land surface
            
    return lithzcrop



#############################################################################
def crop2raster(lith, grid_info, rasterfilename, nanval=0):
    '''Crop the extent of geologic model to the extent of an irregularly-shaped imported raster with a set value indicating empty cells.
    
    lith:           array of lithologic unit indices of dimensions (yres,xres,zres) OR dimensions (yres,xres)
                    this can be the array of surface lith values, all lith values, or elevation-cropped lith values
    grid_info:  [xmin,xmax,xres,dx,ymin,ymax,yres,dy,zmin,zmax,zres,dz] array of model grid and resolution info from importDEM()
    rasterfilename: string indicating the name of the raster file to use for cropping (can be bigger but not smaller than model extent) 
    nanval:         value indicating empty cells in raster file
    
    returns:
    lithxycrop:     array of same dimensions as input, but with empty cells filled with np.nan'''
    
    #Get grid info:
    xres = grid_info[2]
    yres = grid_info[6]
    
    #Import & format raster:
    geo = gdal.Open(rasterfilename)     #import raster file
    geoa = geo.ReadAsArray()            #read file as array (output will have an array for each color channel)
    geoa = geoa[-yres:,0:xres]          #if raster doesn't match DEM size, slice to size
    geoa = geoa.astype(float)           #convert integer array to float array
    geoa[geoa==0] = np.nan              #replace zeros with NaNs (have to convert array to float first)

    #Crop lith array:
    lithxycrop = lith.copy()                  #make a copy to avoid messing up original
    if lithxycrop.shape == geoa.shape:        #if arrays have same extent
        lithxycrop[np.isnan(geoa)] = np.nan   #crop lith to active cells in imported raster
    else:                                     #otherwise, assume lith is 3D
        lithxycrop[np.isnan(geoa),:] = np.nan #crop over all z cells
    
    return lithxycrop



#############################################################################
def set_colorscheme(namestring):
    '''Get list of unit names, colors, and colormap and normalization. Use GemPy colors, or set custom colors by modifying the ArcMap scheme.
    For now, choose namestring = either 'GemPy' or 'ArcMap'.'''
    if namestring=='GemPy':
        unitnames = ['NaN','Garschella','Schrattenkalk','Drusberg','Basement'] #GemPy unit names (NaN required)
        cmap = gp.plotting.colors.cmap                     #get GemPy colormap
        colors = [cmap(i) for i in range(len(unitnames))]  #get list of rgb values from GemPy colormap (but this is not a good way to do it)
        norm = gp.plotting.colors.norm                     #color normalization from GemPy
    if namestring=='ArcMap':
        unitnames = ['Quaternary','Amdener','Garschella','Schrattenkalk','Drusberg','Flysch'] #ArcMap unit names
        colors = ['khaki','orange','orange','c','pink','olivedrab']                           #ArcMap colors
        cmap = matplotlib.colors.ListedColormap(colors)                                       #map colors to values
        norm = matplotlib.colors.Normalize()
    return unitnames,colors,cmap,norm  #return list of unit names, list of colors, and colormap object




#############################################################################
def compare_geology(surflith,geoa,colorscheme='GemPy'):
    '''Plots the GemPy-generated geologic map next to the true geologic map for comparison. 
    surflith:    array of lithologic unit values at the land surface, of dimensions (yres,xres), i.e. (nrow,ncol)
    geoa:        array generated from a raster file of the actual geologic map, of same dimensions as surflith
    colorscheme: string indicating which color scheme to use (generated by colorscheme function)
    '''
    
    names,colors,cmap,norm = set_colorscheme('GemPy')
    f,ax = plt.subplots(1,2,figsize=(20,20))
    ax[0].imshow(surflith, cmap=cmap, norm=norm)    #plot lith surface 
    ax[0].set_title('GemPy')
    ax[1].imshow(geoa, cmap=cmap, norm=norm)        #plot imported geol
    ax[1].set_title('ArcMap')
    patches = [matplotlib.patches.Patch(color=colors[i], label=names[i]) for i in range(len(names))] #set up legend by plotting patches 
    ax[1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2) #put patches into legend & set location

    return f,ax




#############################################################################
def plotXsection(startpoints, endpoints, names, grid_info, lith, surflith, vscale=1, cmap=gp.plotting.colors.cmap, norm=gp.plotting.colors.norm):
    '''Plots an approximate cross-section between the two specified points, using an elevation-cropped array (cells above the land surface should have nan values).
    startpoints: [[x1,y1],[x2,y2],...] float or [[col1,row1],[col1,row2],...] integer array of coordinates of starting points A
    endpoints:   [[x1,y1],[x2,y2],...] float or [[col1,row1],[col1,row2],...] integer array of coordinates of ending points B
    names:       [['A','B'],['C','D'],...] string array of names for starting and ending points
    grid_info:   [xmin,xmax,xres,dx,ymin,ymax,yres,dy,zmin,zmax,zres,dz] model grid info (can get these using importDEM() function if model grid is same as DEM grid)
    lith:        elevation-cropped array of lithologic unit indices of dimensions (nrow,ncol,nlay), i.e. (yres,xres,zres). Can use uncropped array, but will plot above land surface.
    surflith:    array of lithologic unit values at the land surface, of dimensions (yres,xres)
    vscale:      vertical exaggeration factor (y/x, defaults to 1)
    cmap:        colormap to plot with (defaults to gempy colormap)
    norm:        colormap normalization (defaults to gempy normalization)
    '''
    
    #Get model grid and resolution info:
    xmin = grid_info[0]
    xres = grid_info[2]
    dx =   grid_info[3]
    ymin = grid_info[4]
    yres = grid_info[6]
    dy =   grid_info[7]
    zmin = grid_info[8]
    zmax = grid_info[9]
    zres = grid_info[10]
    dz =   grid_info[11]
    
    
    #Plot geologic map once for reference:
    f1 = plt.figure(figsize=(10,10))                            #create empty figure
    plt.imshow(surflith, cmap=cmap, norm=norm)                  #plot geology (normalized to gempy color range)
    f2,ax2 = plt.subplots(len(startpoints),1,figsize=(10,20))   #create figure and axes objects for subplots (one per xsection)
    
    for i in range(len(startpoints)):   #loop over number of sections
        #Get starting coordinates:
        xA = startpoints[i][0]   #get starting x coordinate
        yA = startpoints[i][1]   #get starting y coordinate
        xB = endpoints[i][0]     #get ending x coordinate
        yB = endpoints[i][1]     #get ending y coordinate

        #Calculate corresponding row,col
        if type(xA) != int:                     #if coordinates are NOT integers (i.e. not row,col numbers), convert them
            colA = (xA - xmin)//dx              #col:x calculate column index  c = (x1-x0)/dx 
            rowA = yres - ((yA - ymin)//dy)     #row:y calculate row index     r = ymax - (y1-y0)/dy
            colB = (xB - xmin)//dx                 
            rowB = yres - ((yB - ymin)//dy) 
        else:                                  #if coordinates are already in row,col format
            colA = xA
            rowA = yA
            colB = xB
            rowB = yB

        #Calculate line equation between points A and B:
        m = (rowB - rowA) / (colB - colA)   #calculate slope     m = (y2-y1)/(x2-x1)
        b = -m*colA + rowA                  #calculate intercept b = m*x1 + y1 (slope is neg here bc y axis is flipped)

        #Get xy indices for cells intersected by the x-sec line, then get z values for those xy points:
        xvals = np.arange(colA,colB)    #generate array of x values between the two points
        xvals = xvals.astype(int)       #convert to integer
        yvals = m*xvals + b             #calculate corresponding y values  y = mx + b 
        yvals = yvals.astype(int)       #convert to integers to be able to slice

        xsec = lith[yvals,xvals,:].T    #select x-sec to plot and transpose to make it plot horizontally 

        #Plotting:
        
        #Add xsection lines to geologic map:
        plt.figure(f1.number)                           #make the map the active figure
        plt.plot([colA,colB],[rowA,rowB],'k')           #plot x-sec location line
        plt.annotate(names[i][0],xy=(colA,rowA),xytext=(colA-4,rowA+4)) #annotate start point
        plt.annotate(names[i][1],xy=(colB,rowB),xytext=(colB+1,rowB-1)) #annotate start point
        plt.ylim(bottom=yres, top=0) 
        plt.xlim(left=0, right=xres)

        #Plot cross-sections in a new figure:
        #Set and get correct subplot axes:
        if len(startpoints) == 1:               #check if there are more than 1 subplots (for indexing purposes)
            plt.sca(ax2)                        #make current subplot axes active (automatically makes fig active too)
            cax = plt.gca()                     #get current axes object
        else:          
            plt.sca(ax2[i])                     
            cax = plt.gca()
        cax.imshow(xsec, origin="lower", cmap=cmap, norm=norm)   #plot (with down=lower z indices)
        cax.set_aspect(vscale*dz/dx)                             #apply vertical exaggeration
        cax.set_ylim(bottom=0, top=zres)                     
        cax.set_title(names[i][0]+names[i][1])
        cax.set_anchor('W')                                      #align left (West)

        #Set ticks to accurately reflect elevation (masl):
        locs = cax.get_yticks()                               #get tick locations
        nlabels = len(cax.get_yticklabels())                  #get number of initial ticks 
        labels = np.linspace(zmin, zmax, nlabels)             #generate list of tick labels
        ticks = cax.set(yticks=locs,yticklabels=labels)       #set tick locations and labels
    
    return f1,f2,ax2