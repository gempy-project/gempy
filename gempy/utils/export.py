
from matplotlib.cm import ScalarMappable as SM
from gempy.plot.visualization_2d import PlotData2D


def export_geomap2geotiff(path, geo_model, geo_map=None, geotiff_filepath=None):
    """

    Args:
        path (str): Filepath for the exported geotiff, must end in .tif
        geo_map (np.ndarray): 2-D array containing the geological map
        cmap (matplotlib colormap): The colormap to be used for the export
        geotiff_filepath (str): Filepath of the template geotiff

    Returns:
        Saves the geological map as a geotiff to the given path.
    """
    import gdal

    plot = PlotData2D(geo_model)
    cmap = plot._cmap
    norm = plot._norm

    if geo_map is None:
        geo_map = geo_model.solutions.geological_map[0].reshape(geo_model.grid.topography.resolution)

    if geotiff_filepath is None:
        # call the other function
        print('stupid')

    # **********************************************************************
    geo_map_rgb = SM(norm=norm, cmap=cmap).to_rgba(geo_map.T) # r,g,b,alpha
    # **********************************************************************
    # gdal.UseExceptions()
    ds = gdal.Open(geotiff_filepath)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    [cols, rows] = arr.shape

    outFileName = path
    driver = gdal.GetDriverByName("GTiff")
    options = ['PROFILE=GeoTiff', 'PHOTOMETRIC=RGB', 'COMPRESS=JPEG']
    outdata = driver.Create(outFileName, rows, cols, 3, gdal.GDT_Byte, options=options)

    outdata.SetGeoTransform(ds.GetGeoTransform())  # sets same geotransform as input
    outdata.SetProjection(ds.GetProjection())  # sets same projection as input
    outdata.GetRasterBand(1).WriteArray(geo_map_rgb[:, ::-1, 0].T * 256)
    outdata.GetRasterBand(2).WriteArray(geo_map_rgb[:, ::-1, 1].T * 256)
    outdata.GetRasterBand(3).WriteArray(geo_map_rgb[:, ::-1, 2].T * 256)
    outdata.GetRasterBand(1).SetColorInterpretation(gdal.GCI_RedBand)
    outdata.GetRasterBand(2).SetColorInterpretation(gdal.GCI_GreenBand)
    outdata.GetRasterBand(3).SetColorInterpretation(gdal.GCI_BlueBand)
    # outdata.GetRasterBand(4).SetColorInterpretation(gdal.GCI_AlphaBand)  # alpha band

    # outdata.GetRasterBand(1).SetNoDataValue(999)##if you want these values transparent
    outdata.FlushCache()  # saves to disk
    outdata = None  # closes file (important)
    band = None
    ds = None

    print("Successfully exported geological map to  " +path)