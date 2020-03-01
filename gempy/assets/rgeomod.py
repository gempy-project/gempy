"""Link to geolgocial modeling from remote data sorces, like GoogleEarth, Earth-Explorer, etc.

2020, Florian Wellmann
"""
import pandas as pn
import os
from .geographic import *
# from .struct_geo import *
from tqdm import tqdm
from time import sleep
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from osgeo import ogr, osr
import gdal
import matplotlib.pyplot as plt


# TODO: we should rework the storage of xyz coordinates from single .x, .y, .z class-variables to np.ndarray


def dip(normal_vec):
    return np.arccos(normal_vec[2]) / np.pi * 180.


def dip_dir(normal_vec):
    # +/+
    if normal_vec[0] >= 0 and normal_vec[1] > 0:
        return np.arctan(normal_vec[0] / normal_vec[1]) / np.pi * 180.
    # border cases where arctan not defined:
    elif normal_vec[0] > 0 and normal_vec[1] == 0:
        return 90
    elif normal_vec[0] < 0 and normal_vec[1] == 0:
        return 270
    # +-/-
    elif normal_vec[1] < 0:
        return 180 + np.arctan(normal_vec[0] / normal_vec[1]) / np.pi * 180.
    # -/-
    elif normal_vec[0] < 0 <= normal_vec[1]:
        return 360 + np.arctan(normal_vec[0] / normal_vec[1]) / np.pi * 180.


def check_point_sets(data):
    """Checks if point sets in KmlPoints object contain at least three data points for
    fitting a plane. If not it removes the unsuitable point sets."""
    for i, ps in enumerate(data.point_sets):
        if len(ps.points) < 3:
            data.point_sets.remove(ps)
            print("Removed point set #" + str(i))


def extract_xyz(k):
    x = []
    y = []
    z = []

    for i, ps in enumerate(k.point_sets):
        for j, p in enumerate(ps.points):
            x.append(p.x)
            y.append(p.y)
            try:
                z.append(p.z)
            except AttributeError:
                z.append(-9999)

    return np.array([x, y, z])


def points_to_gempy_interf(ks_coords, formations, filenames, series="Default series", debug=False):
    """Converts KmlPoints coordinates into GemPy interfaces dataframe.

    Args:
        ks_coords: list/array of point set arrays [[x,y,z], [x,y,z]]
        formations: list of fromation names [str, str, ...]
        series (str, optional): Set the series for the given point sets.
        debug (bool, optional): Toggles verbosity.
        filenames (list): filename of .kml file

    Returns:
        GemPy interfaces dataframe with columns ['X', 'Y', 'Z', 'formation', 'series', "formation number"].
    """
    interfaces = pn.DataFrame(columns=['X', 'Y', 'Z', 'formation', 'series', "formation number"])

    for i, k in enumerate(ks_coords):
        temp = pn.DataFrame(columns=['X', 'Y', 'Z', 'formation', 'series', "formation number"])
        if debug:
            print(i)

        temp["X"] = k[0]
        temp["Y"] = k[1]
        temp["Z"] = k[2]
        temp["formation"] = formations[i]
        temp["series"] = series
        temp["formation number"] = int(filenames[i].split("_")[0])

        interfaces = interfaces.append(temp, ignore_index=True)

    return interfaces


def dips_to_gempy_fol(dips, dip_dirs, xs, ys, zs, formation, formation_number, series="Default series"):
    """

    Args:
        dips:
        dip_dirs:
        xs:
        ys:
        zs:
        formation:
        formation_number:
        series:

    Returns:

    """
    # TODO: adjust to naming conventions of gempy
    foliations = pn.DataFrame(columns=['X', 'Y', 'Z', 'dip', 'azimuth', 'polarity', 'formation', 'series'])

    foliations["X"] = xs
    foliations["Y"] = ys
    foliations["Z"] = zs
    foliations["dip"] = dips
    foliations["azimuth"] = dip_dirs
    foliations["polarity"] = 1
    foliations["formation"] = formation
    foliations["series"] = series
    foliations["formation number"] = formation_number

    return foliations


def read_kml_files(folder_path, verbose=False):
    """Reads in all .kml files from given folder, creating a KmlPoints instance for each
    file found.
    Filename convention:
        '01_formationname_dips.kml' for dip picks.
        '04_formationname_interf.kml' for

    Args:
        folder_path (str): Relative path to the folder containing the picked points as .kml files (Google Earth).
        verbose (bool): Toggles verbosity.

    Returns:
        (list):
        (list):
        (np.array, boolean):
    """

    ks = []
    ks_names = []
    ks_bool = []
    filenames = []

    for i, fn in enumerate(os.listdir(folder_path)):
        if ".kml" in fn:
            ks.append(KmlPoints(filename=folder_path + fn, debug=verbose))
            if verbose:
                print(fn)

            filenames.append(fn)

            # auto check if some set contains less than 3 points and throw them out
            check_point_sets(ks[-1])
            if verbose:
                print("\n")

            ks_names.append(fn.split("_")[1])  # append formation name

            if "dips" in fn or "Dips" in fn or "foliation" in fn:
                ks_bool.append(True)
            else:
                ks_bool.append(False)

    return ks, ks_names, np.array(ks_bool).astype(bool), filenames


def get_elevation_from_dtm(geographic_point_sets, fname, verbose=True):
    """Get elevation value from GeoTiff dtm for entire geopgraphic point set

    All z-values are then added to the GeographicPoint objects

    Args:
        geographic_point_sets (list): list of GeographicPonitSet
        fname (filepath): filename of GeoTiff file
        verbose (bool): Show debug information

    Returns: None

    """
    for k in tqdm(geographic_point_sets, desc="Extracting elevation data"):
        sleep(0.2)
        for ps in k.point_sets:
            try:
                ps.get_z_values_from_geotiff(fname)
            except IndexError:
                if verbose:
                    print("Point outside geotiff, drop")
                k.point_sets.remove(ps)
                continue

    if verbose:
        print("Elevation data successfully extracted from DTM.")


def fit_planes_to_points(ks, verbose=True):
    for k in tqdm(ks, desc="Fitting planes to point sets"):
        sleep(0.3)
        for ps in k.point_sets:
            # convert LatLon coordinates to UTM
            try:
                ps.latlong_to_utm()
                # Fit plane to point set
                ps.plane_fit()
            except AttributeError:
                print("Point set is NoneType - ignored")

    if verbose:
        print("Planes successfully fit to point sets.")


def calc_dips_from_points(ks, ks_bool):
    dips = []
    dip_dirs = []
    dip_xs = []
    dip_ys = []
    dip_zs = []

    for k in np.array(ks)[ks_bool]:
        for ps in k.point_sets:
            print(ps.normal)
            if type(ps.normal) == float and np.isnan(ps.normal):
                continue
            # determine dip angle from normal vector of plane
            dips.append(dip(ps.normal))
            # get dip direction from normal vector
            dip_dirs.append(dip_dir(ps.normal))
            # get centroid coordinates
            dip_xs.append(ps.ctr.x)
            dip_ys.append(ps.ctr.y)
            dip_zs.append(ps.ctr.z)

    return dips, dip_dirs, dip_xs, dip_ys, dip_zs


def convert_to_df(ks, ks_names, filenames, ks_bool):
    """

    Args:
        ks:
        ks_names:
        ks_bool:
        filenames:

    Returns:

    """
    # interfaces
    # ----------
    ks_coords = []
    for k in ks:
        ks_coords.append(extract_xyz(k))

    ks_coords_interf = []
    ks_names_interf = []
    ks_filenames = []

    for i, k in enumerate(ks_coords):
        if not ks_bool[i]:
            ks_coords_interf.append(k)
            ks_names_interf.append(ks_names[i])
            ks_filenames.append(filenames[i])

    interfaces = points_to_gempy_interf(ks_coords_interf, ks_names_interf, ks_filenames)

    # foliations
    # ----------
    dips, dip_dirs, dip_xs, dip_ys, dip_zs = calc_dips_from_points(ks, ks_bool)
    foliations = dips_to_gempy_fol(dips, dip_dirs, dip_xs, dip_ys, dip_zs, ks_names[0], 1)
    # TODO: flexible formation number assignment for foliations

    return interfaces, foliations


def convert_dtm_to_gempy_grid(raster, dtm):
    # here are the raster dimensions:
    # raster.RasterXSize, raster.RasterYSize
    geoinformation = raster.GetGeoTransform()
    # get DTM corners:
    dtm_east_min = geoinformation[0]
    dtm_east_max = geoinformation[0] + geoinformation[1] * raster.RasterXSize
    dtm_north_min = geoinformation[3] + geoinformation[5] * raster.RasterYSize
    dtm_north_max = geoinformation[3]
    # dtm_east_min, dtm_east_max, dtm_north_min, dtm_north_max

    # define range for x, y - values
    x_range = np.arange(dtm_east_min, dtm_east_max, geoinformation[1])
    y_range = np.arange(dtm_north_min, dtm_north_max, np.abs(geoinformation[5]))
    xx, yy = np.meshgrid(x_range, y_range, indexing="ij")

    # Create list of input points for interpolation with gempy:
    return np.array(list(zip(xx.ravel(), yy.ravel(), dtm[::-1, :].T.ravel())))


def export_geotiff(path, geo_map, cmap, geotiff_filepath):
    """

    Args:
        path (str): Filepath for the exported geotiff, must end in .tif
        geo_map (np.ndarray): 2-D array containing the geological map
        cmap (matplotlib colormap): The colormap to be used for the export
        geotiff_filepath (str): Filepath of the template geotiff

    Returns:
        Saves the geological map as a geotiff to the given path.
    """

    # **********************************************************************
    geo_map_rgb = cmap(geo_map)  # r,g,b,alpha
    # **********************************************************************
    # gdal.UseExceptions()

    ds = gdal.Open(geotiff_filepath)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    [cols, rows] = arr.shape

    out_file_name = path
    driver = gdal.GetDriverByName("GTiff")
    options = ['PROFILE=GeoTiff', 'PHOTOMETRIC=RGB', 'COMPRESS=JPEG']
    outdata = driver.Create(out_file_name, rows, cols, 3, gdal.GDT_Byte, options=options)

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
    # outdata = None  # closes file (important)
    # band = None
    # ds = None

    print("Successfully exported geological map to " + path)


def calculate_gradient(foliations):
    """
    Calculate the gradient vector of module 1 given dip and azimuth to be able to plot the foliations

    Attributes:
        foliations: extra columns with components xyz of the unity vector.
    """

    foliations['G_x'] = np.sin(np.deg2rad(foliations["dip"].astype('float'))) * \
        np.sin(np.deg2rad(foliations["azimuth"].astype('float'))) * \
        foliations["polarity"].astype('float')
    foliations['G_y'] = np.sin(np.deg2rad(foliations["dip"].astype('float'))) * \
        np.cos(np.deg2rad(foliations["azimuth"].astype('float'))) * \
        foliations["polarity"].astype('float')
    foliations['G_z'] = np.cos(np.deg2rad(foliations["dip"].astype('float'))) * \
        foliations["polarity"].astype('float')


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def export_point(kmlpoints_list, template_fp, placemark_template_fp, filepath, filename):
    with open(placemark_template_fp) as f:
        placemark_template = f.readlines()

    for i, kmlp in enumerate(kmlpoints_list):
        with open(template_fp) as f:
            template = f.readlines()

        for ps in kmlp.point_sets:

            for p in ps.points:
                p.utm_to_latlong()

                # write coordinates
                placemark_template[5 - 1] = "<longitude>" + str(p.x) + "</longitude>\n"
                placemark_template[6 - 1] = "<latitude>" + str(p.y) + "</latitude>\n"
                placemark_template[7 - 1] = "<altitude>" + str(p.z) + "</altitude>\n"

                placemark_template[14 - 1] = "<coordinates>" + str(p.x) + "," + str(p.y) + "," + str(
                    p.z) + "</coordinates>\n"

                # append placemark to template
                template[-3:-3] = placemark_template

        with open(filepath + str(i) + "_" + filename + ".kml", 'w') as file:
            file.write("".join(template))


def plot_input_data_3d_scatter(interfaces, foliations):
    # **********************************************************************
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # **********************************************************************
    for fmt in interfaces["formation"].unique():  # loop over all unique formations
        interf = interfaces[interfaces["formation"] == fmt]  # select only current formation
        ax.scatter(interf["X"], interf["Y"], interf["Z"], alpha=0.85, s=35,
                   label=fmt)  # plot points of current formation

    # plot foliation data
    ax.scatter(foliations["X"], foliations["Y"], foliations["Z"], color="black", alpha=0.85, s=35,
               label="Foliation data")

    calculate_gradient(foliations)

    # **********************************************************************
    # The following code will add arrows to indicate the foliation values
    # **********************************************************************

    # get extent of plot to adjust vectors:
    ext_x, ext_y, ext_z = np.diff(ax.get_xlim3d()), np.diff(ax.get_ylim3d()), np.diff(ax.get_zlim3d())
    m = 1000

    # plot and arrow for each foliation value in the DataFrame
    for i, row in foliations.iterrows():
        a = Arrow3D([row["X"], row["X"] + row["G_x"] * m],
                    [row["Y"], row["Y"] + row["G_y"] * m * ext_y / ext_x],
                    [row["Z"], row["Z"] + row["G_z"] * m * ext_z / ext_x], mutation_scale=20,
                    lw=1, arrowstyle="-|>", color="k"
                    )
        ax.add_artist(a)  # add the arrow artist to the subplot axes

    # **********************************************************************
    # add plot legend and  labels
    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.suptitle("GoogleEarth picks")
    # **********************************************************************
    return ax


def clamp(x):
    return int(max(0, min(x, 255)))


def utm_to_latlong(x, y, zone=40):
    wgs = osr.SpatialReference()
    wgs.ImportFromEPSG(4326)
    if zone == 40:
        utm = osr.SpatialReference()
        utm.ImportFromEPSG(32640)
    else:
        raise AttributeError("Sorry, zone %d not yet implemented (check EPSG code and include in code!)" % zone)

    ct = osr.CoordinateTransformation(utm, wgs)
    x, y = ct.TransformPoint(x, y)[:2]
    return x, y


def gempy_export_points_to_kml(fp, geo_data, placemark_template_fp, template_fp, cmap):
    with open(placemark_template_fp) as f:
        placemark_template = f.readlines()

    formations = geo_data.get_formations()

    for fmt in formations:  # loop over all formations to create a file for each
        with open(template_fp) as f:  # start with a new template for each file
            template = f.readlines()

        f = geo_data.interfaces["formation"] == fmt  # filter

        # set colors
        rgb_c = np.array(cmap(np.unique(geo_data.interfaces[f]["formation number"]), bytes=True)[0])[:-1]
        hex_c = "ff{0:02x}{1:02x}{2:02x}".format(clamp(rgb_c[2]), clamp(rgb_c[1]), clamp(rgb_c[0]))
        template[17 - 1] = "<color>" + hex_c + "</color>\n"
        template[28 - 1] = "<color>" + hex_c + "</color>\n"

        for i, row in geo_data.interfaces[f].iterrows():
            x, y = utm_to_latlong(row["X"], row["Y"])
            placemark_template[3 - 1] = "<longitude>" + str(x) + "</longitude>\n"
            placemark_template[4 - 1] = "<latitude>" + str(y) + "</latitude>\n"
            placemark_template[5 - 1] = "<altitude>" + str(row["Z"]) + "</altitude>\n"

            placemark_template[12 - 1] = "<coordinates>" + str(x) + "," + str(y) + "," + str(
                row["Z"]) + "</coordinates>\n"
            placemark_template[14 - 1] = "</Placemark>\n"
            # append placemark to template
            template[-3:-3] = placemark_template

        with open(fp + fmt + ".kml", 'w') as file:
            file.write("".join(template))


def gempy_export_fol_to_kml(fp, geo_data,
                            placemark_template_fp,
                            template_fp):
    with open(placemark_template_fp) as f:
        placemark = f.readlines()
    with open(template_fp) as f:
        template = f.readlines()

    # loop over all foliation data points
    for i, row in geo_data.orientations.iterrows():
        x, y = utm_to_latlong(row["X"], row["Y"])
        placemark[3 - 1] = "<longitude>" + str(x) + "</longitude>\n"
        placemark[4 - 1] = "<latitude>" + str(y) + "</latitude>\n"
        placemark[5 - 1] = "<altitude>" + str(row["Z"]) + "</altitude>\n"
        placemark[12 - 1] = "<coordinates>" + str(x) + "," + str(y) + "," + str(row["Z"]) + "</coordinates>\n"

        heading = row["azimuth"]
        if heading + 180 > 360:
            heading -= 180
        else:
            heading += 180

        placemark[16 - 1] = "<heading>" + str(heading) + "</heading>"

        template[-3:-3] = placemark

    if ".kml" not in fp:
        fp = fp.join(".kml")

    with open(fp, 'w') as file:
        file.write("".join(template))


if __name__ == '__main__':
    print(test)

