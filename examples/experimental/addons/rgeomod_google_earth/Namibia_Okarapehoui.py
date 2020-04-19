# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: sphinx
#       format_version: '1.1'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python [conda env:remote-geomod]
#     language: python
#     name: conda-env-remote-geomod-py
# ---

"""
# Geological model of Okarapehoui/ Nigeria

Here another test case to create a geological model directly from GoogleEarth - completely work in progress, so beware...

However, this notebook is focused on describing the method to *new* regions of interest - in addition to the Jebel Madar Dome example of the Paper.

<img src="../figures/Namibia_Okauapehuri_perspective_view.png" width="600"/>


"""

import sys, warnings, numpy as np, matplotlib.pyplot as plt, gdal
sys.path.append(r"..")
import rgeomod
warnings.filterwarnings('ignore')
sys.path.append("../../gempy")
import gempy as gp
import mplstereonet
from mpl_toolkits.mplot3d import Axes3D
# %matplotlib inline

###############################################################################
# ## Determine model extent and obtain DTM
#
# As a first step, we determine the model extent this time directly from a path element in GoogleEarth.
#
# Here is an example for this model:
#
# <img src="../figures/Namibia_Okauapehuri_extent.png" width="600"/>
#
# As before, simply save this path to a `.kml` file. We now extract the coordinates from this path and determine min/max extent in each direction. To load the path:
#

extent = rgeomod.KmlPoints()

""
extent.read_kml("../data/Okauapehuri_model_extent.kml")

###############################################################################
# Here are the points of this line:

extent.point_sets[0].points

###############################################################################
# Get minimum and maximum values:

extent.point_sets[0].minmax()
print(extent.point_sets[0].min)
print(extent.point_sets[0].max)

###############################################################################
# Round to 3 digits:

xmin, ymin = np.round(extent.point_sets[0].min, decimals=3)
xmax, ymax = np.round(extent.point_sets[0].max, decimals=3)

""
xmin, ymin, xmax, ymax

###############################################################################
# Save points in point object (for simpler use below):

ex_min = rgeomod.Point(x=xmin, y=ymin, type='latlong')
ex_max = rgeomod.Point(x=xmax, y=ymax, type='latlong')

###############################################################################
# ## Get elevation data
#
# As we described in the paper and the previous notebooks, DTM elevation data can be obtained from a variety of online sources and repositories. Luckily, there is also a nice Python project which makes downloading this data quite simple.
#
# The project is available on: https://github.com/bopen/elevation
#
# <div class="alert alert-warning">
# **To do:** Include package and methods directly into rgeomod!
# </div>
#
# After standard installation (pip install elevation), it can be imported and used to download the DTM data for our region of interest:

import elevation

""
# clip the SRTM1 30m DEM of Rome and save it to Rome-DEM.tif
elevation.clip(bounds=(xmin, ymin, xmax, ymax), 
               output='/Users/flow/git/remote-geomod/projects/Nambia_Okauapehoui/Namibia.tif')

# clean up stale temporary files and fix the cache in the event of a server error
elevation.clean()

""
# load the digital elevation model
geotiff_filepath = "../projects/Nambia_Okauapehoui/Namibia.tif"
raster = gdal.Open(geotiff_filepath)
dtm = raster.ReadAsArray()
plt.figure(figsize=(14,12))
dtmp = plt.imshow(dtm, origin='upper', cmap="viridis");
plt.title("Digital elevation model");
plt.colorbar(dtmp, label="Elevation [m]", orientation='horizontal');
plt.savefig("../projects/Nambia_Okauapehoui/DTM_Namibia.pdf")

###############################################################################
# ## Process input data
#
# We now follow the same steps as in the introduction notebook: load picked points and process orientation data:

folder_path = "../projects/Nambia_Okauapehoui/"
point_sets, formation_names, ps_bool, fn = rgeomod.read_kml_files(folder_path)

""
formation_names

""
# Add elevation values:
# set the path to the geotiff file:
geotiff_filepath = "../projects/Nambia_Okauapehoui/Namibia.tif"

rgeomod.get_elevation_from_dtm(point_sets, geotiff_filepath)

###############################################################################
# ### Fit planes to points
#
# Note: this step requires suitable UTM transformation codes! If you get an error, please see:
#
# http://spatialreference.org/ref/epsg/

rgeomod.fit_planes_to_points(point_sets)

""
interfaces, foliations = rgeomod.convert_to_df(point_sets, formation_names, fn, ps_bool)
rgeomod.plot_input_data_3d_scatter(interfaces, foliations)

""
fig = plt.figure()
ax = fig.add_subplot(111, projection='stereonet')
ax.pole(foliations["azimuth"], foliations["dip"], 'g^', markersize=4)
ax.plane(foliations["azimuth"], foliations["dip"], 'g-', linewidth=0.75, alpha=0.85)
ax.grid(True, color="black", alpha=0.25)

###############################################################################
# ### Save data for further use

interfaces.to_csv("../projects/Nambia_Okauapehoui/gempy_interfaces.csv", index=False)
foliations.to_csv("../projects/Nambia_Okauapehoui/gempy_foliations.csv", index=False)

###############################################################################
# ## Set up model and load data
#
# Just as before, we now set up the model and load back the foliation information.
#
# (NOTE as todo: load data directly in memory - to avoid IO overhead)
#
# We need to work on a projection now, so we use the model extent given by the UTM coordinates.

# convert extent points to UTM:
ex_min.latlong_to_utm()
ex_max.latlong_to_utm()
ex_min.x, ex_max.x, ex_min.y, ex_max.y

""
# get maximum of elevation and round up to next 100:
z_max = np.ceil((np.max(dtm)/100))*100
# vertical extent: simply 2000 meters here:
z_min = z_max - 2000.

""
geo_data=gp.create_data(extent=[ex_min.x, ex_max.x, ex_min.y, ex_max.y, z_min, z_max], 
                        resolution=[50, 50, 50],
                        path_f = "../projects/Nambia_Okauapehoui/gempy_foliations.csv",
                        path_i = "../projects/Nambia_Okauapehoui/gempy_interfaces.csv")

""
formation_order = ["Unit2", "Unit1"]
gp.set_series(geo_data, {"Default series": formation_order},
             order_formations = formation_order)
# gp.plot_data(geo_data, direction="z")

""
gp.plot_data(geo_data, direction="z")

""
# try:
#     gp.plot_data_3D(geo_data)
# except NameError:
#     print("3-D visualization library vtk not installed.")

""
geo_data.interfaces.head()

""
interp_data = gp.InterpolatorInput(geo_data, dtype="float64", u_grade=[3])

""
lith_block, fault_block = gp.compute_model(interp_data)
print("3-D geological model calculated.")

""
gp.plot_section(geo_data, lith_block[0], 25, direction='y', plot_data=False)
plt.savefig("../projects/Nambia_Okauapehoui/cross_section_NS_25.pdf", bbox_inches="tight")

""
gp.plot_section(geo_data, lith_block[0], 25, direction='x', plot_data=False)
plt.savefig("../data/cross_section_EW_25.pdf", bbox_inches="tight")

""
vertices, simplices = gp.get_surfaces(interp_data, potential_lith=lith_block[1], step_size=2)
try:
    gp.plot_surfaces_3D(geo_data, vertices, simplices)
except NameError:
    print("3-D visualization library vtk not installed.")

###############################################################################
# ## Create geological map
#
# We now repeat the same steps as in the example notebook to create the geological map from the generated model. We already have the DTM loaded (see beginning of this notebook) and now simply have to calculate the model intersection:

# get resolution of dtm
dtm.shape

""
# convert the dtm to a gempy-suitable raveled grid
points = rgeomod.convert_dtm_to_gempy_grid(raster, dtm)

# Note: points have to be converted to UTM - *very* inefficient implementation for now...
points_utm = []
for p in points:
    p_tmp = rgeomod.Point(x=p[0], y=p[1], z=p[2], type='latlong')
    p_tmp.latlong_to_utm()
    points_utm.append((p_tmp.x, p_tmp.y, p_tmp.z))

points_utm = np.array(points_utm)

""
geo_data_geomap = gp.create_data(extent=[ex_min.x, ex_max.x, ex_min.y, ex_max.y, z_min, z_max], 
                        resolution=[dtm.shape[1], dtm.shape[0], 1],
                        path_f = "../projects/Nambia_Okauapehoui/gempy_foliations.csv",
                        path_i = "../projects/Nambia_Okauapehoui/gempy_interfaces.csv")

gp.set_series(geo_data_geomap, {"Default series": formation_order},
             order_formations = formation_order, verbose=0)

#NReplace grid points with DTM grid points:
geo_data.grid.grid = points_utm

""
interp_data_geomap = gp.InterpolatorInput(geo_data, dtype="float64")
lith_block, fault_block = gp.compute_model(interp_data_geomap)

""
gp.plot_section(geo_data_geomap, lith_block[0], 0, direction='z', plot_data=False)
plt.title("Geological map");
plt.savefig("../projects/Nambia_Okauapehoui/geological_map.png")

###############################################################################
# ## Interpretation
#
# Ok, so as a first experiment, the good news: the workflow can be adapted to completely new regions on GoogleEarth! However, the generated model (and therefore the map) obviously has some issues:
#
# - The map does not seem to follow the picked points! Is this a problem in the orientation (again)?
# - Issue of resolution of underlying DTM? Idea: always generate resolution map as well! Maybe even diretly in geotiff? E.g.: make every 10th line white/ checkerboard pattern?
# - Problem of orientation data - sometimes very difficult to pick... as an extension: include "dip points" where orientation is directly included in GoogleEarth?
# - Probably the underlying problem: not enough topography, combined with too low spatial resolution?
#
# Many things to check... but first approach still quite promising!
#
# <img src="../figures/Namibia_Okauapehuri_map_draft.png" width="600"/>
#
#

geo_map = lith_block[0].copy().reshape((dtm.shape[1],dtm.shape[0]))
geo_map = geo_map.astype('int16')  # change to int for later use

""
rgeomod.export_geotiff("../projects/Nambia_Okauapehoui/geomap.tif", geo_map, gp.colors.cmap, geotiff_filepath)

""
# export picked points for vis in GoogleEarth:
t = "../rgeomod/templates/ge_template_raw_interf.xml"
pt = "../rgeomod/templates/ge_placemark_template_interf.xml"
rgeomod.gempy_export_points_to_kml("../projects/Nambia_Okauapehoui/", geo_data, pt, t, gp.colors.cmap)


""
t = "../rgeomod/templates/ge_template_raw_fol.xml"
pt = "../rgeomod/templates/ge_placemark_template_fol.xml"
rgeomod.gempy_export_fol_to_kml("../projects/Nambia_Okauapehoui/dips.kml", geo_data, pt, t)

""

