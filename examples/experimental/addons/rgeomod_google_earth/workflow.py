# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: sphinx
#       format_version: '1.1'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python [conda env:googlepicks]
#     language: python
#     name: conda-env-googlepicks-py
# ---

"""
# From Google Earth to a geological model

## Import
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# %matplotlib inline
import mplstereonet
from importlib import reload

import sys
import os
sys.path.append(r"..")  # append local path to access rgeomod module
import rgeomod

###############################################################################
# ## Import, process and visualize Google Earth data
#
# ### Load .kml files

folder_path = "../data/FW1/"

""
ks, ks_names, ks_bool = rgeomod.read_kml_files(folder_path)

###############################################################################
# *ks* contains the KmlPoints objects, containing the point data from the respective .kml files in the given directory:

ks

###############################################################################
# *ks_names* contains the filenames:

ks_names

###############################################################################
# *ks_bool* is a boolean array specifying which object contains dip values:

ks_bool

###############################################################################
# ### Load DTM to obtain elevation data and fit planes

geotiff_filepath = "../data/dome_sub_sub_utm.tif"

""
rgeomod.get_elevation_from_dtm(ks, geotiff_filepath)

""
rgeomod.fit_planes_to_points(ks)

###############################################################################
# ### Convert to data frames

interfaces, foliations = rgeomod.convert_to_df(ks, ks_names, ks_bool)

""
interfaces.tail()

""
foliations.tail()

###############################################################################
# ### Visualize points in 3D

fig = plt.figure(figsize=(14,10))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(interfaces["X"], interfaces["Y"], interfaces["Z"], color="red", alpha=0.85, s=35, label="Interface data")
ax.scatter(foliations["X"], foliations["Y"], foliations["Z"], color="black", alpha=0.85, s=35, label="Foliation data")

ax.legend()
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
fig.suptitle("GoogleEarth picks")

###############################################################################
# # Visualize dip and azimuth 

###############################################################################
# Plot histograms of the extracted dip and dip direction data:

fig = plt.figure(figsize=(10,4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.hist(foliations["dip"], 18)
ax1.set_xlabel("Dips")
ax1.set_ylabel("Counts")

ax2.hist(foliations["azimuth"], 18, color='r');
ax2.set_xlabel("Dip Directions")
ax2.set_ylabel("Counts");

###############################################################################
# ### Stereonet Plots

fig, ax = mplstereonet.subplots()
cax = ax.density_contourf(foliations["azimuth"], foliations["dip"], measurement='poles', cmap='viridis', alpha=0.75)
fig.colorbar(cax)
ax.pole(foliations["azimuth"], foliations["dip"], 'g^', markersize=4)
ax.grid(True, color="black", alpha=0.25)

plt.show()

""
fig = plt.figure()
ax = fig.add_subplot(111, projection='stereonet')
ax.pole(foliations["azimuth"], foliations["dip"], 'g^', markersize=4)
ax.plane(foliations["azimuth"], foliations["dip"], 'g-', linewidth=0.5, alpha=0.85)
ax.grid(True, color="black", alpha=0.25)

###############################################################################
# ## Save to .csv files

interfaces.to_csv("../data/gempy_interfaces.csv", index=False)
foliations.to_csv("../data/gempy_foliations.csv", index=False)

###############################################################################
# # GemPy

# These two lines are necessary only if gempy is not installed
sys.path.append("../../gempy/")
sys.path.append("../gempy/")

# Importing gempy
import gempy as gp
import warnings
warnings.filterwarnings('ignore')


""
geo_data=gp.create_data(extent=[612000, 622000, 2472000, 2480000, -1000, 1000], resolution=[50, 50, 50], 
                        path_f = "../data/gempy_foliations.csv",
                        path_i = "../data/gempy_interfaces.csv")

""
###############################################################################
# geo_data.foliations = geo_data.foliations.drop(np.where(geo_data.foliations.duplicated("X").values==True)[0])

""
gp.set_series(geo_data, {"Default series": tuple(np.array(ks_names)[np.logical_not(ks_bool)])},
             order_formations = tuple(np.array(ks_names)[np.logical_not(ks_bool)]))

gp.set_order_formations(geo_data, np.array(ks_names)[np.logical_not(ks_bool)])

""
###############################################################################
# gp.set_series(geo_data, {"Default series": ("Unit_1", "Unit_2", "Unit_3", "Unit_4")},
#              order_formations = ["Unit_1", "Unit_2", "Unit_3", "Unit_4"])

###############################################################################
# ## Data visualization

###############################################################################
# ### 2D

gp.plot_data(geo_data, direction="z")

""
gp.plot_data(geo_data, direction="x")

""
gp.plot_data(geo_data, direction="y")

###############################################################################
# ### 3D (requires VTK)

gp.plot_data_3D(geo_data)

###############################################################################
# ## Computing the 3D model
#
# Instantiate interpolator

interp_data = gp.InterpolatorInput(geo_data, dtype="float32")

###############################################################################
# Compute

lith_block, fault_block = gp.compute_model(interp_data)

###############################################################################
# ## Model visualization

###############################################################################
# ### 2D Sections

gp.plot_section(geo_data, lith_block[0], 40, direction='y')

###############################################################################
# ### 3D Surfaces (requires VTK)

v_l, s_l = gp.get_surfaces(interp_data, potential_lith=lith_block[1])
gp.plot_surfaces_3D(geo_data, v_l, s_l)

###############################################################################
# ### Pseudo-3D point clouds

###############################################################################
# v_l, s_l = gp.get_surfaces(interp_data, potential_lith=lith_block[1])
#
# %matplotlib inline
# fig = plt.figure(figsize=(13,10))
# ax = fig.add_subplot(111, projection='3d')
#
# cs = ["blue", "red", "orange"]
# for i in range(3):
#     ax.scatter(v_l[i][:,0],v_l[i][:,1],v_l[i][:,2], c=cs[i], s=5)
#     
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")

###############################################################################
# # Map

import gdal
import skimage
import scipy

""
xs = np.linspace(geo_data.extent[0], geo_data.extent[1], geo_data.resolution[0])
ys = np.linspace(geo_data.extent[2], geo_data.extent[3], geo_data.resolution[1])
zs = np.linspace(geo_data.extent[4], geo_data.extent[5], geo_data.resolution[2])

X,Y = np.meshgrid(xs, ys)

""
lb = lith_block[0].reshape(geo_data.resolution)

""
raster = gdal.Open(geotiff_filepath)
dtm = raster.ReadAsArray()

""
dtm.shape

""
dtm.ravel()

""
dtm.ravel().shape

""


""
###############################################################################
# g = np.meshgrid(
#                 np.linspace(612000, 622000, 150, dtype="int32"),
#                 np.linspace(2472000, 2480000, 150, dtype="int32"),
#                 dtm.ravel().astype("int32"), indexing="ij"
#     )
#
# grid1 = np.vstack(map(np.ravel, g)).T.astype("int32")

""


""


###############################################################################
# ## Pixelmap

dtm_resized = skimage.transform.resize(dtm, (50,50), preserve_range=True)


""
def htvi(dtm, zs):
    dz = (zs[-1] - zs[0]) / len(zs)
    dtm_v = (dtm - zs[0]) / dz
    return dtm_v.astype(int)


""
vdtm = plt.imshow(htvi(dtm_resized, zs))
plt.title("Topography indices")
plt.colorbar(vdtm)

""
indices = htvi(dtm_resized, zs)

geomap = np.zeros((50, 50))
for x in range(50):
    for y in range(50):
        geomap[x,y] = lb[x,y,indices[x,y]]

""
plt.imshow(geomap.T, origin="lower", cmap=gp.colors.cmap, norm=gp.colors.norm)
plt.title("Geological map")

""
from copy import copy

""
geoblock = copy(lb)
for x in range(50):
    for y in range(50):
        z = indices[x,y]
        geoblock[x,y,z:] = -1

""
#plt.imshow(geoblock[25,:,:].T, origin="lower", cmap=palette)
geoblock_m = np.ma.masked_where(geoblock < 0, geoblock)
geoblock_m_inv = np.ma.masked_where(geoblock > 0, geoblock)
plt.imshow(geoblock_m_inv[25,:,:].T, origin="lower", cmap="gray")
plt.imshow(geoblock_m[25,:,:].T, origin="lower", cmap=gp.colors.cmap, norm=gp.colors.norm)

""


""
# %matplotlib inline
fig = plt.figure(figsize=(12,8))
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, dtm_resized, cmap="viridis",
                       linewidth=0, antialiased=False)

ax.set_zlim(-1000,1000)

""

