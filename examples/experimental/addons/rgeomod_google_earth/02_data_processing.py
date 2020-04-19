# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: sphinx
#       format_version: '1.1'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

"""
### Step 2
# Process coordinate points and add altitude values
"""

###############################################################################
# ##### Import of Python libraries

import sys, mplstereonet
sys.path.append(r"..")  # append local path to access rgeomod module
import rgeomod
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# %matplotlib inline

###############################################################################
# ### Step 2b: Load and transform coordinate data

###############################################################################
# <div class="alert alert-info">
# **Your task**: Set the **folder_path** variable to the folder where your exported GoogleEarth data is located and execute the cell to load the point sets stored in the *.kml* files.
# </div>

folder_path = "./data/FW2/"
point_sets, formation_names, ps_bool, fn = rgeomod.read_kml_files(folder_path)

###############################################################################
# *point_sets* contains the point set objects, containing the point data from the respective .kml files in the given directory:

point_sets

###############################################################################
# *formation_names* contains the formation names extracted from the filenames:

formation_names

###############################################################################
# *ps_bool* is a boolean array specifying which object contains dip values:

ps_bool

###############################################################################
# ### Step 2c: Add elevation values

###############################################################################
# <div class="alert alert-info">
# **Your task**: Run the following cell to extract the elevation data for our point sets from the digital elevation model.
# </div>

# set the path to the geotiff file:
geotiff_filepath = "./data/dome_sub_sub_utm.tif"

rgeomod.get_elevation_from_dtm(point_sets, geotiff_filepath)

###############################################################################
# ### Step 2d: Fit plane to points and determine orientations

###############################################################################
# <div class="alert alert-info">
# Running the following cell will fit planes to the foliation point sets:
# </div>

rgeomod.fit_planes_to_points(point_sets)

###############################################################################
# ### Step 2e: Visualize data

###############################################################################
# ### Convert data into readable DataFrames
#
# <div class="alert alert-info">
# Now we convert the interface and fitted foliation data stored in the point sets to a more intuitive data format: **Data Frames**, which simply represent data tables. 
# </div>

interfaces, orientations = rgeomod.convert_to_df(point_sets, formation_names, fn, ps_bool)

###############################################################################
# To get a quick view of the data, we can (for example) get a list of the last 5 entries using the `.tail()` function:

orientations.tail()

###############################################################################
# <div class="alert alert-info">
# Now that we have our data in a convenient format (Data Frames), we can use visualization techniques to better analyze our data. In the following we will make use of three different visualizations:
# <br>
#
# <ul>
#     <li>Pseudo-3D visualization of the data points;</li>
#     <li>Create histograms to visualize the distribution of dip angles and dip directions;</li>
#     <li>Plot the orientation dip data in stereoplots.</li>
# </ul>
#
# </div>
#
# #### 3D Point Cloud

rgeomod.plot_input_data_3d_scatter(interfaces, orientations)

###############################################################################
# #### Histograms

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
# #### Stereonet plots

fig, ax = mplstereonet.subplots()
cax = ax.density_contourf(foliations["azimuth"], foliations["dip"], measurement='poles', cmap='viridis', alpha=0.75)
fig.colorbar(cax)
ax.pole(foliations["azimuth"], foliations["dip"], 'g^', markersize=4)
ax.grid(True, color="black", alpha=0.25)

""
fig = plt.figure()
ax = fig.add_subplot(111, projection='stereonet')
ax.pole(foliations["azimuth"], foliations["dip"], 'g^', markersize=4)
ax.plane(foliations["azimuth"], foliations["dip"], 'g-', linewidth=0.75, alpha=0.85)
ax.grid(True, color="black", alpha=0.25)

###############################################################################
# # Step 4: Save the data
#
# <div class="alert alert-info">
# Now that we successfully added height values, fit orientation data and got a more intuitive understanding our picked data through visualization, we can save it as *.csv* files for storage and later use for 3D geomodeling.
# </div>

interfaces.to_csv("./data/gempy_interfaces.csv", index=False)
foliations.to_csv("./data/gempy_foliations.csv", index=False)

###############################################################################
# ### Export as GoogleEarth points

template_fp = "./data/ge_point_template.xml"
placemark_template_fp = "./data/ge_placemark_template.xml"
