
# coding: utf-8

# # 3 - 3D Modeling with GemPy

# In[1]:

from matplotlib import use
use("Agg")

import sys
import numpy as np
# These two lines are necessary only if gempy is not installed
sys.path.append("../../gempy/")
sys.path.append("../../../remote-geomod/rgeomod")
sys.path.append("../gempy/")
sys.path.append("/home/miguel/PycharmProjects/remote-geomod/")
# Importing gempy
import gempy as gp
import pytest

#import gdal
#import rgeomod
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

gdal = pytest.importorskip("gdal")
rgeomod = pytest.importorskip("rgeomod")
input_path = os.path.dirname(__file__)+'/../input_data'


def TOUPDATE_rgeomod_integration(theano_f):
    geo_data=gp.create_data(extent=[612000, 622000, 2472000, 2480000, -1000, 1000],
                            resolution=[50, 50, 50],
                            path_f=input_path+"/gempy_foliations.csv",
                            path_i=input_path+"/gempy_interfaces.csv")



    formation_order = ["Unit4", "Unit3", "Unit2", "Unit1"]



    gp.set_series(geo_data, {"Default series": formation_order},
                 order_formations = formation_order, verbose=1)



    gp.plot_data(geo_data, direction="z")


    #interp_data = gp.InterpolatorData(geo_model, compile_theano=True)
    interp_data = theano_f
    interp_data.update_interpolator(geo_data)

    lith_block, fault_block = gp.compute_model(interp_data)
    print("3-D geological model calculated.")


    gp.plot_section(geo_data, lith_block[0], 25, direction='y', plot_data=False)
    #plt.savefig("../data/cross_section_NS_25.pdf", bbox_inches="tight")

    gp.plot_section(geo_data, lith_block[0], 25, direction='x', plot_data=False)
    #plt.savefig("../data/cross_section_EW_25.pdf", bbox_inches="tight")

    vertices, simplices = gp.get_surfaces(interp_data, potential_lith=lith_block[1], step_size=2)

    fig = plt.figure(figsize=(13,10))
    ax = fig.add_subplot(111, projection='3d')
    cs = ["lightblue", "pink", "lightgreen", "orange"]
    for i in range(4):
        surf = ax.plot_trisurf(vertices[i][:,0], vertices[i][:,1], vertices[i][:,2],
                               color=cs[i], linewidth=0, alpha=0.65, shade=False)
    #plt.savefig("../data/surfaces_3D.pdf", bbox_inches="tight")

    # try:
    #     gp.plot_surfaces_3D(geo_model, vertices, simplices)
    # except NameError:
    #     print("3-D visualization library vtk not installed.")

    # load the digital elevation model
    geotiff_filepath = input_path+"/dome_sub_sub_utm.tif"
    raster = gdal.Open(geotiff_filepath)
    dtm = raster.ReadAsArray()
    dtmp = plt.imshow(dtm, origin='upper', cmap="viridis");
    plt.title("Digital elevation model");
    plt.colorbar(dtmp, label="Elevation [m]");
    plt.savefig(input_path+"temp/DTM.pdf")

    # To be able to use gempy plot functionality we need to create a dummy geo_model object with the
    # resoluion we want. In this case resolution=[339, 271, 1]
    import copy
    geo_data_dummy = copy.deepcopy(geo_data)
    geo_data_dummy.resolution = [339, 271, 1]



    # convert the dtm to a gempy-suitable raveled grid
    points = rgeomod.convert_dtm_to_gempy_grid(raster, dtm)


    # Now we can use the function `compute_model_at` to get the lithology values at a specific location:

    # In[17]:


    # interp_data_geomap = gp.InterpolatorInput(geo_model, dtype="float64")
    lith_block, fault_block = gp.compute_model_at(points, interp_data)


    # <div class="alert alert-info">
    # **Your task:** Create a visual representation of the geological map in a 2-D plot (note: result is also again saved to the `../data`-folder):
    # </div>
    #
    # And here **the geological map**:

    # In[18]:


    gp.plot_section(geo_data_dummy, lith_block[0], 0, direction='z', plot_data=False)
    plt.title("Geological map");
    #plt.savefig("../geological_map.pdf")


    # ### Export the map for visualization in GoogleEarth

    # <div class="alert alert-info">
    # **Your task:** Execute the following code to export a GeoTiff of the generated geological map, as well as `kml`-files with your picked points inside the data folder. Open these files in GoogleEarth and inspect the generated map:
    # </div>
    #
    #
    # <div class="alert alert-warning">
    # **Note (1)**: Use the normal `File -> Open..` dialog in GoogleEarth to open the data - no need to use the `Import` method, as the GeoTiff contains the correct coordinates in the file.
    # </div>
    #
    #
    # <div class="alert alert-warning">
    # **Note (2)**: For a better interpretation of the generated map, use the transparency feature (directly after opening the map, or using `right-click -> Get Info` on the file).
    # </div>

    # In[19]:


    geo_map = lith_block[0].copy().reshape((339,271))
    geo_map = geo_map.astype('int16')  # change to int for later use


    # In[20]:


    rgeomod.export_geotiff(input_path+"temp/geomap.tif", geo_map, gp.plotting.colors.cmap, geotiff_filepath)


    # Export the interface data points:

    # In[21]:


    t = input_path+"/templates/ge_template_raw_interf.xml"
    pt = input_path+"/templates/ge_placemark_template_interf.xml"
    rgeomod.gempy_export_points_to_kml(input_path, geo_data, pt, t, gp.plotting.colors.cmap)


    # Export the foliation data:



    t = input_path+"/templates/ge_template_raw_fol.xml"
    pt = input_path+"/templates/ge_placemark_template_fol.xml"
    rgeomod.gempy_export_fol_to_kml(input_path+"temp/dips.kml", geo_data, pt, t)
