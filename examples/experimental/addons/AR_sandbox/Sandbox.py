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
## GemPy Sandbox Addon Tutorial

"""

###############################################################################
# The Sandbox addon to GemPyconsists in general of four classes:
#     - Kinect class: handles the aquisition of depth data 
#     - Beamer class: handles the frame composition
#     - Calibration class: methods and variables to manage the relation of the sandbox, the beamer and the kinect sensor
#     - Model Class: stores the gempy model 
# It is possible to have multiple instances of each class and you can construct a sandbox with multiple beamers and more than one kinect sensor. This tutorial however covers only the most simple case of a sandbox with a single kinect and beamer. Lets start with importing the required dependancies:

import sys, os
sys.path.append("../../..")

import gempy as gp
import gempy.addons.sandbox as sb 

###############################################################################
# ### Initialize Kinect
# To use the Kinect and the Sandbox addon you need The Freenect Drivers to be installed with Python wrappers (see here for Instructions:https://github.com/OpenKinect/libfreenect/tree/master/wrappers/python ) 
# If you do not have a kinect connected you can use the Argument `dummy=True`  for testing. `Kinect.get_frame()` and `Kinect.get_filtered_frame()` will return a synthetic depth frame, other functions may not work

kinect=sb.Kinect(dummy=True)
#kinect=sb.Kinect()

""
import freenect
import matplotlib.pyplot as plt

""
#image = freenect.sync_get_video()[0]


""
# n=5
# name="test_frameb"+str(n)+".png"
# plt.imsave(name, image)


""
d=kinect.get_filtered_frame()
print(d)

###############################################################################
# ## Initialize beamer 
# create a beamer instance and set the correct native resolution of your beamer. Starting the stream will open a new window in your browser that shows the output and refreshs it in 100ms intervalls. In Chrome use Cmd+Shift+F to hide the Browser bar.

beamer=sb.Beamer()

""
beamer.resolution=(1920,1080)

""
beamer.start_stream()

###############################################################################
# ## Calibration
# A calibration instance is automatically created with the beamer instance. Adjust the Values in the IpyWidget until your beamer window is in alignment with the topography in the sandbox. Calibration can be saved and loaded with `calibration.save()` and `calibration.load()`

calibration=beamer.calibration

""
calibration.create()
#beamer.calibrate() #alternative commands, does the same.


""
#calibration.save(calibration_file="calibration.dat")

""
calibration.load(calibration_file="sandbox_VRlab.dat")

""
kinect.get_frame()

""
import matplotlib.pyplot as plt
plt.imshow(kinect.depth[50:210, 110:378], cmap='viridis')
plt.colorbar()

""
calibration.calibration_data['y_lim']

""
#import numpy as np
#np.save("sand_velocity2", kinect.depth[50:210, 110:378])

###############################################################################
# ## Create a model
# The sandbox can visualize any kind of Gempy model. Check out the other tutorials or this you6ube video [link] to learn how to create your own model.
# We use the model from Chapter 1:

geo_data = gp.create_data([0,2000,0,2000,0,2000],[30,30,30], 
                          path_o = os.pardir+"/../input_data/tut_chapter1/simple_fault_model_orientations.csv", # importing orientation (foliation) data
                          path_i = os.pardir+"/../input_data/tut_chapter1/simple_fault_model_points.csv") # importing point-positional interface data

gp.set_series(geo_data, {"Fault_Series":'Main_Fault', 
                         "Strat_Series": ('Sandstone_2','Siltstone',
                                          'Shale', 'Sandstone_1')},
                       order_series = ["Fault_Series", 'Strat_Series'],
                       order_formations=['Main_Fault', 
                                         'Sandstone_2','Siltstone',
                                         'Shale', 'Sandstone_1',
                                         ], verbose=0) 

interp_data = gp.InterpolatorData(geo_data, 
                                  output='geology', compile_theano=True,
                                  theano_optimizer='fast_compile',
                                  verbose=[])

###############################################################################
# ## Prepare the model for the sandbox
# The Interpolator object we just created is what defines the model in the sandbox.
# First we create a Model instance out of it, then we perform some steps to prepare the Model for the Sandbox.

import threading
    
lock = threading.Lock()

""
model=sb.Model(interp_data, lock=lock)


""
model.calculate_scales()
model.create_empty_depth_grid()

#model.setup(start_stream=False) #conveniently, the steps above are performed automatically when calling this function

###############################################################################
# ## Run Sandbox
# Our Sandbox is now set up and ready to go. 
# You can start the runloop with a simple command:


t = threading.Thread(target=sb.run_model, args=(model,), daemon=None)

""
t.start()

""
t.isAlive()

""
vtk_plot = gp.plotting.vtkPlot(geo_data, lock=lock)

""
vtk_plot.close()

""
vtk_plot.plot_surfaces_3D_real_time(interp_data)

""
model.stop_threat=True

""
vtk_plot.observe_df(geo_data, 'interfaces')

""
vtk_plot.plot_surfaces_3D_real_time(interp_data)

""
vtk_plot.resume()

""
vtk_plot.close()

""
t = threading.Thread(target=sb.run_model, args=(model,), daemon=None)

""
sb.run_model(model)

###############################################################################
# by default the depth data is filtered and smoothed to get clearer and less noisy layer boundaries.  If you need more control or want to play around you can also define your own run loops:
#     

while True:
    depth = kin.get_frame()
    model.update_grid(depth)
    model.render_frame(outfile="current_frame.png")
    beamer.show(input="current_frame.png")

""
while True:
    depth = kin.get_filtered_frame(n_frames=10, sigma_gauss=2)
    model.update_grid(depth)
    model.render_frame(outfile="current_frame.png")
    beamer.show(input="current_frame.png")
