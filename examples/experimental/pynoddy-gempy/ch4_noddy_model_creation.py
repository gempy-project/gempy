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

import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

import sys
sys.path.append("../../../pynoddy/")

import pynoddy
import pynoddy.history
import pynoddy.output
import pynoddy.events
import pynoddy.experiment

import importlib

""
history = "ch4_model.his"
output = "ch4_model_out"

""
cs = 25
extent = (3000, 200, 2000)

""
nm = pynoddy.history.NoddyHistory()
# set model extent
nm.set_extent(extent[0],extent[1],extent[2])
# set model origin x,y,z
nm.set_origin(0., 0., extent[2])
# set model cube size
nm.change_cube_size(cs)

# stratigraphic event
strati_options = {
    'num_layers' : 5,
    'layer_names' : ['5','4','3','2','1'],
    'layer_thickness' : [350,200,200,200,200]
}
nm.add_event('stratigraphy', strati_options)

# fold event
fold_options = {
    "name": "fold1",
    "pos": (500,0,1000),
    "wavelength": 4000.,
    "amplitude": 400
}
nm.add_event("fold", fold_options)

# write history file
nm.write_history(history)

""
nm

""
ex = pynoddy.experiment.Experiment(history)
ex.plot_section()

""
pynoddy.compute_model(history, 'model')

""
ex.plot_section(direction="x")

""
ex.basename = 'model'
ex.load_geology()

""


""


""


""


""
ex.get_drillhole_data(5,5, resolution=2000/500)

""
ex.origin_z, ex.extent_z, 2000/500, ex.extent_x

""
sys.path.append("../..")
import gempy as gp

""
# initialize geo_data object
geo_data = gp.create_data([0, extent[0], 
                           0, extent[1], 
                           0, extent[2]],
                          resolution=[int(extent[0]/cs), 
                                      int(extent[1]/cs), 
                                      int(extent[2]/cs)])

""
x = [250,750,500]
y = [0,100,200]
layers = [2,3,4,5]

for l in layers:
    i_df = ex.export_interfaces_gempy(x,y, layer=l, group_id="l"+str(l)+"_a")
    gp.set_interfaces(geo_data, i_df, append=True)

""
x = [2750,2250,2500]
y = [0,100,200]
layers = [2,3,4,5]

for l in layers:
    i_df = ex.export_interfaces_gempy(x,y, layer=l, group_id="l"+str(l)+"_b")
    gp.set_interfaces(geo_data, i_df, append=True)

""
i_df.dtypes, geo_data.interfaces.dtypes

""
# %matplotlib notebook
gp.plotting.plot_data(geo_data)

""
geo_data.interfaces[['X', 'Y', 'Z']] =  geo_data.interfaces[['X', 'Y', 'Z']].astype('float')

""
geo_data.update_df()

""
gp.get_data(geo_data, 'interfaces')

""
geo_data.interfaces.dtypes

""
indices = np.array([[0, 19, 18],
                    [22,23,24]
                   ])

""
gp.set_orientation_from_interfaces(geo_data, indices)

""
geo_data.orientations[(geo_data.orientations['G_x'] > 0).values * (geo_data.orientations['G_y'] < 0).values]

""
gp.set_orientation_from_interfaces(geo_data, [22,23,24])

""
np.rad2deg(np.arctan(np.deg2rad(0.51)/np.deg2rad(-0.014)))

""
geo_data.calculate_orientations()

""
geo_data.calculate_gradient()
gp.get_data(geo_data, 'orientations')

""


""
gp.get_data(geo_data, 'orientations')


""
def get_orientation(normal):
    """Get orientation (dip, azimuth, polarity ) for points in all point set"""
    #    if "normal" not in dir(self):
    #        self.plane_fit()

    # calculate dip
    dip = np.arccos(normal[2]) / np.pi * 180.

    # calculate dip direction
    # +/+
    if normal[0] >= 0 and normal[1] > 0:
        dip_direction = np.arctan(normal[0] / normal[1]) / np.pi * 180.
    # border cases where arctan not defined:
    elif normal[0] > 0 and normal[1] == 0:
        dip_direction = 90
    elif normal[0] < 0 and normal[1] == 0:
        dip_direction = 270
    # +-/-
    elif normal[1] < 0:
        dip_direction = 180 + np.arctan(normal[0] / normal[1]) / np.pi * 180.
    # -/-
    elif normal[0] < 0 and normal[1] >= 0:
        dip_direction = 360 + np.arctan(normal[0] / normal[1]) / np.pi * 180.
    # if dip is just straight up vertical
    elif normal[0] == 0 and normal[1] == 0:
        dip_direction = 0

    if -90 < dip < 90:
        polarity = 1
    else:
        polarity = -1

    return dip, dip_direction, polarity


""
get_orientation(geo_data.orientations[['G_x', 'G_y', 'G_z']].values)

""
geo_data.orientations[['G_x', 'G_y', 'G_z']].values

""


""


""
geo_data.create_orientation_from_interfaces([0, 19, 18])

""
gp.get_data(geo_data, 'orientations')

""
group_ids = ["l2_a", "l2_b"]
group_obj = []
for gid in group_ids:
    group_obj.append(gp.DataManagement.DataPlane(geo_data, 
                                                 gid, 
                                                 "interf_to_fol"))
for go in group_obj:
    go.set_fol()

""
gp.plotting.plot_data_3D(geo_data)

""
geo_data.set_formation_number()
geo_data.order_table()

""
# %matplotlib inline
gp.plot_data(geo_data, direction='y')

""
gp.plot_data(geo_data)

""
interp_data = gp.InterpolatorInput(geo_data, u_grade=[3])

""
sol_example = gp.compute_model(interp_data)

""
gp.plot_section(geo_data, sol_example[0,0,:], 1, plot_data = True)

""
geo_data.import_data_csv

""
geo_data.interfaces.to_csv("../Tutorial/data/tutorial_ch4_interfaces")

""
geo_data.foliations.to_csv("../Tutorial/data/tutorial_ch4_foliations")

""

