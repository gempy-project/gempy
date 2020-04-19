# -*- coding: utf-8 -*-
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
## Transform 2019: Integrating Striplog and GemPy
"""

# ! pip install welly striplog

""
# Authors: M. de la Varga, Evan Bianco, Brian Burnham and Dieter Werthmüller

# These two lines are necessary only if GemPy is not installed
import sys, os
sys.path.append("../../")

# Importing GemPy
import gempy as gp


# Embedding matplotlib figures in the notebooks
# %matplotlib inline

# Importing auxiliary libraries
import numpy as np
import pandas as pn
import matplotlib.pyplot as plt

import welly
from welly import Location, Project
import glob
from striplog import Striplog, Legend, Decor


###############################################################################
# #### Creating striplog object
# ----

# get well header coordinates
well_heads = {'alpha': {'kb_coords':(0,0,0)},
              'beta': {'kb_coords':(10,10,0)}, 
              'gamma': {'kb_coords':(12,0,0)}, 
              'epsilon': {'kb_coords':(20,0,0)}}

""
# Reading tops file
topsfiles = glob.glob('../data/input_data/striplog_integration/*.tops')
topsfiles

""
# Creating striplog object
my_striplogs = []

for file in topsfiles:
    with open(file) as f:
        text = f.read()
        striplog = Striplog.from_csv(text=text)
        my_striplogs.append(striplog)
        
striplog_dict = {'alpha': my_striplogs[1],
          'beta': my_striplogs[2],  
          'gamma': my_striplogs[3], 
          'epsilon': my_striplogs[0]}
striplog_dict['alpha'][0]

""
# Plot striplog
f, a = plt.subplots(ncols=4, sharey=True)

for e, log in enumerate(striplog_dict.items()):
    log[1].plot(ax=a[e], legend=None)
f.tight_layout()


""
# Striplog to pandas df of bottoms
rows = []
for wellname in striplog_dict.keys():
    for i, interval in enumerate(striplog_dict[wellname]):
            surface_name = interval.primary.lith
            surface_base = interval.base.middle
            x,y = well_heads[wellname]['kb_coords'][:-1]
            series = 1
            rows.append([x, y, surface_base, surface_name, series, wellname])
column_names = ['X','Y','Z','surface', 'series','wellname']
df = pn.DataFrame(rows, columns=column_names)
df

###############################################################################
# #### GemPy model
# -----

# Create gempy model object
geo_model = gp.create_model('welly_integration')

extent = [-100, 300, -100, 200, -150, 0]
res = [60, 60, 60]

# Initializting model using the striplog df
gp.init_data(geo_model, extent, res, surface_points_df = df)

""
geo_model.surface_points.df.head()

""
geo_model.surfaces

""
dec_list =[]
for e, i in enumerate(striplog_dict['alpha']):
    dec_list.append(Decor({'_colour': geo_model.surfaces.df.loc[e, 'color'],
  'width': None,
  'component': i.primary,
  'hatch': None}))

""


""
# welly plot with gempy colors
# Create Decor list
dec_list =[]
for e, i in enumerate(striplog_dict['alpha']):
    dec_list.append(Decor({'_colour': geo_model.surfaces.df.loc[e, 'color'],
  'width': None,
  'component': i.primary,
  'hatch': None}))
    
# Create legend
legend = Legend(dec_list)
legend

""
# Plot striplogs:
# Plot striplog
f, a = plt.subplots(ncols=4, sharey=True)

for e, log in enumerate(striplog_dict.items()):
    log[1].plot(ax=a[e], legend=legend)
f.tight_layout()

""
# Modifying the coordinates to make more sense
geo_model.surface_points.df[['X', 'Y']] = geo_model.surface_points.df[['X', 'Y']] * 10 
geo_model.surface_points.df['Z'] *= -1

""
# Delete points of the basement surface since we are intepolating bottoms (that surface wont exit).
geo_model.delete_surface_points_basement()

""
# Adding an arbitrary orientation. Remember gempy need an orientation per series
geo_model.set_default_orientation()
geo_model.modify_orientations(0, X=-500)

""
gp.plot.plot_data(geo_model)

""
# vtk_obj = gp.plot.plot_3D(geo_model, silent=False)

""
gp.set_interpolation_data(geo_model)

""
gp.compute_model(geo_model)

""
# Plotting the interpolated model with the well logs below. Placing the wells in the same coordinate system need
# quite a bit of love yet.

p = gp.plot.plot_section(geo_model, 30,  show_data=True)
axs = p.fig.axes[0]
axis_to_data = axs.transAxes + axs.transData.inverted() 
data_to_axis = axis_to_data.inverted()

for e, log in enumerate(striplog_dict.items()):
    # X coordinates of heads, times 10 because we are rescaling all data. 
    X = (np.array(well_heads[log[0]]['kb_coords'])[[0]] * 10).astype('float')

    # y scale we use the model  height but it is quite arbitrary
    x, y, width, height = data_to_axis.transform((X[0], -230, 1, 1))
    p.fig.add_axes([x, y,0.03, 0.5])
    log[1].plot(ax=p.fig.axes[-1],width=10, legend=legend)





""
# vp = gp.plot.plot_3D(geo_model)

###############################################################################
# ## Pinch out model
# ----
# As we can see the 3D model generated above does not honor the forth well lets fix it. First lets add an unconformity: between the yellow and green layer:
#

geo_model.add_series('Unconformity')

###############################################################################
# Now we set the green layer in the second series

geo_model.map_series_to_surfaces({'Uncomformity':['brian', 'evan', 'dieter']})

###############################################################################
# Lastly we need to add a dummy orientation to the new series:

geo_model.add_orientations(-500, 0, -100, 'dieter', [0, 0 , 1])

""
geo_model.interpolator.theano_graph.nugget_effect_grad_T.get_value()

###############################################################################
# Now we can compute:

gp.compute_model(geo_model)

""
p = gp.plot.plot_section(geo_model, 30,  show_data=True)
axs = p.fig.axes[0]
axis_to_data = axs.transAxes + axs.transData.inverted() 
data_to_axis = axis_to_data.inverted()

for e, log in enumerate(striplog_dict.items()):
    # X coordinates of heads, times 10 because we are rescaling all data. 
    X = (np.array(well_heads[log[0]]['kb_coords'])[[0]] * 10).astype('float')

    # y scale we use the model  height but it is quite arbitrary
    x, y, width, height = data_to_axis.transform((X[0], -230, 1, 1))
    p.fig.add_axes([x, y,0.03, 0.5])
    log[1].plot(ax=p.fig.axes[-1],width=10, legend=legend)


###############################################################################
# Getting better but not quite there yet. Since the yellow does not show up in the last well the pinch out has to happen somewhere before so lets add an artifial point to get that shape:

geo_model.add_surface_points(200, 0, -75, 'evan');

""
gp.compute_model(geo_model)
p = gp.plot.plot_section(geo_model, 30,  show_data=True)
axs = p.fig.axes[0]
axis_to_data = axs.transAxes + axs.transData.inverted() 
data_to_axis = axis_to_data.inverted()

for e, log in enumerate(striplog_dict.items()):
    # X coordinates of heads, times 10 because we are rescaling all data. 
    X = (np.array(well_heads[log[0]]['kb_coords'])[[0]] * 10).astype('float')

    # y scale we use the model  height but it is quite arbitrary
    x, y, width, height = data_to_axis.transform((X[0], -230, 1, 1))
    p.fig.add_axes([x, y,0.03, 0.5])
    log[1].plot(ax=p.fig.axes[-1],width=10, legend=legend)


""
# gp.save_model(geo_model)
