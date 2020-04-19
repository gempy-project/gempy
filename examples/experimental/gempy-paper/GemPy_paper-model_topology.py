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
# GemPy Paper Code: Model and Topology

In this notebook you will be able to see and run the code utilized to create the figures of the paper *GemPy - an open-source library for implicit geological modeling and uncertainty quantification*
"""

# Importing dependencies

# These two lines are necessary only if gempy is not installed
import sys, os
sys.path.append("../..")

import gempy as gp
# %matplotlib inline

# Aux imports

import numpy as np
import pandas as pn
import matplotlib.pyplot as plt


""
# Funtion to plot labels of the input data in latex
def annotate_plot(frame, label_col, x, y, **kwargs):
    """
    Annotate the plot of a given DataFrame using one of its columns

    Should be called right after a DataFrame or series plot method,
    before telling matplotlib to show the plot.

    Parameters
    ----------
    frame : pandas.DataFrame

    plot_col : str
        The string identifying the column of frame that was plotted

    label_col : str
        The string identifying the column of frame to be used as label

    kwargs:
        Other key-word args that should be passed to plt.annotate

    Returns
    -------
    None

    Notes
    -----
    After calling this function you should call plt.show() to get the
    results. This function only adds the annotations, it doesn't show
    them.
    """
    import matplotlib.pyplot as plt  # Make sure we have pyplot as plt

    for label, x, y in zip(frame[label_col], frame[x], frame[y]):
        plt.annotate(label, xy=(x + 0.1, y + 0.15), **)


###############################################################################
# ## Building a geological model
#
# First we import the raw data and define model parameters such as resolution or extent

geo_model = gp.create_model('GemPy-Paper-1')

gp.init_data(geo_model, [0,20,0,10,-10,0],[100,10,100],
            path_o = "input_data/paper_Orientations.csv",
            path_i = "input_data/paper_Points.csv")

# Example of method to add extra points directly in Python 
#geo_data.add_interface(X=10, Y=4, Z=-7, formation='fault1')

###############################################################################
# Defining all different series that form the most complex model. In the paper you can find figures with different combination of these series to examplify the possible types of topolodies supported in GemPy

# %matplotlib inline
gp.map_series_to_surfaces(geo_model,
                          {'fault_serie1': 'fault1',
                           'younger_serie' : 'Unconformity', 
                           'older_serie': ('Layer1', 'Layer2')})

#fig=plt.gcf()
# fig.savefig('doc/figs/fault_pile.pdf')

""
geo_model.surfaces

""
geo_model.surface_points

###############################################################################
# The next cell show how specific series can be selected

#geo_data = gp.select_series(geo_data, ['older_serie'])

###############################################################################
# Visualizing the final data

gp.plot.plot_data(geo_model)

###############################################################################
# Compiling the theano funciton

gp.set_interpolation_data(geo_model,
                          verbose=[], compile_theano=True)

""
geo_model.surfaces

""
geo_model.set_is_fault('Default series')

###############################################################################
# Computing the model

sol = gp.compute_model(geo_model)

""
geo_model.surface_points

""
# Plotting
gp.plot.plot_section(geo_model, cell_number=5, show_data=True)

geo_model.surface_points.update_annotations()
geo_model.orientations.update_annotations()
annotate_plot(gp.get_data(geo_model), 'annotations', 'X', 'Z', size = 20)

# plt.savefig("model_2.pdf")

""
gp.plot.plot_section(geo_model, direction='x', cell_number=16, show_data=True)


""
gp.plot.plot_section(geo_model, block=geo_model.solutions.block_matrix[0],
                     cell_number=5, show_data=True)


""
gp.plot.plot_scalar_field(geo_model, 5, cmap="viridis", N = 10, plot_data=True, series=1)
annotate_plot(gp.get_data(geo_model), 'annotations', 'X', 'Z', size = 20)

# plt.savefig("doc/figs/scalar_field_simple.pdf")

###############################################################################
# In 3D using vtk:

gp.plot.plot_3D(geo_model)

###############################################################################
# ## Topology

from gempy.assets import topology as tp
G, c, *_ = tp.compute_topology(geo_model)
gp.plot.plot_section(geo_model, 2)
gp.plot.plot_topology(geo_model, G, c)

###############################################################################
# ### Save model

gp.save_model(geo_model)

""

