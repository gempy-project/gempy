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
## Chapter 1.9a: Fault relations
***
A first example to show, which relation between faults and series can be represenented by GemPy. A set of thre horizontally deposited units is offset by a single fault. Two younger units deposited on top of these units, filling the space created by the offset (e.g. faulting occured synsedimentary).

Let's start as always by importing the necessary dependencies:
"""

# These two lines are necessary only if gempy is not installed
import sys, os
sys.path.append("../../..")

# Importing gempy
import gempy as gp

# Embedding matplotlib figures into the notebooks
# #%matplotlib inline

# Aux imports
import numpy as np
import pandas as pn
import matplotlib
import theano
import qgrid

# #%matplotlib widget

###############################################################################
# We import a model from an existing folder.

geo_model = gp.load_model('Tutorial_ch1-9a_Fault_relations', path= '../../data/gempy_models', recompile=True)

""
geo_model.faults.faults_relations_df

""
geo_model.faults

""
geo_model.surfaces

""
gp.compute_model(geo_model, compute_mesh=False)

""
geo_model.solutions.lith_block

""
geo_model.solutions.block_matrix[0]

""
gp.activate_interactive_df(geo_model)

###############################################################################
# It is important to get df with get to update the models sinde the `activate_interactive` method is called

###############################################################################
# If necessary, functions to display input data:

#geo_model.qi.get('orientations')

""
#geo_model.qi.get('surface_points')

###############################################################################
# Displaying the order of the different surfaces and series:

geo_model.qi.get('surfaces')

""
geo_model.qi.get('series')

""
geo_model.qi.get('faults')

""
geo_model.qi.get('faults_relations')

###############################################################################
# Displaying the input data:

gp.plot.plot_data(geo_model, direction='y')

""
gp.plot.plot_section(geo_model, 25, show_data=True)

###############################################################################
# Save model if changes were made:

geo_model.save_model('Tutorial_ch1-9a_Fault_relations')

""

