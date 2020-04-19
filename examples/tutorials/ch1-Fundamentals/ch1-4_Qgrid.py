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
## Chapter 1.4:  Using Qgrid for interactive DataFrames
***

From its documentation :

> Qgrid is a Jupyter notebook widget which uses SlickGrid to render pandas DataFrames within a Jupyter notebook. This allows you to explore your DataFrames with intuitive scrolling, sorting, and filtering controls, as well as edit your DataFrames by double clicking cells.

In practice these library allows us to use `pandas.DataFrames` in Jupyter Notebooks as an excel table. Be aware that `Qgrid` requires of enabling nbextensions so it is important to test that the installation (https://github.com/quantopian/qgrid) was successful before trying to execute the rest of the notebook.

Let's create a bit of default data:
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
# - check pickle works
# - when we set an is fault change the BottomRelation

geo_model = gp.create_model('Tutorial_ch1-4_Qgrid')
gp.init_data(geo_model, [0, 1000, 0, 1000, -1000, 0], [50, 50, 50])

""
geo_model.set_default_surfaces()

""
geo_model.series.add_series(['foo'])
geo_model.update_from_series()
geo_model.series

###############################################################################
# ### Activating Qgrid
#
# Qgrid is only a gempy dependency. Therefore to use it, first we need to activate it in a given model by using:

gp.activate_interactive_df(geo_model)

###############################################################################
# This will create the interactive dataframes objects. This dataframes are tightly linked to the main dataframes of each data class and any change in there will be analogous to use the `DataMutation` methods explained in the Model tutorial. To access this interactive dataframes you can use the property or a getter:

###############################################################################
# #### Model options

geo_model.qi.qgrid_op

###############################################################################
# #### Series

geo_model.qi.get('series')


###############################################################################
# #### surfaces
#
# Notice that if we use the objects found in `Model().qi` updating a parameter of one object will update the related dataframes. For example changing the name of a series in the series dataframe above will modify the `Formatations.df`

geo_model.qi.qgrid_fo

###############################################################################
# #### Faults
#
# And the faults df

geo_model.qi.qgrid_fa

###############################################################################
# #### Faults relations

geo_model.qi.qgrid_fr

###############################################################################
# Remember we are always changing the main df as well!

geo_model.faults.faults_relations_df

###############################################################################
# ### Geometric Data

geo_model.qi.get('surface_points')

""
geo_model.qi.qgrid_fo

""
geo_model.qi.qgrid_se

###############################################################################
# ### Auxiliary data

geo_model.additional_data
