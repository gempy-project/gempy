"""
The Model class
===============

"""

# These two lines are necessary only if GemPy is not installed
import sys, os
sys.path.append("../../..")

# Importing GemPy
import gempy as gp

# Embedding matplotlib figures in the notebooks
# %matplotlib inline

# Importing auxiliary libraries
import numpy as np
import pandas as pn
import matplotlib.pyplot as plt




######################################################################
# The description of the methods nomenclature remains the same as for the
# ``data.py`` module with the particularity that update is splitted in:
# 
# -  *update_from*
# 
#    -  update current object with the rest of dependencies. This is
#       useful if you change an object and you want to update the fields
#       with the rest of the objects. E.g after a set_surface_points
# 
# -  *update_to*
# 
#    -  update dataframes from the current object. This is useful if you
#       modify one of the model dependencies and you want to update all
#       the other dependecies
# 
# When we initialize a ``Model`` class we create all the necessary objects
# already entangled with each other.
# 

model = gp.Model()


######################################################################
# As expected these dependencies are empty:
# 

model.surface_points

model.surfaces

model.series


######################################################################
# The pandas DataFrames are already configurated properly to categories:
# 

model.surfaces.df['series'], model.surfaces.df['surface'] 


######################################################################
# And additional data has everything pretty much empty:
# 

model.additional_data


######################################################################
# Reading data
# ------------
# 
# Usually data will be imported from external files. GemPy uses
# ``pandas.read_table`` powerful functionality for that. The default
# format is XYZ surface_name:
# 

data_path= '../..'

model.read_data(path_i=data_path+"/data/input_data/tut_chapter1/simple_fault_model_points.csv",
                path_o=data_path+"/data/input_data/tut_chapter1/simple_fault_model_orientations.csv")

model.orientations

a = model.surfaces.df['series'].cat
a.set_categories(model.series.df.index)

model.map_series_to_surfaces({"Fault_Series":('Main_Fault', 'Silstone'), 
                               "Strat_Series": ( 'Sandstone_2', 'Sandstone_1', 'Siltstone',
                                             'Shale', )}, )

model.series

model.surfaces

model.surface_points.df.head()

model.orientations.df.head()


######################################################################
# Next we need to categorize each surface into the right series. This will
# update all the Dataframes depending on ``Formations`` and ``Series`` to
# the right categories:
# 

model.map_series_to_surfaces({"Fault_Series":'Main_Fault', 
                                "Strat_Series": ('Sandstone_2','Siltstone',
                                                 'Shale', 'Sandstone_1')})

model.surfaces.df['series']

model.surfaces

model.surface_points.df.head()

model.series


######################################################################
# In the case of having faults we need to assign wich series are faults:
# 

model.faults

model.set_is_fault(['Fault_Series'])

model.surface_points.df.head()


######################################################################
# Again as we can see, as long we use the model methods all the dependent
# objects change inplace accordingly. If for any reason you do not want
# this behaviour you can always use the individual methods of the objects
# (e.g. ``model.faults.set_is_fault``)
# 

model.additional_data


######################################################################
# Setting grid
# ------------
# 
# So far we have worked on data that depends exclusively of input
# (i.e.�sequeantial pile, surface_points, orientations, etc). With things
# like grid the idea is the same:
# 

model.grid.values

model.set_regular_grid([0,10,0,10,0,10], [50,50,50])


######################################################################
# Getting data
# ------------
# 


######################################################################
# Alternatively we can access the dataframe by:
# 

gp.get_data(model, 'surfaces')


######################################################################
# The class ``gempy.core.model.Model`` works as the parent container of
# our project. Therefore the main step of any project is to create an
# instance of this class. In the official documentation we use normally
# geo_model (geo_data in the past) as name of this instance.
# 
# When we instiantiate a ``Model`` object we full data structure is
# created. By using ``gp.init_data`` and ``set_series`` we set the default
# values � given the attributes � to all of fields. Data is stored in
# pandas dataframes. With ``gp.get_data`` and the name of the data object
# it is possible to have access to the dataframes:
# 
# ``str``\ [�all�, �surface_points�, �orientations�, �formations�,
# �series�, �faults�, �faults_relations�, additional data]
# 
# These dataframes are stored in specific objects. These objects contain
# the specific methods to manipulate them. You access these objects with
# the spectific getter or as a attribute of ``Model``
# 