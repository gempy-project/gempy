"""
1.2b: The Implicit coKriging class
==================================

"""

# %%
# Importing GemPy
import gempy as gp

# Importing auxiliary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('precision', 2)

# %%
# The description of the methods nomenclature remains the same as for the
# ``data.py`` module with the particularity that update is slitted in:
# 
# -  ``update_from``
# 
#    -  update current object with the rest of dependencies. This is
#       useful if you change an object and you want to update the fields
#       with the rest of the objects. E.g after a set\_surface\_points
# 
# -  ``update_to``
# 
#    -  update dataframes from the current object. This is useful if you
#       modify one of the model dependencies and you want to update all
#       the other dependencies
# 
# When we initialize a :class:`gempy.core.model.Project` class we create all the
# necessary objects already linked with each other.
# 

# %% 
model = gp.Project()

# %%
# As expected these dependencies are empty:
# 

# %% 
model.surface_points

# %% 
model.surfaces

# %% 
model.stack

# %%
# The pandas DataFrames are already configured properly to categories:
# 

# %% 
model._surfaces.df['series'], model._surfaces.df['surface']

# %%
# And additional data has everything pretty much empty:
# 

# %% 
model._additional_data

# %%
# Reading data
# ------------
# 
# Usually data will be imported from external files. GemPy uses
# ``pandas.read_table`` powerful functionality for that. The default
# format is XYZ surface\_name:
# 

# %% 
data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'

model.read_data(path_i=data_path + "/data/input_data/tut_chapter1/simple_fault_model_points.csv",
                path_o=data_path + "/data/input_data/tut_chapter1/simple_fault_model_orientations.csv")

# %% 
model.orientations

# %% 
model.map_stack_to_surfaces({"Fault_Series": ('Main_Fault', 'Silstone'),
                             "Strat_Series": ('Sandstone_2', 'Sandstone_1', 'Siltstone',
                                              'Shale',)}, )

# %% 
model.stack

# %% 
model.surfaces

# %% 
model.surface_points

# %% 
model._orientations.df.head()

# %%
# Next we need to categorize each surface into the right series. This will
# update all the Dataframes depending on ``Formations`` and ``Series`` to
# the right categories:
# 

# %% 
model.map_stack_to_surfaces({"Fault_Series": 'Main_Fault',
                             "Strat_Series": ('Sandstone_2', 'Siltstone',
                                              'Shale', 'Sandstone_1')})

# %% 
model._surfaces.df['series']

# %% 
model.surfaces

# %% 
gp.get_data(model, 'surface points')

# %% 
model.stack

# %%
# In the case of having faults we need to assign wich series are faults:
# 

# %% 
model.faults

# %% 
model.set_is_fault(['Fault_Series'])

# %% 
model.surface_points

# %%
# Again as we can see, as long we use the model methods, all the dependent
# objects change inplace accordingly. If for any reason you do not want
# this behaviour you can always use the individual methods of the objects
# (e.g. ``model.faults.set_is_fault``)
# 

# %% 
model.additional_data

# %%
# Setting grid
# ------------
# 
# So far we have worked on data that depends exclusively of input (i.e.
# sequential pile, surface_points, orientations, etc). With things like
# grid the idea is the same:
# 

# %% 
model.grid

# %% 
model.set_regular_grid([0, 10, 0, 10, 0, 10], [50, 50, 50])

# %%
# Getting data
# ------------
# 


# %%
# Alternatively we can access the dataframe by:
# 

# %% 
gp.get_data(model, 'surfaces')

# %%
# The class :class:`gempy.core.model.Model` works as the parent container of
# our project. Therefore the main step of any project is to create an
# instance of this class. In the official documentation we use normally
# ``geo_model`` (``geo_data`` in the past) as name of this instance.
# 
# When we instantiate a ``Model`` object we full data structure is
# created. By using ``gp.init_data`` and ``set_series`` we set the default
# values -- given the attributes -- to all of fields. Data is stored in
# pandas dataframes. With ``gp.get_data`` and the name of the data object
# it is possible to have access to the dataframes:
# 
# ``str``\ ['all', 'surface\_points', 'orientations', 'formations',
# 'series', 'faults', 'faults\_relations', additional data]
# 
# These dataframes are stored in specific objects. These objects contain
# the specific methods to manipulate them. You access these objects with
# the specific getter or as a attribute of ``Model``
#
