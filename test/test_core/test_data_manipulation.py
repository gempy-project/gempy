#!/usr/bin/env python
# coding: utf-8

# ## Chapter 1.2: Data Structure and Manipulation
# ***
# In the previous tutorial we saw how we can create a model by calling a few lines of code from imported data. However modelling tends to be an iterative process. Here we will explore the tools that `GemPy` with the help of `pandas` offers to modify the input data of a model.
#
# There is 5 main  funtion "types" in GemPy:
#
# - *create*:
#     - create new objects
#     - return the objects
#
# - *set*
#     - set given values **inplace**
#
# - *update*
#     - update dataframe or other attribute from other object or many objects. Usually this object is not passed as argument (this is the main difference with map)
#
# - *map*
#     - update dataframe (so far mainly df) or other attribute from an object to another object.
#     - Completelly directed. One attribute/property is updated by another one.
#     - In general, we map_from so the method is in the mutated object.
#
# - *get*
#     - return an image of the object
#
# The intention is that a function/method that does not fall in any of these categories has a name (verb in principle) self explanatory.
#
#
# As always we start importing the usual packages and reading expample data:

# In[1]:


# from IPython.display import IFrame
# IFrame("https://atlas.mindmup.com/2018/11/ca2c3230ddc511e887555f7d8bb30b4d/gempy_mind_map/index.html",
#       width=1000, height=1000)


# In[2]:


# These two lines are necessary only if GemPy is not installed
import sys, os
sys.path.append("../..")

# Importing GemPy
import gempy as gp

# Embedding matplotlib figures in the notebooks
get_ipython().run_line_magic('matplotlib', 'inline')

# Importing auxiliary libraries
import numpy as np
import pandas as pn
import matplotlib.pyplot as plt


# ## Series
#
# Series is the object that contains the properties associated with each independent scalar field. Right now it is simply the order of the series (which is infered by the index order). But in the future will be add the unconformity relation or perhaps the type of interpolator
#
# Series and Faults classes are quite entagled since fauls are a type of series

# In[3]:


faults = gp.Faults()
series = gp.Series(faults)
series.df


# We can modify the series bt using `set_series_index`:

# In[4]:


series.set_series_index(['foo', 'foo2', 'foo5', 'foo7'])
series


# The index of series are pandas categories. These provides quite handy backend functionality (see pandas.Categorical).

# In[5]:


series.df.index


# For adding new series:

# In[6]:


series.add_series('foo3')
series


# Delete series

# In[7]:


series.delete_series('foo3')
series


# Rename series:

# In[8]:


series.rename_series({'foo':'boo'})
series


# Reorder series:

# In[9]:


series.reorder_series(['foo2', 'boo', 'foo7', 'foo5'])
series


# ### Faults
#
# The *df faults* is used to charectirize which *mathematical series* behave as fault and if mentioned faults are finite or infinite. Both df should get updated automatically as we modify the series object linked to the fault object (by passing it wehn a Series object is created).

# In[10]:


faults


# Finally we have the *faults relations df* which captures which *mathematical series* a given fault offset in order to reproduce complex faulting networks

# In[11]:


faults.faults_relations_df


# We can use `set_is_fault` to choose which of our series are faults:

# In[12]:


faults.set_is_fault(['boo'])


# Similar thing for the fault relations:

# In[13]:


fr = np.zeros((4, 4))
fr[2, 2] = True
faults.set_fault_relation(fr)


# Now if we change the series df and we update the series already defined will conserve their values while the new ones will be set to false:

# In[14]:


series.add_series('foo20')


# In[15]:


series


# In[16]:


faults


# In[17]:


faults.faults_relations_df


# When we add new series the values switch  to NaN. We will be careful not having any nan in the DataFrames or we will raise errors down the line.

# In[18]:


faults.set_is_fault()


# In[19]:


faults.set_fault_relation()


# ### Formations:
#
# The *df* formation contain three properties. *id* refers to the order of the formation on the sequential pile, i.e. the strict order of computation. *values* on the other hand is the final value that each voxel will have after discretization. This may be useful for example in the case we want to map a specific geophysical property (such as density) to a given unity. By default both are the same since to discretize lithological units the value is arbitrary.

# #### From an empty df
#
# The Formation class needs to have an associate series object. This will limit the name of the series since they are a pandas.Category

# In[20]:


f = gp.Formations(series)


# In[93]:


f.df.reset_index().index +1


# We can set any number of formations by passing a list with the names. By default they will take the name or the first series.

# In[22]:


f.set_formation_names(['foo', 'foo2', 'foo5'])


# In[23]:


series


# We can add new formations:

# In[24]:


f.add_formation(['feeeee'])
f


# The column formation is also a pandas.Categories. This will be important for the Data clases (Interfaces and Orientations)

# In[25]:


f.df['formation']


# ### Set values
#
# To set the values we do it with the following method

# In[26]:


f.set_formation_values_pro([2,2,2,5])


# In[27]:


f


# #### Set values with a given name:
#
# We can give specific names to the properties (i.e. density)

# In[28]:


f.add_formation_values_pro([[2,2,2,6], [2,2,1,8]], ['val_foo', 'val2_foo'])


# In[29]:


f


# ### Delete formations values
#
# To delete a full propery:

# In[30]:


f.delete_formation_values(['val_foo', 'value_0'])


# #### One of the formations must be set be the basement:

# In[31]:


f.set_basement()
f


# #### Set formation values
#
# We can also use set values instead adding. This will delete the previous properties and add the new one

# In[32]:


f.set_formation_values_pro([[2,2,2,6], [2,2,1,8]], ['val_foo', 'val2_foo'])
f


# The last property is the correspondant series that each formation belong to. `series` and `formation` are pandas categories. To get a overview of what this mean check https://pandas.pydata.org/pandas-docs/stable/categorical.html.

# In[33]:


f.df['series']


# In[34]:


f.df['formation']


# ### Map series to formation

# To map a series to a formation we can do it by passing a dict:

# In[35]:


f


# In[36]:


series


# If a series does not exist in the `Series` object, we rise a warning and we set those formations to nans

# In[37]:


d =  {"foo7":'foo', "booX": ('foo2','foo5', 'fee')}


# In[38]:


f.map_series(d)


# In[39]:


f.map_series({"foo7":'foo', "boo": ('foo2','foo5', 'fee')})


# In[40]:


f


# An advantage of categories is that they are order so no we can tidy the df by series and formation

# In[41]:


f.df.sort_values(by='series', inplace=True)


# If we change the basement:

# In[42]:


f.set_basement('foo5')


# Only one formation can be the basement:

# In[43]:


f


# ### Modify formation name

# In[44]:


f.rename_formations({'foo2':'lala'})


# In[45]:


f


# In[46]:


f.df.loc[2, 'val_foo'] = 22


# In[47]:


f


# In[48]:


f.update_sequential_pile()


# In[49]:


f.sequential_pile.figure


# # Data
# #### Interfaces
# These two DataFrames (df from now on) will contain the individual information of each point at an interface or orientation. Some properties of this table are mapped from the *df* below.

# In[50]:


interfaces = gp.Interfaces(f)
#orientations = gp.Orientations()


# In[51]:


interfaces


# In[52]:


f


# In[53]:


interfaces.set_interfaces(pn.DataFrame(np.random.rand(6,3)), ['foo', 'foo5', 'lala', 'foo5', 'lala', 'feeeee'])


# In[54]:


interfaces


# In[55]:


interfaces.map_data_from_formations(f, 'series')
interfaces


# In[56]:


interfaces.map_data_from_formations(f, 'id')
interfaces


# In[57]:


series


# In[58]:


interfaces.map_data_from_series(series, 'order_series')
interfaces


# In[59]:


interfaces.sort_table()
interfaces


# In[60]:


faults


# ### Orientations

# In[61]:


orientations = gp.Orientations(f)


# In[62]:


orientations


# ### Set values passing pole vectors:

# In[63]:


orientations.set_orientations(np.random.rand(6,3)*10,
                            np.random.rand(6,3),
                            surface=['foo', 'foo5', 'lala', 'foo5', 'lala', 'feeeee'])


# In[64]:


orientations


# ### Set values pasing orientation data: azimuth, dip, pole (dip direction)

# In[65]:


orientations.set_orientations(np.random.rand(6,3)*10,
                            orientation = np.random.rand(6,3)*20,
                            surface=['foo', 'foo5', 'lala', 'foo5', 'lala', 'feeeee'])


# In[66]:


orientations


# ### Mapping data from the other df

# In[67]:


orientations.map_data_from_formations(f, 'series')
orientations


# In[68]:


orientations.map_data_from_formations(f, 'id')
orientations


# In[69]:


orientations.map_data_from_series(series, 'order_series')
orientations


# In[70]:


orientations.set_annotations()


# ### Grid

# In[71]:


grid = gp.Grid()
grid.set_regular_grid([0,10,0,10,0,10], [50,50,50])


# In[72]:


grid.values


# #### Rescaling Data

# In[73]:


rescaling = gp.RescaledData(interfaces, orientations, grid)


# In[74]:


interfaces


# In[75]:


orientations


# ### Additional Data

# In[76]:


ad = gp.AdditionalData(interfaces, orientations, grid, faults, f, rescaling)


# In[77]:


ad


# In[78]:


ad.structure_data


# In[79]:


ad.options


# In[80]:


ad.options.df


# In[81]:


ad.options.df.dtypes


# In[82]:


ad.kriging_data


# In[83]:


ad.rescaling_data


# ### Interpolator

# In[84]:


interp = gp.Interpolator(interfaces, orientations, grid, f, faults, ad)


# In[85]:


interp.compile_th_fn()