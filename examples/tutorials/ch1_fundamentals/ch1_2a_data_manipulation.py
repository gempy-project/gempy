"""
1.2: Data Structure and Manipulation
------------------------------------

"""
# Importing GemPy
import gempy as gp
import gempy

# Importing auxiliary libraries
import numpy as np
import pandas as pd
pd.set_option('precision', 2)

# %%
# Series
# ~~~~~~
# 
# Series is the object that contains the properties associated with each
# independent scalar field. Right now it is simply the order of the series
# (which is inferred by the index order). But in the future will be add the
# unconformity relation or perhaps the type of interpolator
# 
# Series and Faults classes are quite entangled since fault are a view of
# series
# 

# %% 
faults = gp.Faults()
series = gp.Series(faults)
series.df

# %%
# We can modify the series bt using ``set_series_index``:
# 

# %% 
series.set_series_index(['foo', 'foo2', 'foo5', 'foo7'])
series

# %%
# The index of series are pandas categories. These provides quite handy
# backend functionality (see pandas.Categorical).
# 

# %% 
series.df.index

# %%
# For adding new series:
# 

# %% 
series.add_series('foo3')
series

# %%
# Delete series
# 

# %% 
series.delete_series('foo3')
series

# %%
# Rename series:
# 

# %% 
series.rename_series({'foo': 'boo'})
series

# %%
# Reorder series:
# 

# %% 
series.reorder_series(['foo2', 'boo', 'foo7', 'foo5'])
series

# %%
# Faults
# ~~~~~~
# 
# The *df faults* is used to characterize which *mathematical series*
# behave as fault and if mentioned faults are finite or infinite. Both dataframes
# get updated automatically as we modify the series object linked
# to the fault object (by passing it when a Series object is created).
# 

# %% 
faults

# %%
# Finally we have the *faults relations df* which captures which
# *mathematical series* a given fault offset in order to reproduce complex
# faulting networks
# 

# %% 
faults.faults_relations_df

# %%
# We can use ``set_is_fault`` to choose which of our series are faults:
# 

# %% 
faults.set_is_fault(['boo'])

# %%
# Similar thing for the fault relations:
# 

# %% 
fr = np.zeros((4, 4))
fr[2, 2] = True
fr[1, 2] = True

faults.set_fault_relation(fr)

# %%
# Now if we change the series df and we update the series already defined
# will conserve their values while the new ones will be set to false:
# 

# %% 
series.add_series('foo20')

# %%
series

# %% 
faults

# %% 
faults.faults_relations_df

# %%
# When we add new series the values switch to NaN. We will be careful not
# having any NaNs in the DataFrames or we will raise errors down the line.
# 

# %% 
faults.set_is_fault()

# %% 
faults.set_fault_relation()

# %%
# Surfaces:
# ~~~~~~~~~
# 
# The *df surfaces* contains three properties. *id* refers to the order of
# the surfaces on the sequential pile, i.e. the strict order of
# computation. *values* on the other hand is the final value that each
# voxel will have after discretization. This may be useful for example in
# the case we want to map a specific geophysical property (such as
# density) to a given unit. By default both are the same since to
# discretize lithological units the value is arbitrary.
# 


# %%
# From an empty df
# ^^^^^^^^^^^^^^^^
# 
# The Surfaces class needs to have an associate series object. This will
# limit the name of the series since they are a ``pandas.Categorical``\ .
# 

# %% 
surfaces = gp.Surfaces(series)

# %%
# We can set any number of formations by passing a list with the names. By
# default they will take the name or the first series.
# 

# %% 
surfaces.set_surfaces_names(['foo', 'foo2', 'foo5'])

# %%
series

# %%
# We can add new formations:
# 

# %% 
surfaces.add_surface(['feeeee'])
surfaces

# %%
# The column formation is also a ''pandas.Categorical''\ . This will be important
# for the Data classes (surface\_points and Orientations)
# 

# %% 
surfaces.df['surface']

# %% 
surfaces

# %%
# Set values
# ~~~~~~~~~~
# 
# To set the values we do it with the following method
# 

# %% 
surfaces.set_surfaces_values([2, 2, 2, 5])

# %% 
surfaces

# %%
# Set values with a given name:
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# We can give specific names to the properties (i.e. density)
# 

# %% 
surfaces.add_surfaces_values([[2, 2, 2, 6], [2, 2, 1, 8]], ['val_foo', 'val2_foo'])

# %% 
surfaces

# %%
# Delete formations values
# ~~~~~~~~~~~~~~~~~~~~~~~~
# 
# To delete a full property:
# 

# %% 
surfaces.delete_surface_values(['val_foo', 'value_0'])

# %%
# One of the formations must be set be the basement:
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 

# %% 
surfaces.set_basement()
surfaces

# %%
# Set formation values
# ^^^^^^^^^^^^^^^^^^^^
# 
# We can also use ``set_surface_values`` instead adding. This will delete the previous
# properties and add the new one
# 

# %% 
surfaces.set_surfaces_values([[2, 2, 2, 6], [2, 2, 1, 8]], ['val_foo', 'val2_foo'])
surfaces

# %%
# The last property is the correspondant series that each formation belong
# to. ``series`` and ``formation`` are pandas categories. To get a
# overview of what this mean check
# https://pandas.pydata.org/pandas-docs/stable/categorical.html.
# 

# %% 
surfaces.df['series']

# %% 
surfaces.df['surface']

# %%
# Map series to formation
# ~~~~~~~~~~~~~~~~~~~~~~~
# 


# %%
# To map a series to a formation we can do it by passing a dict:
# 

# %% 
surfaces

# %% 
series

# %%
# If a series does not exist in the ``Series`` object, we rise a warning
# and we set those formations to nans
# 

# %% 
d = {"foo7": 'foo', "booX": ('foo2', 'foo5', 'fee')}

# %% 
surfaces.map_series(d)

# %% 
surfaces.map_series({"foo7": 'foo', "boo": ('foo2', 'foo5', 'fee')})

# %% 
surfaces

# %%
# An advantage of categories is that they are order so no we can tidy the
# df by series and formation
# 


# %%
# Modify surface name
# ~~~~~~~~~~~~~~~~~~~
# 

# %% 
surfaces.rename_surfaces({'foo2': 'lala'})

# %% 
surfaces

# %% 
surfaces.df.loc[2, 'val_foo'] = 22

# %% 
surfaces

# %%
# Modify surface color
# ~~~~~~~~~~~~~~~~~~~~
# 


# %%
# The surfaces DataFrame also contains a column for the color in which the
# surfaces are displayed. To change the color, call
# 

# %% 
surfaces.colors.change_colors()

# %%
# This allow to change the colors interactively. If you already know which
# colors you want to use, you can also update them with a dictionary
# mapping the surface name to a hex color string:
# 

# %% 
new_colors = {'foo': '#ff8000', 'foo5': '#4741be'}
surfaces.colors.change_colors(new_colors)

# %%
# Data
# ~~~~
# 
# surface\_points
# ^^^^^^^^^^^^^^^
# 
# These two DataFrames (*df* from now on) will contain the individual
# information of each point at an interface or orientation. Some
# properties of this table are mapped from the *df* below.
# 

# %% 
surface_points = gempy.core.data_modules.geometric_data.SurfacePoints(surfaces)

# %%
surface_points

# %% 
surface_points.set_surface_points(pd.DataFrame(np.random.rand(6, 3)),
                                  ['foo', 'foo5', 'lala', 'foo5', 'lala', 'feeeee'])

# %% 
surface_points

# %% 
surface_points.map_data_from_surfaces(surfaces, 'series')
surface_points

# %% 
surface_points.map_data_from_surfaces(surfaces, 'id')
surface_points

# %% 
series

# %% 
surface_points.map_data_from_series(series, 'order_series')
surface_points

# %% 
surface_points.sort_table()
surface_points

# %% 
faults

# %%
# Orientations
# ~~~~~~~~~~~~
# 

# %% 
orientations = gempy.core.data_modules.geometric_data.Orientations(surfaces)

# %% 
orientations

# %%
# Set values passing pole vectors:
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

# %% 
orientations.set_orientations(np.random.rand(6, 3) * 10,
                              np.random.rand(6, 3),
                              surface=['foo', 'foo5', 'lala', 'foo5',
                                       'lala', 'feeeee'])

# %% 
orientations

# %%
# Set values pasing orientation data: azimuth, dip, pole (dip direction)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

# %% 
orientations.set_orientations(np.random.rand(6, 3) * 10,
                              orientation=np.random.rand(6, 3) * 20,
                              surface=['foo', 'foo5', 'lala', 'foo5',
                                       'lala', 'feeeee'])

# %% 
orientations

# %%
# Mapping data from the other df
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

# %% 
orientations.map_data_from_surfaces(surfaces, 'series')
orientations

# %% 
orientations.map_data_from_surfaces(surfaces, 'id')
orientations

# %% 
orientations.map_data_from_series(series, 'order_series')
orientations

# %% 
orientations.update_annotations()

# %%
# Grid
# ~~~~
# 

# %% 
grid = gp.Grid()
grid.create_regular_grid([0, 10, 0, 10, 0, 10], [50, 50, 50])

# %% 
grid.values

# %%
# Rescaling Data
# ^^^^^^^^^^^^^^
# 

# %% 
rescaling = gempy.core.data_modules.geometric_data.RescaledData(
    surface_points, orientations, grid)

# %%
surface_points

# %% 
orientations

# %%
# Additional Data
# ~~~~~~~~~~~~~~~
# 

# %% 
ad = gp.AdditionalData(surface_points, orientations, grid, faults, surfaces, rescaling)

# %%
ad

# %% 
ad.structure_data

# %% 
ad.options

# %% 
ad.options.df

# %% 
ad.options.df.dtypes

# %% 
ad.kriging_data

# %% 
ad.rescaling_data

# %%
# Interpolator
# ~~~~~~~~~~~~
# 

# %%

faults.df['isFault'].values

# %% 
interp = gp.InterpolatorModel(surface_points, orientations, grid, surfaces, series, faults, ad)

# %% 
interp.compile_th_fn_geo()

# %% 
interp.print_theano_shared()
