"""
"Model 1 - Horizontal stratigraphic
===========================

This example doesn't do much, it just makes a simple plot
"""


# %%
# This is the most simpel model of horizontally stacked layers. We start by importing the necessary dependencies:

# Importing GemPy
import gempy as gp
import os

os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu"

# %%

# Creating the model by importing the input data and displaying it:


# %%
geo_data = gp.create_data([0,1000,0,1000,0,1000],resolution=[50,50,50],
                        path_o = os.getcwd()+"/../notebooks/data/input_data/jan_models/model1_orientations.csv",
                        path_i = os.getcwd()+"/../notebooks/data/input_data/jan_models/model1_surface_points.csv")

# %%
"""
Setting and ordering the units and series:
"""

# %%
gp.map_series_to_surfaces(geo_data, {"Strat_Series": ('rock2','rock1'),"Basement_Series":('basement')})

# %%
gp.plot.plot_data(geo_data, direction='y')

# %%
"""
Calculating the model:
"""

# %%
interp_data = gp.set_interpolator(geo_data, compile_theano=True,
                                        theano_optimizer='fast_compile')

# %%
interp_data.theano_graph.number_of_points_per_surface_T.get_value()

# %%
sol = gp.compute_model(geo_data)

# %%
"""
Displaying the result in x and y direction:
"""

# %%
gp.plot.plot_section(geo_data, cell_number=25,
                         direction='x', show_data=True)

# %%
# sphinx_gallery_thumbnail_number = 3
gp.plot.plot_section(geo_data, cell_number=25,
                    direction='y', show_data=True)