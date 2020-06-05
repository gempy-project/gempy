"""
1.6: 2D Visualization.
======================
"""

# %%

# Importing GemPy
import gempy as gp

# Importing auxiliary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(1515)
pd.set_option('precision', 2)

# %%
# Model interpolation
# ~~~~~~~~~~~~~~~~~~~
# 

# %% 
# Data Preparation
data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'

geo_data = gp.create_data('viz_2d', [0, 1000, 0, 1000, 0, 1000], resolution=[10, 10, 10],
                          path_o=data_path + "/data/input_data/jan_models/model5_orientations.csv",
                          path_i=data_path + "/data/input_data/jan_models/model5_surface_points.csv")

# %% 
gp.plot_2d(geo_data)

# %% 
geo_data.set_topography(d_z=(500, 1000))

# %% 
section_dict = {'section1': ([0, 0], [1000, 1000], [100, 80]),
                'section2': ([800, 0], [800, 1000], [150, 100]),
                'section3': ([50, 200], [100, 500], [200, 150])}

# %% 
geo_data.set_section_grid(section_dict)
gp.plot.plot_section_traces(geo_data)

# %% 
geo_data.grid.sections

# %% 
gp.set_interpolator(geo_data, theano_optimizer='fast_compile')

# %% 
gp.map_stack_to_surfaces(geo_data, {"Fault_Series": 'fault',
                                    "Strat_Series": ('rock2', 'rock1')})

geo_data.set_is_fault(['Fault_Series'])

# %% 
geo_data.get_active_grids()

# %% 
gp.compute_model(geo_data)

# %% 
# new plotting api
gp.plot_2d(geo_data, section_names=['section1'])

# %%
# or

# %%
gp.plot.plot_2d(geo_data, section_names=['section1'])

# %%
# Plot 2d: Object oriented:
# ------------------------
# 


# %%
# One plot
# ^^^^^^^^
# 

# %% 
p = gp.plot_2d(geo_data, section_names=[], direction=None, show=False)
p.fig.show()


# %%
p = gp.plot_2d(geo_data, section_names=[], direction=None, show=False)
# -----new code------

sec_name = 'section1'
s1 = p.add_section(sec_name)
p.plot_data(s1, sec_name, projection_distance=200)
p.fig.show()

# %%
p = gp.plot_2d(geo_data, section_names=[], direction=None, show=False)
sec_name = 'section1'
s1 = p.add_section(sec_name)
# -----new code------

p.plot_data(s1, sec_name, projection_distance=200)
p.plot_contacts(s1, sec_name)
p.fig.show()

# %%
p = gp.plot_2d(geo_data, section_names=[], direction=None, show=False)
sec_name = 'section1'
s1 = p.add_section(sec_name)
p.plot_data(s1, sec_name, projection_distance=200)
p.plot_contacts(s1, sec_name)
# -----new code------

p.plot_lith(s1, sec_name)
p.plot_topography(s1, sec_name)
p.fig.show()

# %%
# Several plots
# ^^^^^^^^^^^^^
# 

# %% 

sec_name = 'section1'
sec_name_2 = 'section3'

p2 = gp.plot_2d(geo_data, n_axis=3, figsize=(15, 15),  # General fig options
                section_names=[sec_name, 'topography'], cell_number=[3],  # Defining the sections
                show_data=False, show_lith=False, show_scalar=False, show_boundaries=False)

# %% 

# Create the section. This loacte the axes and give the right
# aspect ratio and labels

p2 = gp.plot_2d(geo_data, n_axis=3, figsize=(15, 15),  # General fig options
                section_names=[sec_name, 'topography'], cell_number=[3],  # Defining the sections
                show_data=False, show_lith=False, show_scalar=False, show_boundaries=False,
                show=False)
# -----new code------

s1 = p2.add_section(sec_name_2, ax_pos=224)
p2.fig.show()

# %% 
# Axes 0

p2 = gp.plot_2d(geo_data, n_axis=3, figsize=(15, 15),  # General fig options
                section_names=[sec_name, 'topography'], cell_number=[3],  # Defining the sections
                show_data=False, show_lith=False, show_scalar=False, show_boundaries=False,
                show=False)
s1 = p2.add_section(sec_name_2, ax_pos=224)
# -----new code------

p2.plot_contacts(s1, sec_name_2)
p2.plot_lith(s1, sec_name_2)
p2.plot_data(s1, sec_name_2, projection_distance=200)
p2.plot_topography(s1, sec_name_2)
p2.fig.show()

# %%
# Axes 1

# sphinx_gallery_thumbnail_number = 12
p2 = gp.plot_2d(geo_data, n_axis=3, figsize=(15, 15),  # General fig options
                section_names=[sec_name, 'topography'], cell_number=[3],  # Defining the sections
                show_data=False, show_lith=False, show_scalar=False, show_boundaries=False,
                show=False)
s1 = p2.add_section(sec_name_2, ax_pos=224)
p2.plot_contacts(s1, sec_name_2)
p2.plot_lith(s1, sec_name_2)
p2.plot_data(s1, sec_name_2, projection_distance=200)
p2.plot_topography(s1, sec_name_2)
# -----new code------

p2.plot_contacts(p2.axes[0], cell_number=3)
p2.plot_scalar_field(p2.axes[0], cell_number=3, series_n=1)
p2.fig.show()


# %%
# Axes2

p2 = gp.plot_2d(geo_data, n_axis=3, figsize=(15, 15),  # General fig options
                section_names=[sec_name, 'topography'], cell_number=[3],  # Defining the sections
                show_data=False, show_lith=False, show_scalar=False, show_boundaries=False,
                show=False)
s1 = p2.add_section(sec_name_2, ax_pos=224)
p2.plot_contacts(s1, sec_name_2)
p2.plot_lith(s1, sec_name_2)
p2.plot_data(s1, sec_name_2, projection_distance=200)
p2.plot_topography(s1, sec_name_2)
p2.plot_contacts(p2.axes[0], cell_number=3)
p2.plot_scalar_field(p2.axes[0], cell_number=3, series_n=1)
# -----new code------

p2.plot_lith(p2.axes[1], 'topography')
p2.plot_contacts(p2.axes[1], 'topography')
p2.fig.show()

# %%
# Plotting traces:
# ''''''''''''''''
# 

# %% 
p2.plot_section_traces(p2.axes[1])
p2.fig.show()

# %% 
gp.plot.plot_section_traces(geo_data)

# %%
# Plot API
# --------
# 


# %%
# If nothing is passed, a Plot2D object is created and therefore you are
# in the same situation as above:
# 

# %% 
p3 = gp.plot_2d(geo_data)

# %%
# Alternatively you can pass section\_names, cell\_numbers + direction or
# any combination of the above:
# 

# %% 
gp.plot_2d(geo_data, section_names=['topography'])

# %% 
gp.plot_2d(geo_data, section_names=['section1'])

# %% 
gp.plot_2d(geo_data, section_names=['section1', 'section2'])

# %% 
gp.plot_2d(geo_data, figsize=(15, 15), section_names=['section1', 'section2', 'topography'],
           cell_number='mid')
