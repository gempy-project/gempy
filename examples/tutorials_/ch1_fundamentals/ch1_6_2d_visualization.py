"""
1.6: 2D Visualization.
======================
"""

# %%

# Importing auxiliary libraries
import numpy as np
import os

# Importing GemPy
import gempy as gp
import gempy_viewer as gpv

np.random.seed(1515)

# %%
# Model interpolation
# ~~~~~~~~~~~~~~~~~~~
# 

# %% 
# Data Preparation
data_path = os.path.abspath('../../')

geo_data: gp.data.GeoModel = gp.create_geomodel(
    project_name='viz_2d',
    extent=[0, 1000, 0, 1000, 0, 1000],
    resolution=[10, 10, 10],
    number_octree_levels=4,
    importer_helper=gp.data.ImporterHelper(
        path_to_orientations=data_path + "/data/input_data/jan_models/model5_orientations.csv",
        path_to_surface_points=data_path + "/data/input_data/jan_models/model5_surface_points.csv",
    )
)

gp.set_topography_from_random(grid=geo_data.grid, d_z=np.array([500, 1000]))

# %% 
gpv.plot_2d(geo_data)

# %% 
section_dict = {'section1': ([0, 0], [1000, 1000], [100, 80]),
                'section2': ([800, 0], [800, 1000], [150, 100]),
                'section3': ([50, 200], [100, 500], [200, 150])}

# %% 
gp.set_section_grid(geo_data.grid, section_dict)
gpv.plot_section_traces(geo_data)

# %% 
geo_data.grid.sections

# %% 

# %% 
gp.map_stack_to_surfaces(
    gempy_model=geo_data,
    mapping_object={
        "Fault_Series": 'fault',
        "Strat_Series": ('rock2', 'rock1')
    }
)

gp.set_is_fault(
    frame=geo_data.structural_frame,
    fault_groups=['Fault_Series']
)

# %% 
geo_data.grid.active_grids

# %% 
gp.compute_model(geo_data)

# %% 
# new plotting api
gpv.plot_2d(geo_data, section_names=['section1'])

# %%
# or

# %%
gpv.plot_2d(geo_data, section_names=['section1'])

# %%
# Plot 2d: Object oriented:
# -------------------------
# 


# %%
# One plot
# ^^^^^^^^
# 

# %% 
p = gpv.plot_2d(geo_data, section_names=[], direction=None, show=False)
p.fig.show()

# %%
p = gpv.plot_2d(geo_data, section_names=[], direction=None, show=False)
# -----new code------

sec_name = 'section1'
s1 = p.add_section(sec_name)
p.plot_data(s1, sec_name, projection_distance=200)
p.fig.show()

# %%
p = gpv.plot_2d(geo_data, section_names=[], direction=None, show=False)
sec_name = 'section1'
s1 = p.add_section(sec_name)
# -----new code------

p.plot_data(s1, sec_name, projection_distance=200)
p.plot_contacts(s1, sec_name)
p.fig.show()

# %%
p = gpv.plot_2d(geo_data, section_names=[], direction=None, show=False)
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

p2 = gpv.plot_2d(
    geo_data,
    n_axis=3,
    figsize=(15, 15),  # General fig options
    section_names=[sec_name, 'topography'],
    cell_number=[3],  # Defining the sections
    show_data=False,
    show_lith=False,
    show_scalar=False,
    show_boundaries=False
)

# %% 

# Create the section. This loacte the axes and give the right
# aspect ratio and labels

p2 = gpv.plot_2d(
    geo_data,
    n_axis=3,
    figsize=(15, 15),  # General fig options
    section_names=[sec_name, 'topography'],
    cell_number=[3],  # Defining the sections
    show_data=False,
    show_lith=False,
    show_scalar=False,
    show_boundaries=False,
    show=False
)
# -----new code------

s1 = p2.add_section(sec_name_2, ax_pos=224)
p2.fig.show()

# %% 
# Axes 0

p2 = gpv.plot_2d(
    geo_data, n_axis=3, figsize=(15, 15),  # General fig options
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
p2 = gpv.plot_2d(
    geo_data,
    n_axis=3,
    figsize=(15, 15),  # General fig options
    section_names=[sec_name, 'topography'],
    cell_number=[3],  # Defining the sections
    show_data=False,
    show_lith=False,
    show_scalar=False,
    show_boundaries=False,
    show=False
)
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

p2 = gpv.plot_2d(
    geo_data,
    n_axis=3,
    figsize=(15, 15),  # General fig options
    section_names=[sec_name, 'topography'],
    cell_number=[3],  # Defining the sections
    show_data=False,
    show_lith=False,
    show_scalar=False,
    show_boundaries=False,
    show=False
)

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
gpv.plot_section_traces(geo_data)

# %%
# Plot API
# --------
# 


# %%
# If nothing is passed, a Plot2D object is created and therefore you are
# in the same situation as above:
# 

# %% 
p3 = gpv.plot_2d(geo_data)

# %%
# Alternatively you can pass section\_names, cell\_numbers + direction or
# any combination of the above:
# 

# %% 
gpv.plot_2d(geo_data, section_names=['topography'])

# %% 
gpv.plot_2d(geo_data, section_names=['section1'])

# %% 
gpv.plot_2d(geo_data, section_names=['section1', 'section2'])

# %% 
gpv.plot_2d(geo_data, figsize=(15, 15), section_names=['section1', 'section2', 'topography'], cell_number='mid')
