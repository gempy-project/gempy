"""
Ch1-10a_2D_Visualization.
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
import matplotlib.pyplot as plt

# %% 
from gempy.plot import visualization_2d_pro as vv


# %%
# Model interpolation
# -------------------
# 

# %% 
# Data Preparation
path_to_data = os.pardir+"/../data/input_data/jan_models/"

geo_data = gp.create_data([0,1000,0,1000,0,1000],resolution=[10,10,10], 
                        path_o = path_to_data + "model5_orientations.csv",
                        path_i = path_to_data + "model5_surface_points.csv") 

# %% 
gp.plot.plot_data(geo_data)

# %% 
geo_data.set_topography()

# %% 
section_dict = {'section1':([0,0],[1000,1000],[100,80]),
                 'section2':([800,0],[800,1000],[150,100]),
                 'section3':([50,200],[100,500],[200,150])} 

# %% 
geo_data.set_section_grid(section_dict)

# %% 
geo_data.grid.sections

# %% 
gp.set_interpolation_data(geo_data, theano_optimizer='fast_compile',
                         verbose=[])

# %% 
gp.map_series_to_surfaces(geo_data, {"Fault_Series":'fault', 
                         "Strat_Series": ('rock2','rock1')})
geo_data.set_is_fault(['Fault_Series'])

# %% 
geo_data.grid.active_grids

# %% 
gp.compute_model(geo_data);

# %% 
# old plotting api
gp.plot.plot_section(geo_data, 0)

# %% 
# old plotting api
gp.plot.plot_section_by_name(geo_data, 'section3', show_all_data=True)

# %% 
# new plotting api
gp._plot.plot_2d(geo_data, section_names = 'section1')


# %%
# Plot2d-Pro: Granular interface:
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 


# %%
# One plot
# ^^^^^^^^
# 

# %% 
# from importlib import reload
# reload(vv)
# p = vv.Plot2D(geo_data)
# p.create_figure((7, 7))

# %% 
p = gp._plot.plot_2d(geo_data)

# %% 
p.fig

# %% 
sec_name = 'section1'

a = p.add_section(sec_name)

# %% 
p.plot_data(a, sec_name, projection_distance=200)
p.fig

# %% 
p.plot_contacts(a, sec_name)
p.fig

# %% 
p.plot_lith(a, sec_name)
p.plot_topography(a, sec_name)

# %% 
p.fig


# %%
# Several plots
# ^^^^^^^^^^^^^
# 

# %% 

sec_name = 'section1'
sec_name_2 = 'section3'


p2 = gp._plot.plot_2d(geo_data, n_axis = 3, figsize=(15,15), # General fig options
                     section_names=[sec_name,'topography'], cell_number=[3], # Defining the sections
                     show_data=False, show_lith=False, show_scalar=False, show_boundaries=False) 

# %% 

# Create the section. This loacte the axes and give the right
# aspect ratio and labels

a = p2.add_section(sec_name_2, ax_pos=224)
# b = p2.add_section(cell_number=3, ax_pos=222)
# t = p2.add_section('topography', ax_pos= 224)

# %% 
p2.fig

# %% 
# Axes 0
p2.plot_contacts(a, sec_name_2)
p2.plot_lith(a, sec_name_2)
p2.plot_data(a, sec_name_2, projection_distance=200)
p2.plot_topography(a, sec_name_2)

# # Axes 1
p2.plot_contacts(p2.axes[0], cell_number=3)
#p2.plot_lith(p2.axes[0], cell_number=3)
p2.plot_scalar_field(p2.axes[0], cell_number=3, sn=1)
#p2.plot_topography(p2.axes[0], cell_number=2)

# #axes2.
p2.plot_lith(p2.axes[1], 'topography')
#p2.plot_scalar_field(p2.axes[1], 'topography', sn=1)
#p2.plot_data(t, 'topography')
p2.plot_contacts(p2.axes[1], 'topography')

# %% 
p2.fig


# %%
# Plotting traces:
# ''''''''''''''''
# 

# %% 
p2.plot_section_traces(p2.axes[1])

# %% 
gp.plot.plot_section_traces(geo_data)


# %%
# Plot API
# ~~~~~~~~
# 


# %%
# If nothing is passed, a Plot2D object is created and therefore you are
# in the same situation as above:
# 

# %% 
p3 = gp._plot.plot_2d(geo_data)



# %%
# Alternatively you can pass section\_names, cell\_numbers + direction or
# any combination of the above:
# 

# %% 
gp._plot.plot_2d(geo_data, section_names=['topography'])

# %% 
gp._plot.plot_2d(geo_data, section_names=['section1'])

# %% 
gp._plot.plot_2d(geo_data, section_names=['section1', 'section2'])

# %% 
gp._plot.plot_2d(geo_data, figsize=(15,15), section_names=['section1', 'section2', 'topography'],
                 cell_number='mid')