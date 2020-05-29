"""
6.3b - Posterior analysis (preview).
"""

 # These two lines are necessary only if GemPy is not installed
import sys, os
sys.path.append("../../gempy")
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cuda"

# Importing GemPy
import gempy as gp
from examples.tutorials.ch5_probabilistic_modeling.aux_functions.aux_funct import plot_geo_setting

# Embedding matplotlib figures in the notebooks
# %matplotlib inline

# Importing auxiliary libraries
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import arviz as az


# %%
# Model definition
# ----------------
# 
# In the previous example we assume constant thickness to be able to
# reduce the problem to one dimension. This keeps the probabilistic model
# fairly simple since we do not need to deel with complex geometric
# structures. Unfortunaly, geology is all about dealing with complex three
# dimensional structures. In the moment data spread across the physical
# space, the probabilistic model will have to expand to relate data from
# different locations. In other words, the model will need to include
# either interpolations, regressions or some other sort of spatial
# functions. In this paper, we use an advance universal co-kriging
# interpolator. Further implications of using this method will be discuss
# below but for this lets treat is a simple spatial interpolation in order
# to keep the focus on the constraction of the probabilistic model.
# 


# %%
path_dir = os.getcwd()+'/examples/tutorials/ch5_probabilistic_modeling'

# %%
geo_model = gp.load_model(r'2-layers', path=path_dir, recompile=True)

# %%


# %% 
geo_model = gp.create_model('2-layers')
gp.init_data(geo_model, extent=[0, 12e3, -2e3, 2e3, 0, 4e3], resolution=[500,1,500])

# %% 
geo_model.add_surfaces('surface 1')
geo_model.add_surfaces('surface 2')
geo_model.add_surfaces('basement')
dz = geo_model.grid.regular_grid.dz
geo_model.surfaces.add_surfaces_values([dz, 0, 0], ['dz'])
geo_model.surfaces.add_surfaces_values(np.array([2.6, 2.4, 3.2]), ['density'])

# %% 
geo_model.add_surface_points(3e3, 0, 3.05e3, 'surface 1')
geo_model.add_surface_points(9e3, 0, 3.05e3, 'surface 1')

geo_model.add_surface_points(3e3, 0, 1.02e3, 'surface 2')
geo_model.add_surface_points(9e3, 0, 1.02e3, 'surface 2')

geo_model.add_orientations(  6e3, 0, 4e3, 'surface 1', [0,0,1])

# %% 
gp.plot.plot_data(geo_model)


# %%
# Plots:
# ~~~~~~
# 

# %% 
import arviz as az


# %%
# Load data
# ^^^^^^^^^
# 

# %% 
data = az.from_netcdf('australia')

# %% 
data.prior['depth_0'] = data.prior['depths'][0 ,:, 0]
data.prior['depth_1'] = data.prior['depths'][0 ,:, 1]
data.prior['depth_2'] = data.prior['depths'][0 ,:, 2]
data.prior['depth_3'] = data.prior['depths'][0 ,:, 3]

# %% 
data.posterior['depth_0'] =  data.posterior['depths'][0 ,:, 0]
data.posterior['depth_1'] =  data.posterior['depths'][0 ,:, 1]
data.posterior['depth_2'] =  data.posterior['depths'][0 ,:, 2]
data.posterior['depth_3'] =  data.posterior['depths'][0 ,:, 3]

# %% 
az.plot_trace(data, var_names=['depth_0', 'depth_1', 'depth_2', 'depth_3', 'gravity']);

# %% 
az.plot_joint(data, var_names=['depth_1', 'depth_2'])

# %% 
# !git pull
from gempy.bayesian import plot_posterior as pp

import seaborn as sns
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets



# %% 
# %matplotlib notebook
from importlib import reload
reload(pp)
p = pp.PlotPosterior(data)
p.create_figure(figsize=(9,3), joyplot=True)
def change_iteration(iteration):
    p.plot_posterior(['depth_1', 'depth_3'], ['gravity', 'sigma'], 'y', iteration)
interact(change_iteration, iteration=(0, 700, 30))


# %%
# --------------
# 
# Gif 2D
# ------
# 

# %% 
# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import imageio

# !git pull
from gempy.bayesian import plot_posterior as pp

import seaborn as sns


# %% 
p = pp.PlotPosterior(data)
p.create_figure(figsize=(9,6), joyplot=True)


# %% 
image_list2 = []
for i in range(0, 700, 1):
    p.plot_posterior(['depth_1', 'depth_3'], ['gravity', 'sigma'], 'y', i)
    # Used to return the plot as an image rray
    fig = p.fig     # draw the canvas, cache the renderer
    fig.canvas.draw()   
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    image_list2.append(image)

# %% 
kwargs_write = {'fps':1.0, 'quantizer':'nq'}
imageio.mimsave('./hard1_slow_mo.mov', image_list2, fps=3)

# %% 
imageio.mimsave('./hard1.mov', image_list2, fps=9)


# %%
# Plot 3D
# -------
# 

# %% 
geo_model.set_regular_grid(geo_model.grid.regular_grid.extent, [40,20,40])

# %% 
geo_model.set_active_grid('regular', reset=True)

# %% 
gp.set_interpolator(geo_model, output=['geology'],  gradient=False,
                    theano_optimizer='fast_run') 

# %% 
gp.compute_model(geo_model)

# %% 
import pyvista
pyvista.set_plot_theme('doc')

# %% 
pv = gp._plot.plot_3d(geo_model, plotter_type='background', 
                     render_surfaces=False)

# %% 
grid = pv.plot_structured_grid('lith')

# %% 
iteration = 200
geo_model.modify_surface_points([0,1,2,3],
                                Z=data.posterior['depths'][0][iteration])
gp.compute_model(geo_model)

# %% 
pv.set_scalar_data(grid[0])


# %% 
pv.p.open_movie('3D.mov', framerate=9)

for i in range(0, 700, 1):
    geo_model.modify_surface_points([0,1,2,3],
                                    Z=data.posterior['depths'][0][i])
    gp.compute_model(geo_model)
    pv.set_scalar_data(grid[0])
    pv.p.ren_win.Render()
    pv.p.write_frame()

# Close movie and delete object
pv.p.close()


# %%
# --------------
# 


# %%
# Alternative code:
# ~~~~~~~~~~~~~~~~~
# 
# Plot models:
# ^^^^^^^^^^^^
# 

# %% 
geo_model.set_active_grid('regular')

# %% 

geo_model.set_regular_grid(extent=[0, 10000, 0, 10000, 0, 10000], resolution=[100,2,100])
iteration=300

geo_model.modify_surface_points([0,1,2,3], Z=data.posterior['depths'][0, iteration])#data.get_values('depths')[iteration])
gp.compute_model(geo_model, output='gravity')
gp.plot.plot_section(geo_model, 0)

# %% 
geo_model.surfaces.df

# %% 
gp.plot.plot_section(geo_model, 0)


# %%
# Making gif
# ~~~~~~~~~~
# 

# %% 
import imageio

# %% 
pictures = []

for iteration in range(700):
    p.plot_posterior(['depth_2', 'depth_3'], ['gravity', 'sigma'], 'y', iteration)
    p.axjoin.set_xlim(2000, 14000)
    p.axjoin.set_ylim(2000, 14000)
    p.fig.canvas.draw()
    image = np.frombuffer(p.fig.canvas.tostring_rgb(), dtype='uint8')
    pictures.append(image.reshape(p.fig.canvas.get_width_height()[::-1] + (3,)))

# %% 
kwargs_write = {'fps':1.0, 'quantizer':'nq'}
imageio.mimsave('./learning_cheap.gif', pictures[:500:3], fps=24)


# %%
# Pyvista
# ~~~~~~~
# 

# %% 
from gempy.plot import vista
import pyvista as pv
from importlib import reload
reload(vista)
pv.set_plot_theme('document')

gv = vista.Vista(geo_model, plotter_type='basic', notebook=False, real_time=True)
   
a = gv.set_structured_grid()
gv.p.open_gif('learning_3D-block.gif')


# %% 
gv.p.show(auto_close=False, cpos='xz')


# %% 
for iteration in range(0,500,1):
    geo_model.modify_surface_points([0,1,2,3], Z=data.posterior['depths'][0, iteration])#data.get_values('depths')[iteration])
    gp.compute_model(geo_model, output='gravity');
    gv.p.remove_actor(a)
    a = gv.set_structured_grid()    
    gv.p.write_frame()


# %% 
gv.p.close()