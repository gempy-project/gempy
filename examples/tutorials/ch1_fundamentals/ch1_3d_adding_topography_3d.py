"""
1.3d: Adding topography to geological models
============================================

"""

# %%
import gempy as gp
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(55500)

# %%
# 1. The common procedure to set up a model:
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

# %% 
geo_model = gp.create_model('Tutorial_ch1-7_Single_layer_topo')

data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
gp.init_data(geo_model, extent=[450000, 460000, 70000, 80000, -1000, 500],
             resolution=[50, 50, 50],
             path_i=data_path + "/data/input_data/tut-ch1-7/onelayer_interfaces.csv",
             path_o=data_path + "/data/input_data/tut-ch1-7/onelayer_orient.csv")

# %%
# use happy spring colors! 
geo_model.surfaces.colors.change_colors({'layer1': '#ff8000', 'basement': '#88cc60'})

# %%
gp.map_stack_to_surfaces(geo_model, {'series': ('layer1', 'basement')})

# %%
# 2. Adding topography
# ~~~~~~~~~~~~~~~~~~~~
# 


# %%
# 2.b create fun topography
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# 

# %% 
geo_model.set_topography(d_z=np.array([-100, 200]))

# %% 
gp.plot_2d(geo_model)
plt.show()


# %%
g3d = gp.plot_3d(geo_model,
                 show_topography=True,
                 show_lith=False,
                 show_surfaces=False,
                 ve=5)


# %% 
gp.set_interpolator(geo_model, theano_optimizer='fast_compile')

# %% 
gp.compute_model(geo_model)

# %%
# sphinx_gallery_thumbnail_number = 3
g3d = gp.plot_3d(geo_model,
                 show_topography=True,
                 show_lith=False,
                 show_surfaces=False,
                 ve=5)
