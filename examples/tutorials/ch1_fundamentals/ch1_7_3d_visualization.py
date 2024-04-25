"""
1.7: 3-D Visualization
======================

"""

# %% 
# Importing GemPy
import gempy as gp
import gempy_viewer as gpv
from gempy import generate_example_model
from gempy.core.data.enumerators import ExampleModel

# %%
# Loading an example geomodel
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

# %%


geo_model = generate_example_model(ExampleModel.GRABEN)

gp.compute_model(geo_model)

# %%
# Basic plotting API
# ------------------
# 


# %%
# Data plot
# ~~~~~~~~~
# 

# %% 
gpv.plot_3d(
    model=geo_model,
    show_surfaces=False,
    show_data=True,
    show_lith=False,
    image=False
)

# %%
# Geomodel plot
# ~~~~~~~~~~~~~
# 

# %% 
gpv.plot_3d(geo_model, image=False)

# %%
# sphinx_gallery_thumbnail_number = 2

# %%
# TODO Add explanation
link = gpv.plot_to_liquid_earth(
    geo_model=geo_model,
    space_name="new_space_name",
    file_name="gempy_model",
    user_token=None,  # If None, it will try to grab it from the environment
    grab_link=True
)

print(link)
link