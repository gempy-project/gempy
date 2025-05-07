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
import dotenv

dotenv.load_dotenv()

# sphinx_gallery_thumbnail_number = -1

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
# LiquidEarth Integration
# ~~~~~~~~~~~~~~~~~~~~~~~
# Beyond the classical plotting capabilities introduced in GemPy v3, users can now also upload models to LiquidEarth. 
# `LiquidEarth <https://www.terranigma-solutions.com/liquidearth>`_ is a collaborative platform designed for 3D visualization,
# developed by many of the main `gempy` maintainers,  with a strong focus on collaboration and sharing. 
# This makes it an excellent tool for sharing your models with others and viewing them across different platforms.
# To upload a model to LiquidEarth, you must have an account and a user token. Once your model is uploaded, 
# you can easily share the link with anyone.

# %%
link = gpv.plot_to_liquid_earth(
    geo_model=geo_model,
    space_name="[PUBLIC] GemPy Tutorial 1.7: 3-D Visualization",
    file_name="gempy_model",
    user_token=None,  # If None, it will try to grab it from the environment
    grab_link=True,
    make_new_space=False
)

print(f"Generated Link: {link}")

# %%
# Now we can use `this link <https://liquidearth.app.link/gempy-promo>`_ to visualize the model in Liquid Earth.

# %%
# .. image:: /_static/gp_model_in_le.png
