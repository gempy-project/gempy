"""
GemPy - Subsurface Link
=======================
"""

import pooch
import numpy as np
import pandas as pd

import subsurface as sb
from subsurface.reader import read_netcdf

data_url = "https://raw.githubusercontent.com/softwareunderground/subsurface/main" \
           "/examples/tutorials/wells_unstructured.nc"

data_hash = "206290db4e563e379361725349ebf4a02628f4700d361599aedff37fab9cf5b9"
borehole_unstructured_data_file = pooch.retrieve(url=data_url,
                                                 known_hash=data_hash)

unstruct = read_netcdf.read_unstruct(borehole_unstructured_data_file)
unstruct

# %%
element = sb.LineSet(unstruct)
lines_mesh = sb.visualization.to_pyvista_line(element, radius=50)

# Plot default LITH
sb.visualization.pv_plot([lines_mesh])


# %% md
# Findig the boreholes bases
# --------------------------
# GemPy interpolates the bottom of a unit, therefore we need to be able to extract those points to be able tointerpolate them.
# xarray, pandas and numpy are using the same type of memory representation what makes possible to use the same or at least similar methods to manipulate the data to our will.
# Lets find the base points of each well:

# %%

# Creating references to the xarray.DataArray
cells_attr = unstruct.data.cell_attrs
cells = unstruct.data.cells
vertex = unstruct.data.vertex
# Find vertex points at the boundary of two units
# Marking each vertex
bool_prop_change = cells_attr.values[1:] != cells_attr.values[:-1]
# Getting the index of the vertex
args_prop_change = np.where(bool_prop_change)[0]
# Getting the attr values at those points
vals_prop_change = cells_attr[args_prop_change]
vals_prop_change.to_pandas()

# %%
# Getting the vertex values at those points
vertex_args_prop_change = cells[args_prop_change, 1]
interface_points = vertex[vertex_args_prop_change]
interface_points


# %%
# Creating a new UnstructuredData
interf_us= sb.UnstructuredData.from_array(vertex=interface_points.values, cells="points",
                                          cells_attr=vals_prop_change.to_pandas())
interf_us

# %% md
# This new UnstructuredData object instead containing data that represent lines, contain point data at the bottom of each
# unit. We can plot it very similar as before:

element = sb.PointSet(interf_us)
point_mesh = sb.visualization.to_pyvista_points(element)
sb.visualization.pv_plot([lines_mesh, point_mesh])

# %% md
# GemPy: Initialize model
# -----------------------
# The first step to create a GemPy model is create a gempy.

# %%

import gempy as gp
geo_model = gp.create_model("getting started")
geo_model.set_regular_grid(extent=[275619, 323824, 3914125, 3961793, -3972.6, 313.922], resolution=[50,50,50])
gp.set_interpolator(geo_model, aesara_optimizer='fast_compile', verbose=[])

# %% md
# Making a model step by step.
# ----------------------------

# The temptation at this point is to bring all the points into gempy and just interpolate. However, often that strategy
# results in ill posed problems due to noise or irregularities in the data. gempy has been design to being able to
# iterate rapidly and therefore a much better workflow use to be creating the model step by step.
#
# To do that, lets define a function that we can pass the name of the formation and get the assotiated vertex. Grab from
# the interf_us the XYZ coordinates of the first layer:

# %%
def get_interface_coord_from_surfaces(surface_names: list, verbose=False):
    df = pd.DataFrame(columns=["X", "Y", "Z", "surface"])

    for e, surface_name in enumerate(surface_names):
        # The properties in subsurface start at 1
        val_property = formations.index(surface_name) + 1
        # Find the cells with the surface id
        args_from_first_surface = np.where(vals_prop_change == val_property)[0]
        if verbose: print(args_from_first_surface)
        # Find the vertex
        points_from_first_surface = interface_points[args_from_first_surface]
        if verbose: print(points_from_first_surface)

        # xarray.DataArray to pandas.DataFrame
        surface_pandas = points_from_first_surface.to_pandas()

        # Add formation column
        surface_pandas["surface"] = surface_name
        df = df.append(surface_pandas)

    return df.reset_index()

# %% md
# Surfaces
# ++++++++

formations = ["topo", "etchegoin", "macoma", "chanac", "mclure",
              "santa_margarita", "fruitvale",
              "round_mountain", "olcese", "freeman_jewett", "vedder", "eocene",
              "cretaceous",
              "basement", "null"]

# %%
geo_model.add_features("Formations")
one_formation_every = 3
geo_model.add_surfaces(formations[0:4*one_formation_every:one_formation_every])

geo_model.map_stack_to_surfaces({"Formations": ["etchegoin", "macoma", "chanac", "mclure"],
                                 "Default series": ["topo"]},
                                set_series=False)


# %%
gempy_surface_points = get_interface_coord_from_surfaces(formations[0:3*one_formation_every:one_formation_every])

# %%
geo_model.set_surface_points(gempy_surface_points, update_surfaces=False)
geo_model.update_to_interpolator()


# %% md
# Adding orientations
# -------------------

# %%
# find neighbours
neighbours = gp.select_nearest_surfaces_points(geo_model, geo_model._surface_points.df, 2)

# calculate all fault orientations
gp.set_orientation_from_neighbours_all(geo_model, neighbours)

# %% md
# Using the flag to subsurface, the result of the interpolation will get stored in `subsurface` data objects. In the
# future exporting to subsurface will be the default behaviour.

# %%
gp.compute_model(geo_model, to_subsurface=True)

# %%
p3d = gp.plot_3d(geo_model)

# %%
geo_model.solutions.s_regular_grid

# %%
geo_model.solutions.meshes




