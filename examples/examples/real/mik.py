# %% md
# <img src="https://docs.gempy.org/_static/logos/gempy.png" alt="drawing" width="400"/>
# 
# # 2.2 From Drillhole Data to GemPy Model
# 
# %%
# List of relative paths used during the workshop
import numpy as np
import pandas as pd
import pooch

from subsurface.core.geological_formats import Collars, Survey, BoreholeSet
from subsurface.core.geological_formats.boreholes._combine_trajectories import MergeOptions
from subsurface.core.reader_helpers.readers_data import GenericReaderFilesHelper
from subsurface.core.structs.base_structures.base_structures_enum import SpecialCellCase
from subsurface.modules.reader.wells.read_borehole_interface import read_collar, read_survey, read_lith
import subsurface as ss
from subsurface.modules.visualization import to_pyvista_points, pv_plot, to_pyvista_line, init_plotter

path_to_well_png = '../../common/basics/data/wells.png'
path_to_checkpoint_1 = '../../common/basics/checkpoints/checkpoint1.pickle'
path_to_checkpoint_2 = '../../common/basics/checkpoints/checkpoint2.pickle'
upgrade_pickles = False
# %%
# Importing GemPy
import gempy as gp

#
# %% md
# We use `pooch` to download the dataset into a temp file:
# %%
url = "https://raw.githubusercontent.com/softwareunderground/subsurface/main/tests/data/borehole/kim_ready.csv"
# known_hash = "efa90898bb435daa15912ca6f3e08cd3285311923a36dbc697d2aafebbafa25f"
known_hash = "a91445cb960526398e25d8c1d2ab3b3a32f7d35feaf33e18887629b242256ab6"

# Your code here:
raw_borehole_data_csv = pooch.retrieve(url, known_hash)

# %% md
# Now we can use `subsurface` function to help us reading csv files into pandas dataframes that the package can understand. Since the combination of styles data is provided can highly vary from project to project, `subsurface` provides some *helpers* functions to parse different combination of .csv
# %%

collar_df: pd.DataFrame = read_collar(
    GenericReaderFilesHelper(
        file_or_buffer=raw_borehole_data_csv,
        index_col="name",
        usecols=['x', 'y', 'altitude', "name"],
        columns_map={
            "name": "id",  # ? Index name is not mapped
            "X": "x",
            "Y": "y",
            "altitude": "z"
        }
    )
)
# TODO: df to unstruct
unstruc: ss.UnstructuredData = ss.UnstructuredData.from_array(
    vertex=collar_df[["x", "y", "z"]].values,
    cells=SpecialCellCase.POINTS
)
points = ss.PointSet(data=unstruc)
collars: Collars = Collars(
    ids=collar_df.index.to_list(),
    collar_loc=points
)

s = to_pyvista_points(collars.collar_loc)
pv_plot([s], image_2d=True)

# %%
# Your code here:
survey_df: pd.DataFrame = read_survey(
    GenericReaderFilesHelper(
        file_or_buffer=raw_borehole_data_csv,
        index_col="name",
        usecols=["name", "md"]
    )
)

survey: Survey = Survey.from_df(survey_df)

survey

# %%
# Your code here:
lith = read_lith(
    GenericReaderFilesHelper(
        file_or_buffer=raw_borehole_data_csv,
        usecols=['name', 'top', 'base', 'formation'],
        columns_map={
            'top': 'top',
            'base': 'base',
            'formation': 'component lith',
        }
    )
)

survey.update_survey_with_lith(lith)

lith

borehole_set = BoreholeSet(
    collars=collars,
    survey=survey,
    merge_option=MergeOptions.INTERSECT
)

# %%

s = to_pyvista_line(
    line_set=borehole_set.combined_trajectory,
    active_scalar="lith_ids",
    radius=40
)

p = init_plotter()
import matplotlib.pyplot as plt

boring_cmap = plt.get_cmap(name="viridis", lut=14)
p.add_mesh(s, cmap=boring_cmap)

collar_mesh = to_pyvista_points(collars.collar_loc)
p.add_mesh(collar_mesh, render_points_as_spheres=True)
p.add_point_labels(
    points=collars.collar_loc.points,
    labels=collars.ids,
    point_size=10,
    shape_opacity=0.5,
    font_size=12,
    bold=True
)

if plot3D := False:
    p.show()
else:
    img = p.show(screenshot=True)
    img = p.last_image
    fig = plt.imshow(img)
    plt.axis('off')
    plt.show(block=False)
    p.close()

# %% m
# Welly is a very powerful tool to inspect well data but it was not design for 3D. However they have a method to export XYZ coordinates of each of the well that we can take advanatage of to create a `subsurface.UnstructuredData` object. This object is one of the core data class of `subsurface` and we will use it from now on to keep working in 3D.
# %%
elements = gp.structural_elements_from_borehole_set(
    borehole_set=borehole_set,
    elements_dict={
        "null": {
            "id": -1,
            "color": "#983999"
        },
        "etchgoin": {
            "id": 1,
            "color": "#00923f"
        },
        "macoma": {
            "id": 2,
            "color": "#da251d"
        },
        "chanac": {
            "id": 3,
            "color": "#f8c300"
        },
        "mclure": {
            "id": 4,
            "color": "#bb825b"
        },
        "santa_margarita": {
            "id": 5,
            "color": "#983999"
        },
        "fruitvale": {
            "id": 6,
            "color": "#00923f"
        },
        "round_mountain": {
            "id": 7,
            "color": "#da251d"
        },
        "olcese": {
            "id": 8,
            "color": "#f8c300"
        },
        "freeman_jewett": {
            "id": 9,
            "color": "#bb825b"
        },
        "vedder": {
            "id": 10,
            "color": "#983999"
        },
        "eocene": {
            "id": 11,
            "color": "#00923f"
        },
        "cretaceous": {
            "id": 12,
            "color": "#da251d"
        },
    }
)


# %% md
# ## GemPy: Initialize model
# 
# The first step to create a GemPy model is create a `gempy.Model` object that will
# contain all the other data structures and necessary functionality. In addition
#  for this example we will define a *regular grid* since the beginning.
# This is the grid where we will interpolate the 3D geological model.
# 
# GemPy is based on a **meshless interpolator**. In practice this means that we can
# interpolate any point in a 3D space. However, for convenience, we have built some
# standard grids for different purposes. At the current day the standard grids are:
# 
# - **Regular grid**: default grid mainly for general visualization
# - **Custom grid**: GemPy's wrapper to interpolate on a user grid
# - **Topography**: Topographic data use to be of high density. Treating it as an independent
#   grid allow for high resolution geological maps
# - **Sections**: If we predefine the section 2D grid we can directly interpolate at those
#   locations for perfect, high resolution estimations
# - **Center grids**: Half sphere grids around a given point at surface. This are specially tuned
#   for geophysical forward computations
# %%
import gempy as gp

extent = [275619, 323824, 3914125, 3961793, -3972.6, 313.922]

# Your code here:
geo_model = gp.create_model("getting started")
geo_model.set_regular_grid(extent=extent, resolution=[50, 50, 50])
# %% md
# GemPy core code is written in Python. However for efficiency and gradient based
# machine learning most of heavy computations happen in optimize compile code,
#  either C or CUDA for GPU.
# 
# To do so, GemPy rely on the library `Theano`. To guarantee maximum optimization
# `Theano` requires to compile the code for every Python kernel. The compilation is
# done by calling the following line at any point (before computing the model):
# %%
gp.set_interpolator(geo_model, theano_optimizer='fast_compile', verbose=[])


# %% md
# ## Making a model step by step.
# 
# The temptation at this point is to bring all the points into `gempy` and just interpolate. However, often that strategy results in ill posed problems due to noise or irregularities in the data. `gempy` has been design to being able to iterate rapidly and therefore a much better workflow use to be creating the model step by step.
# 
# To do that, lets define a function that we can pass the name of the formation and get the assotiated vertex. Grab from the `interf_us` the XYZ coordinates of the first layer:
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
# ### Surfaces
# 
# GemPy is a surface based interpolator. This means that all the input data we add has to be refereed to a surface. The
#  surfaces always mark the **bottom** of a unit. 
#  
# This is a list with the formations names for this data set.
# %%
formations
# %% md
# Lets add the first two (remember we always need a basement defined).
# %%
geo_model.add_surfaces(formations[:2])
# %% md
# Using the function defined above we just extract the surface points for **topo**:
# %%
gempy_surface_points = get_interface_coord_from_surfaces(["topo"])
gempy_surface_points
# %% md
# And we can set them into the `gempy` model:
# %%
geo_model.set_surface_points(gempy_surface_points, update_surfaces=False)
geo_model.update_to_interpolator()
# %%
g2d = gp.plot_2d(geo_model)
# %%
g2d.fig
# %% md
# The **minimum amount of input data** to interpolate anything in `gempy` is:
# 
# a) 2 surface points per surface
# 
# b) One orientation per series.
# 
# Lets add an orientation:
# %%
geo_model.add_orientations(X=300000, Y=3930000, Z=0, surface="topo", pole_vector=(0, 0, 1))
# %% md
# GemPy depends on multiple data objects to store all the data structures necessary
# to construct an structural model. To keep all the necessary objects in sync the
# class `gempy.ImplicitCoKriging` (which `geo_model` is instance of) will provide the
# necessary methods to update these data structures coherently.
# 
# At current state (gempy 2.2), the data classes are:
# 
# - `gempy.SurfacePoints`
# - `gempy.Orientations`
# - `gempy.Surfaces`
# - `gempy.Stack` (combination of `gempy.Series` and `gempy.Faults`)
# - `gempy.Grid`
# - `gempy.AdditionalData`
# - `gempy.Solutions`
# 
# Today we will look into details only some of these classes but what is important
# to notice is that you can access these objects as follows:
# %%
geo_model.additional_data
# %%
gp.compute_model(geo_model)
# %%
g3d = gp.plot_3d(geo_model, plotter_type="background")
# %%
g3d.p.add_mesh(pyvista_mesh)
# %%
plot_pyvista_to_notebook(g3d.p)
# %% md
# ## Second layer
# %%
geo_model.add_surfaces(formations[2])
# %%
gempy_surface_points = get_interface_coord_from_surfaces(formations[:2])
geo_model.set_surface_points(gempy_surface_points, update_surfaces=False)
geo_model.update_to_interpolator()
# %%
gp.compute_model(geo_model)
# %%
live_plot = gp.plot_3d(geo_model, plotter_type="background", show_results=True)
# %%
plot_pyvista_to_notebook(live_plot.p)
# %%
live_plot.toggle_live_updating()
# %% md
# ### Trying to fix the model: Multiple Geo. Features/Series
# %%
geo_model.add_features("Formations")
# %%
geo_model.map_stack_to_surfaces({"Form1": ["etchegoin", "macoma"]}, set_series=False)
# %%
geo_model.add_orientations(X=300000, Y=3930000, Z=0, surface="etchegoin", pole_vector=(0, 0, 1), idx=1)
# %%
gp.compute_model(geo_model)
# %%
h3d = gp.plot_3d(geo_model, plotter_type="background", show_lith=False, ve=5)
# %%
plot_pyvista_to_notebook(h3d.p)
# %% md
# ## Last layers for today
# %%
geo_model.add_surfaces(formations[3:5])
# %%
f_last = formations[:4]
f_last
# %%
gempy_surface_points = get_interface_coord_from_surfaces(f_last)
geo_model.set_surface_points(gempy_surface_points, update_surfaces=False)
geo_model.update_to_interpolator()
# %%
gp.compute_model(geo_model)
# %%
p3d_4 = gp.plot_3d(geo_model, plotter_type="background", show_lith=False, ve=5)
# %%
plot_pyvista_to_notebook(p3d_4.p)
# %%
geo_model.add_orientations(X=321687.059770, Y=3.945955e+06, Z=0, surface="etchegoin", pole_vector=(0, 0, 1), idx=1)
gp.compute_model(geo_model)
p3d_4.plot_surfaces()
# %%
geo_model.add_orientations(X=277278.652995, Y=3.929298e+06, Z=0, surface="etchegoin", pole_vector=(0, 0, 1), idx=2)
gp.compute_model(geo_model)
p3d_4.plot_surfaces()
# %% md
# ## Adding many more orientations
# %%
# find neighbours
neighbours = gp.select_nearest_surfaces_points(geo_model, geo_model._surface_points.df, 2)

# calculate all fault orientations
new_ori = gp.set_orientation_from_neighbours_all(geo_model, neighbours)
new_ori.df.head()
# %%
gp.compute_model(geo_model)
# %%
p3d_4.plot_orientations()
p3d_4.plot_surfaces()
# %%
p3d_4.p.add_mesh(pyvista_mesh)

# %%
plot_pyvista_to_notebook(p3d_4.p)
# %% md
# -----
# 
# ## Thank you for your attention
# 
# 
# #### Extra Resources
# 
# Page:
# https://www.gempy.org/
# 
# Paper:
# https://www.gempy.org/theory
# 
# Gitub:
# https://github.com/cgre-aachen/gempy
# 
# #### Further training and collaborations
# https://www.terranigma-solutions.com/
# 
# ![Terranigma_Logotype_black.png](attachment:200622_Terranigma_Logotype_black.png)
# 
#
