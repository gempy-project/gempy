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
import pyvista

from subsurface.core.geological_formats import Collars, Survey, BoreholeSet
from subsurface.core.geological_formats.boreholes._combine_trajectories import MergeOptions
from subsurface.core.reader_helpers.readers_data import GenericReaderFilesHelper
from subsurface.core.structs.base_structures.base_structures_enum import SpecialCellCase
from subsurface.modules.reader.wells.read_borehole_interface import read_collar, read_survey, read_lith
import subsurface as ss
from subsurface.modules.visualization import to_pyvista_points, pv_plot, to_pyvista_line, init_plotter

path_to_well_png = '../../common/basics/data/boreholes_concept.png'
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

well_mesh = to_pyvista_points(collars.collar_loc)
pv_plot([well_mesh], image_2d=True)

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

well_mesh = to_pyvista_line(
    line_set=borehole_set.combined_trajectory,
    active_scalar="lith_ids",
    radius=40
)

p = init_plotter()
import matplotlib.pyplot as plt

boring_cmap = plt.get_cmap(name="viridis", lut=14)
p.add_mesh(well_mesh, cmap=boring_cmap)

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

colors_generator = gp.data.ColorsGenerator()
elements = gp.structural_elements_from_borehole_set(
    borehole_set=borehole_set,
    elements_dict={
        "Basement": {
            "id": -1,
            "color": next(colors_generator)
        },
        "etchgoin": {
            "id": 1,
            "color": next(colors_generator)
        },
        "macoma": {
            "id": 2,
            "color": next(colors_generator)
        },
        "chanac": {
            "id": 3,
            "color": next(colors_generator)
        },
        "mclure": {
            "id": 4,
            "color": next(colors_generator)
        },
        "santa_margarita": {
            "id": 5,
            "color": next(colors_generator)
        },
        "fruitvale": {
            "id": 6,
            "color": next(colors_generator)
        },
        "round_mountain": {
            "id": 7,
            "color": next(colors_generator)
        },
        "olcese": {
            "id": 8,
            "color": next(colors_generator)
        },
        "freeman_jewett": {
            "id": 9,
            "color": next(colors_generator)
        },
        "vedder": {
            "id": 10,
            "color": next(colors_generator)
        },
        "eocene": {
            "id": 11,
            "color": next(colors_generator)
        },
        "cretaceous": {
            "id": 12,
            "color": next(colors_generator)
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
import gempy_viewer as gpv

group = gp.data.StructuralGroup(
    name="Stratigraphic Pile",
    elements=elements,
    structural_relation=gp.data.StackRelationType.ERODE
)
structural_frame = gp.data.StructuralFrame(
    structural_groups=[group],
    color_gen=colors_generator
)


# %% [markdown]
# Determine the extent of the geological model from the surface points coordinates.
all_surface_points_coords: gp.data.SurfacePointsTable = structural_frame.surface_points_copy
extent_from_data = all_surface_points_coords.xyz.min(axis=0), all_surface_points_coords.xyz.max(axis=0)

# %% [markdown]
# Create a GeoModel with the specified extent, grid resolution, and interpolation options.
geo_model = gp.data.GeoModel(
    name="Stratigraphic Pile",
    structural_frame=structural_frame,
    grid=gp.data.Grid(
        extent=[extent_from_data[0][0], extent_from_data[1][0], extent_from_data[0][1], extent_from_data[1][1], extent_from_data[0][2], extent_from_data[1][2]],
        resolution=(50, 50, 50)
    ),
    interpolation_options=gp.data.InterpolationOptions(
        range=5,
        c_o=10,
        mesh_extraction=True,
        number_octree_levels=3,
    ),
)

# %% [markdown]
# Visualize the 3D geological model using GemPy's plot_3d function.
gempy_plot = gpv.plot_3d(
    model=geo_model,
    kwargs_pyvista_bounds={
            'show_xlabels': False,
            'show_ylabels': False,
    },
    show=True,
    image=False
)

# %% [markdown]
# Combine all visual elements and display them together.
sp_mesh: pyvista.PolyData = gempy_plot.surface_points_mesh

pyvista_plotter = init_plotter()
pyvista_plotter.show_bounds(all_edges=True)

units_limit = [0, 13]
pyvista_plotter.add_mesh(
    well_mesh.threshold(units_limit),
    cmap="tab20c",
    clim=units_limit
)

pyvista_plotter.add_mesh(
    collar_mesh,
    point_size=10,
    render_points_as_spheres=True
)

pyvista_plotter.add_point_labels(
    points=collars.collar_loc.points,
    labels=collars.ids,
    point_size=10,
    shape_opacity=0.5,
    font_size=12,
    bold=True
)
pyvista_plotter.add_actor(gempy_plot.surface_points_actor)

pyvista_plotter.show()


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
elements
# %% md
# Lets add the first two (remember we always need a basement defined).
# %%
group = gp.data.StructuralGroup(
    name="Stratigraphic Pile Top",
    elements=elements[:3],
    structural_relation=gp.data.StackRelationType.ERODE
)
geo_model.structural_frame.structural_groups[0] = group


# %% md
# And we can set them into the `gempy` model:
# %%
# %%
g2d = gpv.plot_2d(geo_model)
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
# geo_model.add_orientations(X=300000, Y=3930000, Z=0, surface="topo", pole_vector=(0, 0, 1))
gp.add_orientations(
    x=[300000],
    y=[3930000],
    z=[0],
    elements_names=elements[0].name,
    pole_vector=np.array([0, 0, 1]),
    geo_model=geo_model
)

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
geo_model.interpolation_options

# %%
gp.compute_model(geo_model)

# %%
g3d = gpv.plot_3d(geo_model,show_lith=False, show=False)
# %%
g3d.p.add_mesh(well_mesh)
# %%
g3d.p.show()

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
