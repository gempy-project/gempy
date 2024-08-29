# %% md
# <img src="https://docs.gempy.org/_static/logos/gempy.png" alt="drawing" width="400"/>
# 
# # 2.2 From Drillhole Data to GemPy Model
# 
# %%
# List of relative paths used during the workshop
import pooch

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
reading_collars = ReaderFilesHelper(
    file_or_buffer=raw_borehole_data_csv,
    index_col="name",
    usecols=['x', 'y', 'altitude', "name"]
)

reading_collars
# %%
from dataclasses import asdict

asdict(reading_collars)
# %%
collar = read_collar(reading_collars)

collar
# %%
# Your code here:
survey = read_survey(
    ReaderFilesHelper(
        file_or_buffer=raw_borehole_data_csv,
        index_col="name",
        usecols=["name", "md"]
    )
)

survey
# %%
# Your code here:
lith = read_lith(
    ReaderFilesHelper(
        file_or_buffer=raw_borehole_data_csv,
        usecols=['name', 'top', 'base', 'formation'],
        columns_map={'top'      : 'top',
                     'base'     : 'base',
                     'formation': 'component lith',
                     }
    )
)

lith

# %% md
# ### Welly
# 
# Welly is a family of classes to facilitate the loading, processing, and analysis of subsurface wells and well data, such as striplogs, formation tops, well log curves, and synthetic seismograms.
# 
# We are using welly to convert pandas data frames into classes to manipulate well data. The final goal is to extract 3D coordinates and properties for multiple wells.
# 
# The class `WellyToSubsurfaceHelper` contains the methods to create a `welly` project and export it to a `subsurface` data class.
# %%
wts = sb.reader.wells.WellyToSubsurfaceHelper(collar_df=collar, survey_df=survey, lith_df=lith)
# %% md
# In the field p is stored a welly project (https://github.com/agile-geoscience/welly/blob/master/tutorial/04_Project.ipynb)and we can use it to explore and visualize properties of each well.
# %%
wts.p
# %%
stripLog = wts.p[0].data['lith']
stripLog
# %%
stripLog.plot()
plt.gcf()
# %%
welly_well = wts.p[0].data["lith_log"]
welly_well
# %% md
# ## Welly to Subsurface 
# 
# Welly is a very powerful tool to inspect well data but it was not design for 3D. However they have a method to export XYZ coordinates of each of the well that we can take advanatage of to create a `subsurface.UnstructuredData` object. This object is one of the core data class of `subsurface` and we will use it from now on to keep working in 3D.
# %%
formations = ["topo", "etchegoin", "macoma", "chanac", "mclure",
              "santa_margarita", "fruitvale",
              "round_mountain", "olcese", "freeman_jewett", "vedder", "eocene",
              "cretaceous",
              "basement", "null"]

unstruct = sb.reader.wells.welly_to_subsurface(wts, table=[Component({'lith': l}) for l in formations])
unstruct.data
# %% md
# At each core `UstructuredData` is a wrapper of a `xarray.Dataset`. Although slightly flexible, any `UnstructuredData` will contain 4 `xarray.DataArray` objects containing vertex, cells, cell attributes and vertex attibutes. This is the minimum amount of information necessary to work in 3D. 
# %% md
# From an `UnstructuredData` we can construct *elements*. *elements* are a higher level construct and includes the definion of type of geometric representation - e.g. points, lines, surfaces, etc. For the case of borehole we will use LineSets. *elements* have a very close relation to `vtk` data structures what enables easily to plot the data using `pyvista`
# %%
element = sb.LineSet(unstruct)
pyvista_mesh = sb.visualization.to_pyvista_line(element, radius=50)

# Plot default LITH
interactive_plot = sb.visualization.pv_plot([pyvista_mesh], background_plotter=True)
# %%
plot_pyvista_to_notebook(interactive_plot)
# %% md
# ## Finding the boreholes bases
# 
# `GemPy` interpolates the bottom of a unit, therefore we need to be able to extract those points to be able tointerpolate them. `xarray`, `pandas` and `numpy` are using the same type of memory representation what makes possible to use the same or at least similar methods to manipulate the data to our will. 
# 
# Lets find the base points of each well:
# %%
# Creating references to the xarray.DataArray
cells_attr = unstruct.data.cell_attrs
cells = unstruct.data.cells
vertex = unstruct.data.vertex
# %%
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
interf_us = sb.UnstructuredData.from_array(vertex=interface_points.values, cells="points",
                                           cells_attr=vals_prop_change.to_pandas())
interf_us
# %% md
# This new `UnstructuredData` object instead containing data that represent lines, contain point data at the bottom of each unit. We can plot it very similar as before:
# %%
element = sb.PointSet(interf_us)
pyvista_mesh = sb.visualization.to_pyvista_points(element)
interactive_plot.add_mesh(pyvista_mesh)
# %%
plot_pyvista_to_notebook(interactive_plot)
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