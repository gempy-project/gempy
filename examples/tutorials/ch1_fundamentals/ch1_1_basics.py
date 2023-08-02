"""
1.1 -Basics of geological modeling with GemPy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

"""

# %%
# Importing Necessary Libraries
# """"""""""""""""""""""""""""""
import gempy as gp
import gempy_viewer as gpv

# %%
# Importing and Defining Input Data
# """""""""""""""""""""""""""""""""
# :obj:`gempy.core.data.GeoModel`
# GemPy uses Python objects to store the data that builds the geological model. The main data classes include:
#
#     -  :obj:`gempy.core.data.GeoModel`
#     -  :obj:`gempy.core.data.StructuralFrame`
#     -  :obj:`gempy.core.data.StructuralGroup`
#     -  :obj:`gempy.core.data.StructuralElement`
#     -  :obj:`gempy.core.data.SurfacePointsTable`
#     -  :obj:`gempy.core.data.OrientationsTable`
#     -  :obj:`gempy.core.data.Grid`
#
# Each of these classes will be covered in more depth in a later tutorial :doc:`ch1_2a_data_manipulation`.
#
# You can also create data from raw CSV files (comma-separated values). This could be useful if you are exporting model data
# from a different program or creating it in a spreadsheet software like Microsoft Excel or LibreOffice Calc.
#
# In this tutorial, we'll use CSV files to generate input data. You can find these example files in the `gempy data`
# repository on GitHub. The data consists of x, y, and z positional values for all surface points and orientation
# measurements. Additional data includes poles, azimuth and polarity (or the gradient components). Surface points are
# assigned a formation, which can be a lithological unit (like "Sandstone") or a structural feature (like "Main Fault"). 
#
# It's important to note that, in GemPy, interface position points mark the **bottom** of a layer. If you need points
# to represent the top of a formation (for example, when modeling an intrusion), you can define an inverted orientation measurement.
#
# While generating data from CSV files, we also need to define the model's real extent in x, y, and z. This extent
# defines the area used for interpolation and many of the plotting functions. We also set a resolution to establish a
# regular grid right away. This resolution will dictate the number of voxels used during modeling. We're using a medium
# resolution of 50x50x50 here, which results in 125,000 voxels. The model extent should enclose all relevant data in a
# representative space. As our model voxels are prisms rather than cubes, the resolution can differ from the extent.
# However, it is recommended to avoid going beyond 100 cells in each direction (1,000,000 voxels) to prevent excessive
# computational costs.
#
# .. admonition:: New in GemPy 3!
#
#     GemPy 3 introduces octrees, which allow us to define resolution by specifying the number of octree levels instead
#     of passing a resolution for a regular grid. The number of octree levels corresponds to how many times the grid is
#     halved. Thus, the number of voxels is 2^octree_levels in each direction. For example, 3 octree levels will create
#     a grid with 8x8x8 voxels, 4 octree levels will create a grid with 16x16x16 voxels, and so on. This provides an
#     effective way to control model resolution. However, it is recommended not to exceed 6 octree levels to avoid 
#     escalating computational costs.
#
#

# %%
data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'

geo_model: gp.data.GeoModel = gp.create_geomodel(
    project_name='Tutorial_ch1_1_Basics',
    extent=[0, 2000, 0, 2000, 0, 750],
    resolution=[50, 50, 50],  # * Here we define the resolution of the voxels
    number_octree_levels=4,  # * Here we define the number of octree levels. If octree levels are defined, the resolution is ignored.
    importer_helper=gp.data.ImporterHelper(
        path_to_orientations=data_path + "/data/input_data/getting_started/simple_fault_model_orientations.csv",
        path_to_surface_points=data_path + "/data/input_data/getting_started/simple_fault_model_points.csv",
        hash_surface_points="4cdd54cd510cf345a583610585f2206a2936a05faaae05595b61febfc0191563",
        hash_orientations="7ba1de060fc8df668d411d0207a326bc94a6cdca9f5fe2ed511fd4db6b3f3526"
    )
)

# %% 
# .. admonition:: New in GemPy 3!
#
#    GemPy 3 has introduced the ``ImporterHelper`` class to streamline importing data from various sources. This class
#    simplifies the process of passing multiple arguments needed for importing data and will likely see further 
#    extensions in the future. Currently, one of its uses is to handle `pooch` arguments for downloading data from the internet.
#
# 

# The input data can be reviewed using the properties `surface_points` and `orientations`. However, note that at this point,
# the sequence of formations and their assignment to series are still arbitrary. We will rectify this in the subsequent steps.

# %% 
geo_model.surface_points

# %% 
geo_model.orientations


# %%
# Declaring the Sequential Order of Geological Formations
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# In our model, we want the geological units to appear in the correct chronological order. 
# Such order could be determined by a sequence of stratigraphic deposition, unconformities 
# due to erosion, or other lithological genesis events like igneous intrusions. A similar 
# age-related order is declared for faults in our model. In GemPy, we use the function 
# `gempy.map_stack_to_surfaces` to assign formations or faults to different sequential series 
# by declaring them in a Python dictionary.

# The correct ordering of series is crucial for model construction! It's possible to assign 
# several surfaces to one series. The order of units within a series only affects the color 
# code, so we recommend maintaining consistency. The order can be defined by simply changing 
# the order of the lists within `gempy.core.data.StructuralFrame.structural_groups` and 
# `gempy.core.data.StructuralGroups.elements` attributes.

# Faults are treated as independent groups and must be younger than the groups they affect. 
# The relative order between different faults defines their tectonic relationship 
# (the first entry is the youngest).

# For a model with simple sequential stratigraphy, all layer formations can be assigned to 
# one series without an issue. All unit boundaries and their order would then be determined 
# by interface points. However, to model more complex lithostratigraphical relations and 
# interactions, separate series definition becomes important. For example, modeling an 
# unconformity or an intrusion that disrupts older stratigraphy would require declaring a 
# "newer" series.

# By default, we create a simple sequence inferred from the data:
# 

# %%
geo_model.structural_frame

# %%
# Our example model comprises four main layers (plus an underlying
# basement that is automatically generated by GemPy) and one main normal
# fault displacing those layers. Assuming a simple stratigraphy where each
# younger unit was deposited onto the underlying older one, we can assign
# these layer formations to one series called "Strat\_Series". For the
# fault, we declare a respective "Fault\_Series" as the first key entry in
# the mapping  dictionary. We could give any other names to these
# series, the formations however have to be referred to as named in the
# input data. 
# 


# %%
gp.map_stack_to_surfaces(
    gempy_model=geo_model,
    mapping_object=  # TODO: This mapping I do not like it too much. We should be able to do it passing the data objects directly
    {
        "Fault_Series": 'Main_Fault',
        "Strat_Series": ('Sandstone_2', 'Siltstone', 'Shale', 'Sandstone_1')
    }

)

geo_model.structural_frame  # Display the resulting structural frame


# %% 
gp.set_is_fault(
    frame=geo_model.structural_frame,
    fault_groups=['Fault_Series']
)

# %%
# Now, all surfaces have been assigned to a series and are displayed in the correct order 
# (from young to old).
#
# Returning Information from Our Input Data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Our model input data, named "geo_model", contains all essential information for constructing 
# our model. You can access different types of information by accessing the attributes.
# For instance, you can retrieve the coordinates of our modeling grid as follows:


# %% 
geo_model.grid


# %%
# 
# Visualizing input data
# ~~~~~~~~~~~~~~~~~~~~~~
# 
# We can also visualize our input data. This might for example be useful
# to check if all points and measurements are defined the way we want them
# to. Using the function :obj:`gempy_viewer.plot2d`, we attain a 2D projection of our
# data points onto a plane of chosen *direction* (we can choose this
# attribute to be either :math:`x`, :math:`y` or :math:`z`).
# 

# %%
plot = gpv.plot_2d(geo_model, show_lith=False, show_boundaries=False)


# %%
# Using  :obj:`gempy_viewer.plot_3d`, # we can also visualize this data in 3D. Note that
# direct 3D visualization in GemPy requires `the Visualization
# Toolkit <https://www.vtk.org/>`__ (VTK) to be installed.
# 

# %%
gpv.plot_3d(geo_model, image=False, plotter_type='basic')

# %%
# Model generation
# ~~~~~~~~~~~~~~~~
# 
# Once we have made sure that we have defined all our primary information
# as desired in our object :obj:`gempy.core.data.GeoModel` (named
# ``geo_model`` in these tutorials), we can continue with the next step
# towards creating our geological model: preparing the input data for
# interpolation.
# 
# 
# .. admonition:: New in GemPy 3!
#
#    GemPy 3 does not use either ``theano`` or ``asera`` anymore. Instead, it uses ``numpy`` or ``tensorflow``. For
#    this reason, we do not need to we do need to recompile all the theano fuctions anymore (tensorflow uses eager
#    execution after having profile the XLA compiler and not notice any speed difference).
#


# %%
# The parameters used for the interpolation can be found on :obj:`gempy.core.data.GeoModel.interpolation_options`.
# These fields have meaningful default values, but can be changed if needed. However, users
# should be careful doing so, if they do not fully understand their significance.
# 

# %% 
geo_model.interpolation_options

# %%
# At this point, we have all we need to compute our full model via
# :obj:`gempy.compute_model`. This funtion will return a :obj:`gempy.core.data.Solutions` object

# Below, we illustrate these different model solutions and how they can be
# used.
# 


# %% 
sol = gp.compute_model(geo_model)

# %% 
sol

# %% 
geo_model.solutions

# %%
# Direct model visualization in GemPy
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Model solutions can be easily visualized in 2D sections in GemPy
# directly. Let's take a look at our lithology block:
#

# %%
gpv.plot_2d(geo_model, show_data=True)

# %% 


# %%
# With ``cell_number=25`` and remembering that we defined our resolution
# to be 50 cells in each direction, we have chosen a section going through
# the middle of our block. We have moved 25 cells in ``direction='y'``,
# the plot thus depicts a plane parallel to the :math:`x`- and
# :math:`y`-axes. Setting ``plot_data=True``, we could plot original data
# together with the results. Changing the values for ``cell_number`` and
# ``direction``, we can move through our 3D block model and explore it by
# looking at different 2D planes.
# 
# We can do the same with out lithological scalar-field solution:
# 

# %%
gpv.plot_2d(geo_model, show_data=False, show_scalar=True, show_lith=False)

# %%
gpv.plot_2d(geo_model, series_n=1, show_data=False, show_scalar=True, show_lith=False)

# %%
# This illustrates well the fold-related deformation of the stratigraphy,
# as well as the way the layers are influenced by the fault.
# 
# The fault network modeling solutions can be visualized in the same way:
# 

# # %% 
# geo_model.solutions.scalar_field_at_surface_points
# 
# # %%
# gp.plot_2d(geo_model, show_block=True, show_lith=False)
# plt.show()
# 
# # %%
# gp.plot_2d(geo_model, series_n=1, show_block=True, show_lith=False)
# plt.show()
# 
# # %%
# # Marching cubes and vtk visualization
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # 
# # In addition to 2D sections we can extract surfaces to visualize in 3D
# # renderers. Surfaces can be visualized as 3D triangle complexes in VTK
# # (see function plot\_surfaces\_3D below). To create these triangles, we
# # need to extract respective vertices and simplices from the potential
# # fields of lithologies and faults. This process is automatized in GemPy
# # with the function ``get_surface``\ .
# # 
# 
# # %% 
# ver, sim = gp.get_surfaces(geo_model)
gpv = gpv.plot_3d(geo_model, image=False, plotter_type='basic')
# 
# # %%
# # Using the rescaled interpolation data, we can also run our 3D VTK
# # visualization in an interactive mode which allows us to alter and update
# # our model in real time. Similarly to the interactive 3D visualization of
# # our input data, the changes are permanently saved (in the
# # ``InterpolationInput.dataframe`` object). Additionally, the resulting changes
# # in the geological models are re-computed in real time.
# # 
# 
# 
# # %%
# # Adding topography
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# geo_model.set_topography(d_z=(350, 750))
# 
# # %%
# gp.compute_model(geo_model)
# gp.plot_2d(geo_model, show_topography=True)
# plt.show()
# 
# 
# # sphinx_gallery_thumbnail_number = 9
# gpv = gp.plot_3d(geo_model, plotter_type='basic', show_topography=True, show_surfaces=True,
#                  show_lith=True,
#                  image=False)
# 
# # %%
# # Compute at a given location
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # 
# # This is done by modifying the grid to a custom grid and recomputing.
# # Notice that the results are given as *grid + surfaces\_points\_ref +
# # surface\_points\_rest locations*
# # 
# 
# # %% 
# x_i = np.array([[3, 5, 6]])
# sol = gp.compute_model(geo_model, at=x_i)
# 
# # %%
# # Therefore if we just want the value at **x\_i**:
# 
# # %%
# sol.custom
# 
# # %%
# # This return the id, and the scalar field values for each series
# 
# # %%
# # Save the model
# # ~~~~~~~~~~~~~~
# # 
# 
# # %%
# # GemPy uses Python [pickle] for fast storing temporary objects
# # (https://docs.python.org/3/library/pickle.html). However, module version
# # consistency is required. For loading a pickle into GemPy, you have to
# # make sure that you are using the same version of pickle and dependent
# # modules (e.g.: ``Pandas``, ``NumPy``) as were used when the data was
# # originally stored.
# # 
# # For long term-safer storage we can export the ``pandas.DataFrames`` to
# # csv by using:
# # 
# 
# # %% 
# gp.save_model(geo_model)
