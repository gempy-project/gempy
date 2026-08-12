"""
Basics: Data Classes and Modeling API
========================================

A closer look at gempy's data classes and modeling API

This tutorial is a written companion to the video tutorials, covering the same simple
fault model but going into more technical depth: the data classes gempy is built on,
constructing structural elements and groups directly rather than through a CSV import,
inspecting a computed model's solutions and meshes, and saving a model to disk.
"""

# %%
import numpy as np

import gempy as gp
import gempy_viewer as gpv

# %%
# gempy's data classes
# -----------------------
# gempy uses a small set of Python classes to store everything that goes into a model:
#
#     -  :obj:`gempy.core.data.GeoModel`
#     -  :obj:`gempy.core.data.StructuralFrame`
#     -  :obj:`gempy.core.data.StructuralGroup`
#     -  :obj:`gempy.core.data.StructuralElement`
#     -  :obj:`gempy.core.data.SurfacePointsTable`
#     -  :obj:`gempy.core.data.OrientationsTable`
#     -  :obj:`gempy.core.data.Grid`
#
# A ``GeoModel`` holds one ``StructuralFrame``, which is an ordered list of
# ``StructuralGroup`` objects (also called series or stacks), each containing one or
# more ``StructuralElement`` objects -- a lithological unit or a fault surface, defined
# by a ``SurfacePointsTable`` and an ``OrientationsTable``. The rest of this tutorial
# builds up a model through these classes and looks at each one along the way.

# %%
# Model setup
# -------------
# Surface points mark the **bottom** of a layer (if you need the top of a formation --
# modeling an intrusion, say -- use an inverted orientation instead). Data can be
# supplied from CSV files, as here, or built up point by point in code, which the next
# tutorial covers.
#
# The model's ``extent`` defines the volume used for interpolation and plotting, and
# should enclose all the input data. ``refinement`` sets the number of octree levels
# used to extract smooth surfaces (see the Grids tutorial for the full explanation of
# how this interacts with ``resolution``).

# %%
data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'

geo_model = gp.create_geomodel(
    project_name='Tutorial_Basics',
    extent=[0, 2000, 0, 2000, 0, 750],
    refinement=6,
    importer_helper=gp.data.ImporterHelper(
        path_to_orientations=data_path + "/data/input_data/getting_started/simple_fault_model_orientations.csv",
        path_to_surface_points=data_path + "/data/input_data/getting_started/simple_fault_model_points.csv",
        hash_surface_points="4cdd54cd510cf345a583610585f2206a2936a05faaae05595b61febfc0191563",
        hash_orientations="7ba1de060fc8df668d411d0207a326bc94a6cdca9f5fe2ed511fd4db6b3f3526"
    )
)

# %%
# ``ImporterHelper`` bundles everything needed to import data from various sources --
# here, CSV files fetched over HTTP and verified against a known hash, matching every
# other tutorial in this documentation.
#
# Reviewing the imported data
# ------------------------------
# The raw imported points and orientations are available as ``surface_points_copy`` and
# ``orientations_copy``:

# %%
geo_model.surface_points_copy

# %%
geo_model.orientations_copy

# %%
# Each structural element is internally tracked by a numeric ID. Note these aren't the
# small, sequential IDs used to color the lithology block in plots -- they're derived
# directly from each element's name and used for tracking identity regardless of
# reordering. ``element_id_name_map`` looks up which ID corresponds to which element:

# %%
geo_model.structural_frame.element_id_name_map

# %%
# Structural groups and series
# -------------------------------
# Geological units need to appear in the correct chronological order -- a sequence of
# deposition, unconformities, intrusions, and so on. In gempy this is expressed by
# assigning each unit (and each fault) to a **structural group**, using
# ``map_stack_to_surfaces``. Units in the same group share one continuous scalar field,
# so the order *within* a group only affects the default color; the order *between*
# groups is what encodes geological age, oldest at the bottom.
#
# Faults are always their own group and must be younger than whatever they affect.
# Where multiple faults are involved, their relative order encodes their tectonic
# relationship (the first entry is the youngest).
#
# This model has one fault and four stratigraphic layers, assigned to two groups:

# %%
gp.map_stack_to_surfaces(
    gempy_model=geo_model,
    mapping_object={
        "Fault_Series": 'Main_Fault',
        "Strat_Series": ('Sandstone_2', 'Siltstone', 'Shale', 'Sandstone_1')
    }
)

# %%
# ``map_stack_to_surfaces`` doesn't yet mark ``Fault_Series`` as a fault -- every group
# defaults to an ``ERODE`` relation (the next section explains what that means).
# ``set_is_fault`` does that:

# %%
gp.set_is_fault(geo_model, ["Fault_Series"])

# %%
# Setting a group as a fault also populates ``fault_relations``: a boolean matrix of
# which groups each fault offsets. Here, ``Fault_Series`` (row 0) affects
# ``Strat_Series`` (column 1), and nothing affects the fault itself:

# %%
geo_model.structural_frame.fault_relations

# %%
# Building structural elements and groups directly
# ---------------------------------------------------
# Importing from a CSV is only one way to get data into a model. Since a
# ``StructuralElement`` is just a plain data class, it can be constructed directly from
# arrays -- useful when adding a unit that doesn't come from a file, or when building a
# model up incrementally (the next tutorial does exactly this, one borehole reading at
# a time). A new element needs at least two surface points and one orientation
# somewhere in its group before a model can be computed:

# %%
new_element = gp.data.StructuralElement(
    name='Example_Surface',
    color=next(geo_model.structural_frame.color_generator),
    surface_points=gp.data.SurfacePointsTable.from_arrays(
        x=np.array([500, 1500]),
        y=np.array([1000, 1000]),
        z=np.array([600, 600]),
        names='Example_Surface'
    ),
    orientations=gp.data.OrientationsTable.initialize_empty()
)
new_element

# %%
# A ``StructuralGroup`` is likewise just a name, a list of elements, and a relation
# type:

# %%
new_group = gp.data.StructuralGroup(
    name='Example_Series',
    elements=[new_element],
    structural_relation=gp.data.StackRelationType.ERODE
)
new_group

# %%
# Adding either of these to a live model is a matter of inserting them into the
# structural frame -- ``existing_group.append_element(...)`` for an element joining an
# existing group, or ``structural_frame.insert_group(index, group)`` for a whole new
# group -- both covered as part of an actual worked example in the next tutorial. This
# example isn't inserted here, to keep the model above unchanged for the rest of this
# tutorial.
#
# Visualizing input data
# -------------------------
# With the data imported and organized into groups, it can be checked visually before
# computing anything. ``plot_2d`` projects the input data onto a plane along a chosen
# ``direction`` (``'x'``, ``'y'``, or ``'z'``, default ``'y'``):

# %%
gpv.plot_2d(geo_model, show_lith=False, show_boundaries=False)

# %%
# and ``plot_3d`` shows the same data in an interactive 3D view:

# %%
gpv.plot_3d(geo_model, show_lith=False)

# %%
# Computing the model
# ----------------------
# The interpolation parameters live in ``interpolation_options``, with sensible
# defaults (see the Grids tutorial for what ``number_octree_levels`` specifically
# controls) -- change them only if you understand the implications:

# %%
geo_model.interpolation_options

# %%
# ``compute_model`` runs the interpolation and returns a ``Solutions`` object, which is
# also stored on the model itself as ``geo_model.solutions`` for later reference:

# %%
gp.compute_model(geo_model)
geo_model.solutions

# %%
# Visualizing the result
# --------------------------
# The computed lithology block plots the same way as the input data, by default showing
# a section through the middle of the model:

# %%
gpv.plot_2d(geo_model, show_data=True, cell_number="mid", direction='y')

# %%
# Each structural group has its own scalar field, selectable via ``series_n`` (its
# position in ``map_stack_to_surfaces``, 0-indexed) -- series 0 is the fault:

# %%
gpv.plot_2d(geo_model, series_n=0, show_data=False, show_scalar=True, show_lith=False)

# %%
# and series 1 is the stratigraphy, visibly offset by the fault:

# %%
gpv.plot_2d(geo_model, series_n=1, show_data=False, show_scalar=True, show_lith=False)

# %%
# The same result in 3D, with the surfaces extracted via dual contouring:

# %%
gpv.plot_3d(geo_model, show_data=False)

# %%
# Adding topography
# --------------------
# gempy supports several other grid types for different purposes -- the Grids tutorial
# covers all of them in depth. A quick, practical one to see here is topography, which
# lets a model's surfaces be intersected with real (or, as below, synthetic) terrain:

# %%
gp.set_topography_from_random(
    grid=geo_model.grid,
    fractal_dimension=1.2,
    d_z=np.array([350, 750]),
    topography_resolution=np.array([50, 50]),
)

gp.compute_model(geo_model)
gpv.plot_2d(geo_model, show_topography=True)

# %%
gpv.plot_3d(geo_model, show_lith=True, show_topography=True)

# %%
# Extracting solutions
# ------------------------
# Beyond plotting, ``geo_model.solutions`` holds the raw building blocks of the model
# for further analysis or export. ``dc_meshes`` is a list of the extracted surface
# meshes, in the same order as the structural frame -- index ``0`` is the youngest
# element, the fault:

# %%
vertices = geo_model.solutions.dc_meshes[0].vertices
edges = geo_model.solutions.dc_meshes[0].edges
vertices.shape, edges.shape

# %%
# These vertex coordinates are in gempy's internal, rescaled coordinate system rather
# than the model's real-world extent. ``input_transform`` (the same transform used to
# normalize input data before interpolation) maps them back:

# %%
geo_model.input_transform.apply_inverse(vertices)

# %%
# ``raw_arrays`` holds the underlying arrays directly -- the lithology block
# (``lith_block``), for instance, comes back as a flat array that needs reshaping to
# the grid's actual resolution to index into as a volume:

# %%
lith_block = geo_model.solutions.raw_arrays.lith_block
lith_block.shape

# %%
lith_block.reshape(geo_model.grid.regular_grid.resolution).shape

# %%
# Saving and loading a model
# ------------------------------
# A ``GeoModel`` can be saved to a single file and reloaded later, without needing to
# redo the setup above:

# %%
gp.save_model(geo_model, path='tutorial_basics_model.gempy')

# %%
reloaded_model = gp.load_model('tutorial_basics_model.gempy')
reloaded_model.structural_frame

# %%
# .. note::
#    Model serialization is still marked as under active development in gempy (you'll
#    see a ``UserWarning`` when saving/loading) -- it works, but the format may still
#    change in a future release.

# sphinx_gallery_thumbnail_number = -3
