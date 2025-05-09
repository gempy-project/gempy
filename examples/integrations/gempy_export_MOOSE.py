"""
Export a geological model from GemPy to use in MOOSE
====================================================

"""

# %%
import gempy as gp

# %%
# Creating a geological model
# ---------------------------
# 
# The procedure of generating a geological model is presented in detail in
# `Chapter
# 1-1 <https://nbviewer.jupyter.org/github/cgre-aachen/gempy/blob/master/notebooks/tutorials/ch1-1_Basics.ipynb>`__
# of the GemPy tutorials, so it will only be briefly presented here
# 

# %% 
# Initiate a model
geo_model = gp.create_model('tutorial_moose_exp')
data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'

# %% 
# Import data from CSV-files with setting the resolution and model extent
gp.init_data(geo_model, [0, 2000., 0, 2000., 0, 2000.], [50, 50, 80],
             path_o=data_path + "/data/input_data/tut_chapter1/simple_fault_model_orientations.csv",
             path_i=data_path + "/data/input_data/tut_chapter1/simple_fault_model_points.csv",
             default_values=True)

# %% 
# present the units and series
geo_model.surfaces

# %% 
# combine units in series and make two series, as the fault needs its own
gp.map_stack_to_surfaces(geo_model,
                         {"Fault_Series": 'Main_Fault',
                          "Strat_Series": ('Sandstone_2', 'Siltstone', 'Shale', 'Sandstone_1', 'basement')},
                         remove_unused_series=True)

# set the fault series to be fault object
geo_model.set_is_fault(['Fault_Series'], change_color=False)

# %% 
# check whether series were assigned correctly
geo_model.surfaces

# %%
# Model generation
# ----------------
# 
# After loading in the data, we set it up for interpolation and compute
# the model.
# 

# %% 
# set up interpolator
gp.set_interpolator(geo_model,
                    compile_aesara=True,
                    aesara_optimizer='fast_compile',
                    verbose=[])

# %% 
# compute the model
gp.compute_model(geo_model, compute_mesh=False);

# %% 
# have a look at the data and computed model
gp.plot_3d(geo_model)

# %%
# Exporting the Model to MOOSE
# ----------------------------
# 
# The voxel-model above already is the same as a model discretized in a
# hexahedral grid, so my immediately be used as input in a simulation
# tool, e.g. `MOOSE <https://mooseframework.org/>`__. For this, we need to
# access to the unit IDs assigned to each voxel in GemPy. The array
# containing these IDs is called ``lith_block``.
# 

# %% 
ids = geo_model.solutions.lith_block
print(ids)

# %%
# This array has the shape of ``(x,)`` and would be immediately useful, if
# GemPy and the chosen simulation code would *populate* a grid in the same
# way. Of course, however, that is not the case. This is why we have to
# restructure the ``lith_block`` array, so it can be read correctly by
# MOOSE.
# 

# %% 
# model resolution
nx, ny, nz = geo_model.grid.octree_grid.resolution

# model extent
xmin, xmax, ymin, ymax, zmin, zmax = geo_model.grid.octree_grid.extent

# %%
# These two parameters are important to, a) restructure ``lith_block``,
# and b) write the input file for MOOSE correctly. For a), we need to
# reshape ``lith_block`` again to its three dimensions and *re-flatten* it
# in a *MOOSE-conform* way.
# 

# %% 
# reshape to 3D array
units = ids.reshape((nx, ny, nz))
# flatten MOOSE conform
units = units.flatten('F')

# %%
# | The importance of ``nx, ny, nz`` is apparent from the cell above. But
#   what about ``xmin``, …, ``zmax``?
# | A MOOSE input-file for mesh generation has the following syntax:
# 
# .. code:: python
# 
#    [MeshGenerators]
#      [./gmg]
#        type = GeneratedMeshGenerator
#        dim = 3
#        nx = 50
#        ny = 50
#        nz = 80
#        xmin = 0.0
#        xmax = 2000.0
#        yim = 0.0
#        ymax = 2000.0
#        zmin = 0.0
#        zmax = 2000.0
#        block_id = '1 2 3 4 5 6'
#        block_name = 'Main_Fault Sandstone_2 Siltstone Shale Sandstone_1 basement'
#      [../]
# 
#      [./subdomains]
#        type = ElementSubdomainIDGenerator
#        input = gmg
#        subdomain_ids = ' ' # here you paste the transformed lith_block vector
#      [../]
#    []
# 
#    [Mesh]
#      type = MeshGeneratorMesh
#    []
# 
# So these parameters are required inputs in the ``[MeshGenerators]``
# object in the MOOSE input file. ``GemPy`` has a method to directly
# create such an input file, stored in ``gempy.utils.export.py``.
# 
# The following cell shows how to call the method:
# 

# %%

# sphinx_gallery_thumbnail_path = '_static/GemPy_model_combined.png'
import gempy_plugins.utils.export as export
export.export_moose_input(geo_model, path='')

# %%
# This method automatically stores a file
# ``geo_model_units_moose_input.i`` at the specified path. Either this
# input file could be extended with parameters to directly run a
# simulation, or it is used just for creating a mesh. In the latter case,
# the next step would be, to run the compiled MOOSE executable witch the
# optional flag ``--mesh-only``.
# 
# E.g. with using the `PorousFlow
# module <https://mooseframework.inl.gov/modules/porous_flow/>`__:
# 
# .. code:: bash
# 
#    $path_to_moose/moose/modules/porous_flow/porous_flow-opt -i pct_voxel_mesh.i --mesh-only
# 
# How to compile MOOSE is described in their
# `documentation <https://mooseframework.inl.gov/getting_started/index.html>`__.
# 
# The now generated mesh with the name
# ``geo_model_units_moose_input_in.e`` can be used as input for another
# MOOSE input file, which contains the main simulation parameters. To call
# the file with the grid, the following part has to be added in the MOOSE
# simulation input file:
# 
# .. code:: python
# 
#    [Mesh]
#      file = geo_model_units_moose_input_in.e
#    []
# 
# .. raw:: html
# 
#    <hr>
# 
# The final output of the simulation may also be such an ``.e``, which
# can, for instance, be opened with
# `paraview <https://www.paraview.org/>`__. A simulated temperature field
# (purely conductive) of the created model would look like this:
# 
# .. figure:: https://raw.githubusercontent.com/Japhiolite/a-Moose-and-you/master/imgs/GemPy_model_combined.png
#    :alt: gempy_temperature
#
#
