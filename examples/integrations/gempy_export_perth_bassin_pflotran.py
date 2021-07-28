"""
Simulation of the Perth Basin model with PFLOTRAN
=================================================

"""

# %% 
# The purpose of this example is to introduce how to export GemPy geological 
# model to the finite volume reactive transport simulator `PFLOTRAN <https://www.pflotran.org>`_.

# %% 
# Load the Perth Basin model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# Load the Perth basin model previously saved in a zip file,
# add a random topography and compute model:

import gempy as gp
geo_model = gp.load_model("Perth_Basin", path="Perth_Basin.zip")
geo_model.set_topography(source='random')
interp_data = gp.set_interpolator(geo_model,
                                  compile_theano=True,
                                  theano_optimizer='fast_run', gradient=False,
                                  dtype='float32')
gp.compute_model(geo_model)



# %% 
#
# Export the model in PFLOTRAN format
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Two format are available for PFLOTRAN and are automatically determined by the extension:
# 
# * `ascii`: every file are writed into human readable files (`ugi` extension).
# * `hdf5`: all the mesh information are stored into one binary HDF5 file.
# 
# In this example, the `ascii` format is chosen, so the model is saved
# under the name ``perth_bassin_mesh.ugi``:
# 

import gempy.utils.export as export
export.export_pflotran_input(geo_model, path='', filename="perth_basin_mesh.ugi")


# %% 
#
# PFLOTRAN ASCII format description
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
#When using the `ascii` file format, several file are created. Mesh 
#informations are stored into the main file ``perth_bassin_mesh.ugi``. Also, 
#the stratigraphic information are stored in separed files, with the
#name of the formation and the extension `.vs` 
#(for Volume Submesh). For example, the Perth Basin model got several
#formations which are saved in PFLOTRAN format under the following files:
#
#* basement -> basement.vs
#* Cretaceous -> Cretaceous.vs
#* Eneabba -> Eneabba.vs
#* Lesueur -> Lesueur.vs
#* and so on..
#

# %% 
#Group based on stratigraphic information are loaded in PFLOTRAN with the
#card ``REGION`` (see `here <https://www.pflotran.org/documentation/user_guide/cards/subsurface/region_card.html>`_).
#For example for the `Cretaceous` formation:
# .. code-block:: python
#  
#  REGION Cretaceous
#    FILE ./Cretaceous.vs
#  END
#
#Also, when a topography layer is used, the cell located above the 
#topography are stored into the file ``inactive_cells.vs``, so they can
#be inactivated in PFLOTRAN using:
# .. code-block:: python
#  
#  REGION inactive_cells
#    FILE ./inactive_cells.vs
#  END
#    
#  STRATA
#    REGION inactive_cells
#    INACTIVE
#  END 
#
#A sample PFLOTRAN input file is provided to correctly read the GemPy 
#output ``pflotran_perth_bassin.in``.
#It basically read the model and perform a saturated flow simulation 
#using different hydraulic conductivity for each formation.
#
#  .. figure:: permeability_pflotran.png
#    :width: 600
#
#Note: GemPy export a file named ``topography_surface.ss`` that represent
#horizontal cell defining the topography. They can be used to apply a 
#boundary condition such as specifying a rain.


# %% 
# 
#PFLOTRAN HDF5 format description
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
#At the contrary, if the `hdf5` format is used (extension ``.h5``), GemPy
#only create one binary file. The stratigraphic information are stored
#in the HDF5 file, and can also be assessed in PFLOTRAN using the ``REGION`` 
#card. For example, for the `Cretaceous` formation:
#
# .. code-block:: python
#   
#  REGION Cretaceous
#    FILE ./perth_bassin_mesh.h5
#  END
#
#
#The inactive cells above the topography are stored in a group named `Inactive` 
#and can be remove from PFLOTRAN simulation using:
#
# .. code-block:: python
#  
#  REGION Inactive
#    FILE ./perth_bassin_mesh.h5
#  END
#  
#  STRATA
#    REGION inactive_cells
#    INACTIVE
#  END
#
#

# %%
#
#Getting help with PFLOTRAN
#^^^^^^^^^^^^^^^^^^^^^^^^^^
#
#For issue relating with PFLOTRAN, please see the discussion group `here <https://groups.google.com/g/pflotran-users>`_.
#
#
#
  
