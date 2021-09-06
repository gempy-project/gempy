"""
Simulation of the Perth Basin model with PFLOTRAN
=================================================

"""

# %% 
#
# The purpose of this example is to introduce how to export GemPy geological 
# model to the finite volume reactive transport simulator `PFLOTRAN <https://www.pflotran.org>`_.

# %% 
# The Perth Basin model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# First, create the perf bassin model as in `examples/examples/real/Perth_bassin.py`.

import gempy as gp
data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
geo_model = gp.create_model('Perth_Basin')
gp.init_data(geo_model,
             extent=[337000, 400000, 6640000, 6710000, -18000, 1000],
             resolution=[100, 100, 100],
             path_i=data_path + "/data/input_data/Perth_basin/Paper_GU2F_sc_faults_topo_Points.csv",
             path_o=data_path + "/data/input_data/Perth_basin/Paper_GU2F_sc_faults_topo_Foliations.csv")
del_surfaces = ['Cadda', 'Woodada_Kockatea', 'Cattamarra']
geo_model.delete_surfaces(del_surfaces, remove_data=True)
ret = gp.map_stack_to_surfaces(geo_model,
                          {"fault_Abrolhos_Transfer": ["Abrolhos_Transfer"],
                           "fault_Coomallo": ["Coomallo"],
                           "fault_Eneabba_South": ["Eneabba_South"],
                           "fault_Hypo_fault_W": ["Hypo_fault_W"],
                           "fault_Hypo_fault_E": ["Hypo_fault_E"],
                           "fault_Urella_North": ["Urella_North"],
                           "fault_Urella_South": ["Urella_South"],
                           "fault_Darling": ["Darling"],
                           "Sedimentary_Series": ['Cretaceous',
                                                  'Yarragadee',
                                                  'Eneabba',
                                                  'Lesueur',
                                                  'Permian']
                           })
order_series = ["fault_Abrolhos_Transfer",
                "fault_Coomallo",
                "fault_Eneabba_South",
                "fault_Hypo_fault_W",
                "fault_Hypo_fault_E",
                "fault_Urella_North",
                "fault_Darling",
                "fault_Urella_South",
                "Sedimentary_Series", 'Basement']
geo_model.reorder_series(order_series)

geo_model.surface_points.df.dropna(inplace=True)
geo_model.orientations.df.dropna(inplace=True)
geo_model.set_is_fault(["fault_Abrolhos_Transfer",
                        "fault_Coomallo",
                        "fault_Eneabba_South",
                        "fault_Hypo_fault_W",
                        "fault_Hypo_fault_E",
                        "fault_Urella_North",
                        "fault_Darling",
                        "fault_Urella_South"])
                        

fr = geo_model.faults.faults_relations_df.values
fr[:, :-2] = False
ret = geo_model.set_fault_relation(fr)

geo_model.set_topography(source='random')
interp_data = gp.set_interpolator(geo_model,
                                  compile_theano=True,
                                  theano_optimizer='fast_run', gradient=False,
                                  dtype='float32')
gp.compute_model(geo_model)

# %%
# Here is the results:

gp.plot_3d(geo_model, show_topography=True)


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

# sphinx_gallery_thumbnail_path = '_static/permeability_pflotran.png'
import gempy.utils.export as export
export.export_pflotran_input(geo_model, path='', filename="perth_basin_mesh.ugi")


# %% 
#
#PFLOTRAN ASCII format description
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
#horizontal faces defining the topography. They can be used to apply a 
#boundary condition such as specifying a rain. They are imported in PFLOTRAN by:
# .. code-block:: python
#  
#  REGION topo
#    FILE ./topography_surface.ss
#  END

# %% 
# 
#PFLOTRAN HDF5 format description
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
#At the contrary, if the `hdf5` format is used (extension ``.h5``), GemPy
#only create one binary file. The stratigraphic information are stored
#in the HDF5 file, and can also be assessed in PFLOTRAN using the ``REGION`` 
#card. For example, for the `Cretaceous` formation and the topographic
#surface:
#
# .. code-block:: python
#   
#  REGION Cretaceous
#    FILE ./perth_bassin_mesh.h5
#  END
#  REGION Topography_surface
#    FILE ./perth_bassin_mesh.h5
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

