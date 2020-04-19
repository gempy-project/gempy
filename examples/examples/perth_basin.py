# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: sphinx
#       format_version: '1.1'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# These two lines are necessary only if GemPy is not installed
import sys, os
sys.path.append("../..")

# Importing GemPy
import gempy as gp

# Embedding matplotlib figures in the notebooks
# %matplotlib inline

# Importing auxiliary libraries
import numpy as np
import matplotlib.pyplot as plt


""
geo_model = gp.create_model('Perth_Basin')

""
gp.init_data(geo_model,
             extent = [337000, 400000, 6640000, 6710000, -18000, 1000],
             resolution = [100,100,100],
             path_i = os.pardir+"/data/input_data/perth_basin/Paper_GU2F_sc_faults_topo_Points.csv", 
             path_o = os.pardir+"/data/input_data/perth_basin/Paper_GU2F_sc_faults_topo_Foliations.csv")



""
geo_model.surfaces

""
del_surfaces = ['Cadda', 'Woodada_Kockatea', 'Cattamarra']

""
geo_model.delete_surfaces(del_surfaces, remove_data=True)

""
# %debug

""
geo_model.series

""
gp.map_series_to_surfaces(geo_model, 
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


""
geo_model.series

""
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

"""
Drop input data from the deleted series:
"""

geo_model.surface_points.df.dropna(inplace=True)
geo_model.orientations.df.dropna(inplace=True)

###############################################################################
# ### Select which series are faults

geo_model.faults

""
geo_model.set_is_fault(["fault_Abrolhos_Transfer",
                        "fault_Coomallo",
                        "fault_Eneabba_South",
                        "fault_Hypo_fault_W",
                        "fault_Hypo_fault_E",
                        "fault_Urella_North",
                        "fault_Darling",
                        "fault_Urella_South"])

###############################################################################
# ## Fault Network 

geo_model.faults.faults_relations_df

""
fr = geo_model.faults.faults_relations_df.values

""
fr[:, :-2] = False
fr

""
geo_model.set_fault_relation(fr)

""
# %matplotlib inline
gp.plot.plot_data(geo_model, direction='z')

""
geo_model.set_topography(source='random')

""
#gp.plot.plot_3D(geo_model)

""
interp_data = gp.set_interpolation_data(geo_model,  
                                        compile_theano=True,
                                        theano_optimizer='fast_run', gradient=False)

""
gp.compute_model(geo_model)

""
gp.plot.plot_section(geo_model, 25)

""
gp.plot.plot_scalar_field(geo_model, 25, series=-1)

""
gp.plot.plot_section(geo_model, 12, direction="y", show_data=True)

""
gp.plot.plot_3D(geo_model, render_data=True, bg_color=(1,1,1))

###############################################################################
# ## Times
#
# #### Fast run
# - 1M voxels:
#   + CPU: intel® Core™ i7-7700HQ CPU @ 2.80GHz × 8  15 s ± 1.02 s per loop (mean ± std. dev. of 7 runs, 1 loop each)   
#   + GPU (4gb) not enough memmory
#   + Ceres 1M voxels 2080 851 ms
#   
# - 250k voxels
#     + GPU 1050Ti: 3.11 s ± 11.8 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
#     + CPU: intel® Core™ i7-7700HQ CPU @ 2.80GHz × 8  2.27 s ± 47.3 ms
#      + 
#     
# #### Fast Compile
# - 250k voxels
#     + GPU 1050Ti: 3.7 s ± 11.8 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
#     + CPU: intel® Core™ i7-7700HQ CPU @ 2.80GHz × 8  14.2 s ± 51.1 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
#
#     

# # %%timeit
# gp.compute_model(geo_model)

""
# ver = np.load('ver.npy')
# sim = np.load('sim.npy')
# lith_block = np.load('lith.npy')

""
# Not updated to GemPy v2 yet
#ste = gp.steno3D(geo_data, 'PerthBasin')
# ste.plot3D_steno_grid(lith_block)
# ste.plot3D_steno_surface(ver, sim)
# ste.proj.upload()
