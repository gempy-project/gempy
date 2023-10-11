"""
Perth basin.
============
"""
import os

# Importing GemPy
import gempy as gp
import gempy_viewer as gpv

# Importing auxiliary libraries
import matplotlib

matplotlib.rcParams['figure.figsize'] = (20.0, 10.0)
os.environ["aesara_FLAGS"] = "mode=FAST_RUN,device=cuda"

# %%
cwd = os.getcwd()
if 'examples' not in cwd:
    data_path = os.getcwd() + '/examples'
else:
    data_path = cwd + '/../..'

# %% 
geo_model: gp.data.GeoModel = gp.create_geomodel(
    project_name='Perth_Basin',
    extent=[337000, 400000, 6640000, 6710000, -18000, 1000],
    resolution=[100, 100, 100],
    refinement=6,
    importer_helper=gp.data.ImporterHelper(
        path_to_orientations=data_path + "/data/input_data/perth_basin/Paper_GU2F_sc_faults_topo_Foliations.csv",
        path_to_surface_points=data_path + "/data/input_data/perth_basin/Paper_GU2F_sc_faults_topo_Points.csv",
    )
)

# %%
geo_model.structural_frame

# %% 
del_surfaces = ['Cadda', 'Woodada_Kockatea', 'Cattamarra']
for s in del_surfaces:
    gp.remove_element_by_name(geo_model, s)

geo_model.structural_frame

# %% 
gp.map_stack_to_surfaces(
    gempy_model=geo_model,
    mapping_object={
        "fault_Abrolhos_Transfer": ["Abrolhos_Transfer"],
        "fault_Coomallo": ["Coomallo"],
        "fault_Eneabba_South": ["Eneabba_South"],
        "fault_Hypo_fault_W": ["Hypo_fault_W"],
        "fault_Hypo_fault_E": ["Hypo_fault_E"],
        "fault_Urella_North": ["Urella_North"],
        "fault_Darling": ["Darling"],
        "fault_Urella_South": ["Urella_South"],
        "Sedimentary_Series": ['Cretaceous', 'Yarragadee', 'Eneabba', 'Lesueur', 'Permian']
    }
)

# %%
# Select which series are faults
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

gp.set_is_fault(
    geo_model,
    fault_groups=[
        "fault_Abrolhos_Transfer",
        "fault_Coomallo",
        "fault_Eneabba_South",
        "fault_Hypo_fault_W",
        "fault_Hypo_fault_E",
        "fault_Urella_North",
        "fault_Darling",
        "fault_Urella_South"
    ],
)


# %% 
# gp.set_fault_relation(geo_model, fr)

print(geo_model.structural_frame.fault_relations)

# %% 
# %matplotlib inline
gpv.plot_2d(geo_model, direction=['z'])

# %% 
gp.set_topography_from_random(geo_model.grid)

# %% 
gpv.plot_3d(geo_model)

# %% 
gp.compute_model(
    gempy_model=geo_model,
    engine_config= gp.data.GemPyEngineConfig(
        backend=gp.data.AvailableBackends.PYTORCH,
        dtype="float64",
    )
)

# %% 
gpv.plot_2d(geo_model, cell_number=[25])

# %% 
gpv.plot_2d(geo_model, cell_number=[25], series_n=-1, show_scalar=True)

# %% 
gpv.plot_2d(geo_model, cell_number=[12], direction=["y"], show_data=True, show_topography=True)

# %%
# sphinx_gallery_thumbnail_number = 6
gpv.plot_3d(geo_model, show_topography=True)

# %%
# Times
# -----
# 
# Fast run
# ^^^^^^^^
# 
# -  1M voxels:
# 
#    -  CPU: intel® Core™ i7-7700HQ CPU @ 2.80GHz × 8 15 s ± 1.02 s per
#       loop (mean ± std. dev. of 7 runs, 1 loop each)
#    -  GPU (4gb) not enough memmory
#    -  Ceres 1M voxels 2080 851 ms
# 
# -  250k voxels
# 
#    -  GPU 1050Ti: 3.11 s ± 11.8 ms per loop (mean ± std. dev. of 7 runs,
#       1 loop each)
#    -  CPU: intel® Core™ i7-7700HQ CPU @ 2.80GHz × 8 2.27 s ± 47.3 ms
#    -  
# 
# Fast Compile
# ^^^^^^^^^^^^
# 
# -  250k voxels
# 
#    -  GPU 1050Ti: 3.7 s ± 11.8 ms per loop (mean ± std. dev. of 7 runs,
#       1 loop each)
#    -  CPU: intel® Core™ i7-7700HQ CPU @ 2.80GHz × 8 14.2 s ± 51.1 ms per
#       loop (mean ± std. dev. of 7 runs, 1 loop each)
# 
