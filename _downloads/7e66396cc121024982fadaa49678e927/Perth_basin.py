"""
Perth Basin, Western Australia
=================================

Regional stratigraphy and major normal faults of the northern Perth Basin

This model covers the northern Perth Basin, Western Australia, including eight major
normal faults (among them the Darling Fault) offsetting a Permian-to-Cretaceous
sedimentary sequence.
"""
import os

# Importing GemPy
import gempy as gp
import gempy_viewer as gpv

# Importing auxiliary libraries
import matplotlib

matplotlib.rcParams['figure.figsize'] = (20.0, 10.0)

# %%
cwd = os.getcwd()
if 'examples' not in cwd:
    data_path = os.getcwd() + '/examples'
else:
    data_path = cwd + '/../..'

# %%
geo_model = gp.create_geomodel(
    project_name='Perth_Basin',
    extent=[337000, 400000, 6640000, 6710000, -18000, 1000],
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
    [
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

print(geo_model.structural_frame.fault_relations)

# %%
gpv.plot_2d(geo_model, direction=['z'])

# %% 
gp.set_topography_from_random(geo_model.grid)

# %% 
gpv.plot_3d(geo_model)

# %% 
gp.compute_model(
    gempy_model=geo_model,
    engine_config=gp.data.GemPyEngineConfig(
        backend=gp.data.AvailableBackends.PYTORCH,
        dtype="float64",
    )
)

# %% 
gpv.plot_2d(geo_model, cell_number="mid")

# %% 
gpv.plot_2d(geo_model, cell_number="mid", series_n=-1, show_scalar=True)

# %% 
gpv.plot_2d(geo_model, cell_number=[12], direction=["y"], show_data=True, show_topography=True)

# %%
# sphinx_gallery_thumbnail_number = 6
gpv.plot_3d(geo_model, show_lith=True, show_boundaries=True, show_topography=True, ve=None)