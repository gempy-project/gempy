"""
GemPy Models - Some More Complex Examples
-----------------------------------------

A set of more complex examples

"""

# %%
# Importing gempy
import gempy as gp
import gempy_viewer as gpv

# Aux imports
import numpy as np
import os

# %%
# Choose a model and load the corresponding data set in the line below
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Model 1: Discordant layering
# 
# Model 2: Anticlinal layering
# 
# Model 3: Parallel steep dipping layers
# 
# Model 5: Normal faulting
# 
# Model 6: Synclinal layering
# 
# Model 7: Graben structure with two faults
# 
# Model 8: Similar to Model 5
# 
# Model 9: Parallel flat dipping layers
# 

# %%

data_path = os.path.abspath('../..')


# %%
def create_example(name_model, plot_section=True):
    """
    Create an inter_data from one of the examples data_set
    
    Attr:
        name_model (str): name of the model that you want to generate. It has to be in ['Model 1' ,'Model 2', 'Model 3', 'Model 4','Model 5' 'Model 6','Model 7',
                          'Model 8', 'Model 9']
        save_pickle (bool, str): Save to a pickle the interp_data object. You can pass the path as a string otherwse
                                the default name will be given
        plot_section (bool)
    
    """
    name_list = np.array(['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5', 'Model 6', 'Model 7', 'Model 8', 'Model 9'])
    assert name_model in name_list, 'Name model must be in the following list: ' + str(name_list)
    # Extract number of the model
    n_model = name_model[-1]

    # # Load right gempy geodata
    geo_data: gp.data.GeoModel = gp.create_geomodel(
        project_name=name_model,
        extent=[0, 2000, 0, 2000, 0, 1600],
        resolution=[50, 50, 50],
        refinement=6,  # * For this model is better not to use octrees because we want to see what is happening in the scalar fields
        importer_helper=gp.data.ImporterHelper(
            path_to_orientations=data_path + "/data/input_data/lisa_models/foliations" + n_model + ".csv",
            path_to_surface_points=data_path + "/data/input_data/lisa_models/interfaces" + n_model + ".csv",
        )
    )

    # Set the right sequential pile
    subset_list_1 = np.array(['Model 1'])
    subset_list_2 = np.array(['Model 5', 'Model 8'])
    subset_list_3 = np.array(['Model 2', 'Model 3', 'Model 9', 'Model 6'])
    subset_list_4 = np.array(['Model 7'])
    
    ### Model 1 - Discordant layering ###
    if name_model in subset_list_1:
        gp.map_stack_to_surfaces(
            gempy_model=geo_data,
            mapping_object={
                "Strat_Series_1": ('Sandstone', 'Siltstone', 'Shale'),
                "Strat_Series_2": ('Sandstone2', 'Siltstone2', 'Shale2')
            },
        )
    ### Model 5 - One normal Fault ###
    ### Model 8 - ###
    elif name_model in subset_list_2:
        gp.map_stack_to_surfaces(
            gempy_model=geo_data,
            mapping_object={
            "Fault_Series": 'Main_Fault',
            "Strat_Series": ('Sandstone', 'Siltstone', 'Shale', 'Sandstone_2', 'Schist', 'Gneiss')
            },
        )
        
        gp.set_is_fault(geo_data, ['Fault_Series'])
        
    elif name_model in subset_list_3:
        ### Model 2 - Aufwölbung (durch Salzstock?) ###
        ### Model 3+9 - Parallele NNE Schichtung ohne Verwerfung ###
        ### Model 6 - Mulde ###
        gp.map_stack_to_surfaces(
            gempy_model=geo_data,
            mapping_object={
            "Strat_Series": ('Sandstone', 'Siltstone', 'Shale', 'Sandstone_2', 'Schist', 'Gneiss')
            },
        )
    elif name_model in subset_list_4:
        ### Model 7 - Graben ###
        gp.map_stack_to_surfaces(
            gempy_model=geo_data,
            mapping_object={
            "Fault_1": 'Fault_1', "Fault_2": 'Fault_2',
            "Strat_Series": ('Sandstone', 'Siltstone', 'Shale', 'Sandstone_2', 'Schist', 'Gneiss')
            },
        )
        
        gp.set_is_fault(geo_data, ['Fault_1', 'Fault_2'])
    else:
        print('You would never reach this point. Look for the bug')

    # Interpolation and Computation
    sol = gp.compute_model(
        gempy_model=geo_data,
        engine_config=gp.data.GemPyEngineConfig(backend=gp.data.AvailableBackends.PYTORCH)
    )

    if plot_section is True:
        # 2D Plot
        gpv.plot_2d(geo_data, cell_number=['mid'], direction='y', show_data=True)
        gpv.plot_3d(geo_data, image=False)

    return geo_data.structural_frame


# %%
# Discordant layering 1
# =====================

create_example('Model 1')

# %%
# Discordant layering 2
# =====================

create_example('Model 2')

# %%
# Discordant layering 3
# =====================

create_example('Model 3')

# %%
# One normal Fault 1
# ==================

create_example('Model 5')

# %%
# Fold
# ====

create_example('Model 6')

# %%
# Graben
# ======

create_example('Model 7')

# %%
#  One normal Fault 2
# ===================

create_example('Model 8')

# %%
# Horizontal tilted
# =================

create_example('Model 9')

# %%

# generate_all_models(['Model 1', 'Model 2', 'Model 3',
#                     'Model 5', 'Model 6', 'Model 7',
#                     'Model 8', 'Model 9'])
