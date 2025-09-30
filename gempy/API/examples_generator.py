import os

import numpy as np

import gempy as gp
from gempy_engine.core.data.stack_relation_type import StackRelationType
from gempy.core.data.enumerators import ExampleModel




def generate_example_model(example_model: ExampleModel, compute_model: bool = True) -> gp.data.GeoModel:
    match example_model:
        case ExampleModel.HORIZONTAL_STRAT:
            return _generate_horizontal_stratigraphic_model(compute_model)
        case ExampleModel.ANTICLINE:
            return _generate_anticline_model(compute_model)
        case ExampleModel.ONE_FAULT:
            return _generate_one_fault_model(compute_model)
        case ExampleModel.TWO_AND_A_HALF_D:
            return _generate_2_5d_model(compute_model)
        case ExampleModel.COMBINATION:
            return _generate_combination_model(compute_model)
        case ExampleModel.ONE_FAULT_GRAVITY:
            return _generate_one_fault_model_gravity(compute_model)
        case ExampleModel.GRABEN:
            return _generate_graben_model(compute_model)
        case ExampleModel.GREENSTONE:
            return _generate_greenstone_model(compute_model)
        case _:
            raise NotImplementedError(f"Example model {example_model} not implemented.")


def _generate_2_5d_model(compute_model: bool) -> gp.data.GeoModel:
    geo_model: gp.data.GeoModel = gp.create_geomodel(
        project_name='Model1',
        extent=[0, 791, -200, 200, -582, 0],
        resolution=[50, 50, 50],
        refinement=1,
        structural_frame=gp.data.StructuralFrame.initialize_default_structure()
    )

    gp.add_surface_points(
        geo_model=geo_model,
        x=[223, 458, 612],
        y=[0.01, 0, 0],
        z=[-94, -197, -14],
        elements_names='surface1'
    )

    gp.add_orientations(
        geo_model=geo_model,
        x=[350],
        y=[1],
        z=[-300],
        elements_names=['surface1'],
        pole_vector=[[0, 0, 1]]
    )

    geo_model.update_transform(gp.data.GlobalAnisotropy.NONE)  # * Remove the auto anisotropy for this 2.5D model

    element2 = gp.data.StructuralElement(
        name='surface2',
        color=next(geo_model.structural_frame.color_generator),
        surface_points=gp.data.SurfacePointsTable.from_arrays(
            x=np.array([225, 459]),
            y=np.array([0, 0]),
            z=np.array([-269, -279]),
            names='surface2'
        ),
        orientations=gp.data.OrientationsTable.initialize_empty()
    )

    geo_model.structural_frame.structural_groups[0].append_element(element2)

    element3 = gp.data.StructuralElement(
        name='surface3',
        color=next(geo_model.structural_frame.color_generator),
        surface_points=gp.data.SurfacePointsTable.from_arrays(
            x=np.array([225, 464, 619]),
            y=np.array([0, 0, 0]),
            z=np.array([-439, -456, -433]),
            names='surface3'
        ),
        orientations=gp.data.OrientationsTable.initialize_empty()
    )

    geo_model.structural_frame.structural_groups[0].append_element(element3)

    element_fault = gp.data.StructuralElement(
        name='fault1',
        color=next(geo_model.structural_frame.color_generator),
        surface_points=gp.data.SurfacePointsTable.from_arrays(
            x=np.array([550, 650]),
            y=np.array([0, 0]),
            z=np.array([-30, -200]),
            names='fault1'
        ),
        orientations=gp.data.OrientationsTable.from_arrays(
            x=np.array([600]),
            y=np.array([0]),
            z=np.array([-100]),
            G_x=np.array([.3]),
            G_y=np.array([0]),
            G_z=np.array([.3]),
            names='fault1'
        )
    )

    group_fault = gp.data.StructuralGroup(
        name='Fault1',
        elements=[element_fault],
        structural_relation=gp.data.StackRelationType.FAULT,
        fault_relations=gp.data.FaultsRelationSpecialCase.OFFSET_ALL
    )

    geo_model.structural_frame.insert_group(0, group_fault)  # * We are placing it already in the right place so we do not need to map anything

    gp.set_topography_from_random(
        grid=geo_model.grid,
        fractal_dimension=1.9,
        d_z=np.array([-150, 0]),
        topography_resolution=np.array([50, 40])
    )

    if compute_model:
        gp.compute_model(geo_model)

    return geo_model


def _generate_horizontal_stratigraphic_model(compute_model: bool) -> gp.data.GeoModel:
    """
    Function to create a geological model of horizontally stacked layers,
    map the geological series to surfaces, and compute the geological model.
    """
    # Define the path to data
    data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'

    # Create a GeoModel instance
    geo_data = gp.create_geomodel(
        project_name='horizontal',
        extent=[0, 1000, 0, 1000, 0, 1000],
        resolution=[50, 5, 50],
        refinement=3,
        importer_helper=gp.data.ImporterHelper(
            path_to_orientations=data_path + "/data/input_data/jan_models/model1_orientations.csv",
            path_to_surface_points=data_path + "/data/input_data/jan_models/model1_surface_points.csv"
        )
    )

    # Map geological series to surfaces
    gp.map_stack_to_surfaces(
        gempy_model=geo_data,
        mapping_object={"Strat_Series": ('rock2', 'rock1')}
    )

    if compute_model:
        # Compute the geological model
        gp.compute_model(geo_data)

    return geo_data


def _generate_anticline_model(compute_model: bool) -> gp.data.GeoModel:
    """
    Function to create a geological model of an anticline structure,
    map the geological series to surfaces, and compute the geological model.
    """
    # Define the path to data
    data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
    path_to_data = data_path + "/data/input_data/jan_models/"

    # Create a GeoModel instance
    geo_data: gp.data.GeoModel = gp.create_geomodel(
        project_name='fold',
        extent=[0, 1000, 0, 1000, 0, 1000],
        refinement=5,
        importer_helper=gp.data.ImporterHelper(
            path_to_orientations=path_to_data + "model2_orientations.csv",
            path_to_surface_points=path_to_data + "model2_surface_points.csv"
        )
    )

    # Map geological series to surfaces 
    gp.map_stack_to_surfaces(
        gempy_model=geo_data,
        mapping_object={"Strat_Series": ('rock2', 'rock1')}
    )

    if compute_model:
        # Compute the geological model
        gp.compute_model(geo_data)

    return geo_data


def _generate_one_fault_model(compute_model: bool) -> gp.data.GeoModel:
    """
    Function to create a simple fault model,
    map the geological series to surfaces, and compute the geological model.
    """
    # Define the path to data
    data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
    path_to_data = data_path + "/data/input_data/jan_models/"

    # Create a GeoModel instance
    geo_data = gp.create_geomodel(
        project_name='fault',
        extent=[0, 1000, 0, 1000, 0, 1000],
        refinement=6,
        # resolution=[20, 20, 20],
        importer_helper=gp.data.ImporterHelper(
            path_to_orientations=path_to_data + "model5_orientations.csv",
            path_to_surface_points=path_to_data + "model5_surface_points.csv"
        )
    )

    # Map geological series to surfaces
    gp.map_stack_to_surfaces(
        gempy_model=geo_data,
        mapping_object={
                "Fault_Series": 'fault',
                "Strat_Series": ('rock2', 'rock1')
        }
    )

    # Define fault groups
    geo_data.structural_frame.structural_groups[0].structural_relation = StackRelationType.FAULT
    geo_data.structural_frame.fault_relations = np.array([[0, 1],
                                                          [0, 0]])
    gp.set_is_fault(
        frame=geo_data,
        fault_groups=['Fault_Series']
    )

    if compute_model:
        # Compute the geological model
        gp.compute_model(geo_data)

    return geo_data


def _generate_combination_model(compute_model: bool) -> gp.data.GeoModel:
    """
    Function to create a model with a folded domain featuring an unconformity and a fault,
    map the geological series to surfaces, and compute the geological model.
    """
    # Define the path to data
    data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
    path_to_data = data_path + "/data/input_data/jan_models/"

    # Create a GeoModel instance
    geo_data = gp.create_geomodel(
        project_name='combination',
        extent=[0, 2500, 0, 1000, 0, 1000],
        refinement=4,
        importer_helper=gp.data.ImporterHelper(
            path_to_orientations=path_to_data + "model7_orientations.csv",
            path_to_surface_points=path_to_data + "model7_surface_points.csv"
        )
    )
    geo_data.interpolation_options.evaluation_options.number_octree_levels_surface = 4

    # Map geological series to surfaces
    gp.map_stack_to_surfaces(
        gempy_model=geo_data,
        mapping_object={
                "Fault_Series" : ('fault'),
                "Strat_Series1": ('rock3'),
                "Strat_Series2": ('rock2', 'rock1'),
        }
    )

    # Define the structural relation
    geo_data.structural_frame.structural_groups[0].structural_relation = StackRelationType.FAULT
    geo_data.structural_frame.fault_relations = np.array(
        [[0, 1, 1],
         [0, 0, 0],
         [0, 0, 0]]
    )

    # Compute the geological model
    if compute_model:
        gp.compute_model(
            gempy_model=geo_data,
            engine_config=gp.data.GemPyEngineConfig(
                backend=gp.data.AvailableBackends.numpy
            )
        )

    return geo_data


def _generate_one_fault_model_gravity(compute_model):
    from gempy_engine.core.backend_tensor import BackendTensor
    from gempy.optional_dependencies import require_pandas
    pd = require_pandas()
    
    resolution = [150, 10, 150]
    extent = [0, 200, -100, 100, -100, 0]

    # %%
    # Configure GemPy for geological modeling with PyTorch backend
    BackendTensor.change_backend_gempy(engine_backend=gp.data.AvailableBackends.PYTORCH, dtype="float64")

    geo_model: gp.data.GeoModel = gp.create_geomodel(
        project_name='Fault model',
        extent=extent,
        resolution=resolution,
        structural_frame=gp.data.StructuralFrame.initialize_default_structure()
    )

    interpolation_options = geo_model.interpolation_options
    interpolation_options.mesh_extraction = False
    interpolation_options.kernel_options.range = .7
    interpolation_options.kernel_options.c_o = 3
    interpolation_options.kernel_options.compute_condition_number = True

    gp.add_surface_points(
        geo_model=geo_model,
        x=[40, 60, 120, 140],
        y=[0, 0, 0, 0],
        z=[-50, -50, -60, -60],
        elements_names=['surface1', 'surface1', 'surface1', 'surface1']
    )

    gp.add_orientations(
        geo_model=geo_model,
        x=[130],
        y=[0],
        z=[-50],
        elements_names=['surface1'],
        pole_vector=[[0, 0, 1.]]
    )

    # Define second element
    element2 = gp.data.StructuralElement(
        name='surface2',
        color=next(geo_model.structural_frame.color_generator),
        surface_points=gp.data.SurfacePointsTable.from_arrays(
            x=np.array([120]),
            y=np.array([0]),
            z=np.array([-40]),
            names='surface2'
        ),
        orientations=gp.data.OrientationsTable.initialize_empty()
    )

    # Add second element to structural frame
    geo_model.structural_frame.structural_groups[0].append_element(element2)

    # add fault
    # Calculate orientation from point values
    fault_point_1 = (80, -20)
    fault_point_2 = (110, -80)

    # calculate angle
    angle = np.arctan((fault_point_2[0] - fault_point_1[0]) / (fault_point_2[1] - fault_point_1[1]))

    x = np.cos(angle)
    z = - np.sin(angle)

    element_fault = gp.data.StructuralElement(
        name='fault1',
        color=next(geo_model.structural_frame.color_generator),
        surface_points=gp.data.SurfacePointsTable.from_arrays(
            x=np.array([fault_point_1[0], fault_point_2[0]]),
            y=np.array([0, 0]),
            z=np.array([fault_point_1[1], fault_point_2[1]]),
            names='fault1'
        ),
        orientations=gp.data.OrientationsTable.from_arrays(
            x=np.array([fault_point_1[0]]),
            y=np.array([0]),
            z=np.array([fault_point_1[1]]),
            G_x=np.array([x]),
            G_y=np.array([0]),
            G_z=np.array([z]),
            names='fault1'
        )
    )

    group_fault = gp.data.StructuralGroup(
        name='Fault1',
        elements=[element_fault],
        structural_relation=gp.data.StackRelationType.FAULT,
        fault_relations=gp.data.FaultsRelationSpecialCase.OFFSET_ALL
    )

    # Insert the fault group into the structural frame:
    geo_model.structural_frame.insert_group(0, group_fault)
    # %% md
    ## Compute model
    # %%
    geo_model.update_transform(gp.data.GlobalAnisotropy.NONE)
    
    interesting_columns = pd.DataFrame()
    # x_vals = np.arange(20, 191, 10)

    x_vals = np.linspace(20, 191, 6)
    interesting_columns['X'] = x_vals
    interesting_columns['Y'] = np.zeros_like(x_vals)

    # Configuring the data correctly is key for accurate gravity calculations.
    device_location = interesting_columns[['X', 'Y']]
    device_location['Z'] = 0  # Add a Z-coordinate

    # Set up a centered grid for geophysical calculations
    # This grid will be used for gravity gradient calculations.
    gp.set_centered_grid(
        grid=geo_model.grid,
        centers=device_location,
        resolution=np.array([75/3, 5, 150/3]),
        radius=np.array([150, 10, 300])
    )

    # Calculate the gravity gradient using GemPy
    # Gravity gradient data is critical for geophysical modeling and inversion.
    gravity_gradient = gp.calculate_gravity_gradient(geo_model.grid.centered_grid)

    densities_tensor = BackendTensor.t.array([2., 2., 3., 2.])
    densities_tensor.requires_grad = True

    # Set geophysics input for the GemPy model
    # Configuring this input is crucial for the forward gravity calculation.
    geo_model.geophysics_input = gp.data.GeophysicsInput(
        tz=BackendTensor.t.array(gravity_gradient),
        densities=densities_tensor
    )

    # %%
    # Compute the geological model with geophysical data
    # This computation integrates the geological model with gravity data.
    if compute_model:
        sol = gp.compute_model(
            gempy_model=geo_model,
            engine_config=gp.data.GemPyEngineConfig(
                backend=gp.data.AvailableBackends.PYTORCH,
                dtype='float32',
                use_gpu=True
            )
        )
        grav = - sol.gravity
        grav[0].backward()
    
    return geo_model


def _generate_graben_model(compute_model: bool) -> gp.data.GeoModel:
    # Data path is in root/examples/data
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # data_path = os.path.join(script_dir, '.../examples/')
    # n_model = 7
    # https: // github.com / gempy - project / gempy / blob / 279
    # bbe904283e16320c54d868fe74be873177cca / examples / data / input_data / lisa_models / interfaces7.csv
    # csv_ = data_path + "/data/input_data/lisa_models/foliations" + n_model + ".csv"

    data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
    path_to_data = data_path + "/data/input_data/lisa_models/"
    
    geo_data: gp.data.GeoModel = gp.create_geomodel(
        project_name="Graben",
        extent=[0, 2000, 0, 2000, 0, 1600],
        resolution=[50, 50, 50],
        refinement=6,  # * For this model is better not to use octrees because we want to see what is happening in the scalar fields
        importer_helper=gp.data.ImporterHelper(
            path_to_orientations= path_to_data + "foliations7.csv",
            path_to_surface_points= path_to_data + "interfaces7.csv"
        )
    )

    gp.map_stack_to_surfaces(
        gempy_model=geo_data,
        mapping_object={
                "Fault_1"     : 'Fault_1', "Fault_2": 'Fault_2',
                "Strat_Series": ('Sandstone', 'Siltstone', 'Shale', 'Sandstone_2', 'Schist', 'Gneiss')
        },
    )

    gp.set_is_fault(geo_data, ['Fault_1', 'Fault_2'])
    if compute_model:
        sol = gp.compute_model(gempy_model=geo_data)

    return geo_data


def _generate_greenstone_model(compute_model: bool) -> gp.data.GeoModel:
    test_dir = os.path.dirname(os.path.abspath(__file__))

    # Build the path relative to the test file location
    path = os.path.join(test_dir, '..', '..', 'examples', 'data', 'gempy_models', 'Greenstone.gempy')
    with open(path, 'rb') as f:
        binary_file = f.read()

    from gempy.modules.serialization.save_load import _load_model_from_bytes
    geo_model: gp.data.GeoModel = _load_model_from_bytes(binary_file)
    
    if compute_model:
        sol = gp.compute_model(
            gempy_model=geo_model,
            engine_config=gp.data.GemPyEngineConfig(
                backend=gp.data.AvailableBackends.numpy,
                dtype='float32'
            )
    )
    return geo_model
        

