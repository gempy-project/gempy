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
        resolution=[20, 20, 20],
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
    geo_data.structural_frame.fault_relations = np.array([[0, 1], [0, 0]])
    gp.set_is_fault(
        frame=geo_data,
        fault_groups=['Fault_Series']
    )
    
    if compute_model:
        # Compute the geological model
        gp.compute_model(geo_data)

    return geo_data


def _generate_combination_model(compute_model:bool) -> gp.data.GeoModel:
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
        resolution=[125, 50, 50],
        importer_helper=gp.data.ImporterHelper(
            path_to_orientations=path_to_data + "model7_orientations.csv",
            path_to_surface_points=path_to_data + "model7_surface_points.csv"
        )
    )

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
        gp.compute_model(geo_data)

    return geo_data
