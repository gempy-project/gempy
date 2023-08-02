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
        case _:
            raise NotImplementedError(f"Example model {example_model} not implemented.")


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
        importer_helper=gp.ImporterHelper(
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
        resolution=[50, 5, 50],
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
        resolution=[20, 5, 20],
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