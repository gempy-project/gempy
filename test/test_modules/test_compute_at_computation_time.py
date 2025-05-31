import numpy as np

import gempy as gp
import time

PLOT = True


def test_compute_at_computation_time():

    # Define the path to data
    data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
    path_to_data = data_path + "/data/input_data/jan_models/"

    # Create a GeoModel instance
    geo_model = gp.create_geomodel(
        project_name='EGU_example',
        extent=[0, 2500, 0, 1000, 0, 1000],
        resolution=[125, 50, 50],
        importer_helper=gp.data.ImporterHelper(
            path_to_orientations=path_to_data + "model7_orientations.csv",
            path_to_surface_points=path_to_data + "model7_surface_points.csv"
        )
    )

    # Map geological series to surfaces
    gp.map_stack_to_surfaces(
        gempy_model=geo_model,
        mapping_object={
            "Fault_Series": ('fault'),
            "Strat_Series1": ('rock3'),
            "Strat_Series2": ('rock2', 'rock1'),
        }
    )

    # Define youngest structural group as fault
    gp.set_is_fault(geo_model, ["Fault_Series"])

    # Compute a solution for the model
    start_time = time.perf_counter()
    gp.compute_model(geo_model)
    end_time = time.perf_counter()
    computation_time_model = end_time - start_time

    # Setting a randomly generated topography
    gp.set_topography_from_random(
        grid=geo_model.grid,
        fractal_dimension=2,
        d_z=np.array([700, 950]),
        topography_resolution=np.array([125, 50])
    )

    # Recompute model as a new grid was added
    start_time = time.perf_counter()
    gp.compute_model(geo_model)
    end_time = time.perf_counter()
    computation_time_topo = end_time - start_time

    print(f"Computation time without topography: {computation_time_model:.2f} seconds")
    print(f"Computation time with topography: {computation_time_topo:.2f} seconds")
