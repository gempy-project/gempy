import numpy as np

import gempy as gp
import time

PLOT = True
DENSE_RESOLUTION = [20, 20, 20]


def test_compute_time_dense_dense():
    geo_model: gp.data.GeoModel = _setup_model()
    geo_model.interpolation_options.evaluation_options.mesh_extraction = True

    geo_model.grid.active_grids = gp.data.Grid.GridTypes.DENSE
    start_time = time.perf_counter()
    gp.compute_model(geo_model)
    print(f"Computing model on grids: {geo_model.grid.active_grids}")
    end_time = time.perf_counter()
    computation_time_dense_I = end_time - start_time

    start_time = time.perf_counter()
    gp.compute_model(geo_model)
    print(f"Computing model on grids: {geo_model.grid.active_grids}")
    end_time = time.perf_counter()
    computation_time_dense_II = end_time - start_time

    print(f"Computation only model dense grid 125*50*50: {computation_time_dense_I:.2f} seconds")
    print(f"Computation only model dense grid 125*50*50: {computation_time_dense_II:.2f} seconds")
    
    # Assert that it is not too different
    assert abs(computation_time_dense_I - computation_time_dense_II) < 0.2


def test_compute_time_topo_dense_grid():
    geo_model: gp.data.GeoModel = _setup_model()

    geo_model.grid.active_grids = gp.data.Grid.GridTypes.DENSE
    start_time = time.perf_counter()
    gp.compute_model(geo_model)
    print(f"Computing model on grids: {geo_model.grid.active_grids}")
    end_time = time.perf_counter()
    computation_time_dense = end_time - start_time

    # Compute a solution for the model
    geo_model.grid.active_grids = gp.data.Grid.GridTypes.TOPOGRAPHY
    start_time = time.perf_counter()
    gp.compute_model(geo_model)
    print(f"Computing model on grids: {geo_model.grid.active_grids}")
    end_time = time.perf_counter()
    computation_time_topo = end_time - start_time

    # Recompute model as a new grid was added
    geo_model.grid.active_grids = gp.data.Grid.GridTypes.TOPOGRAPHY | gp.data.Grid.GridTypes.DENSE
    start_time = time.perf_counter()
    gp.compute_model(geo_model)
    print(f"Computing model on grids: {geo_model.grid.active_grids}")
    end_time = time.perf_counter()
    computation_time_topo_dense = end_time - start_time

    print(f"Computation only model dense grid 125*50*50: {computation_time_dense:.2f} seconds")
    print(f"Computation time with topography 125*50: {computation_time_topo:.2f} seconds")
    print(f"Computation time with topography and dense grid 125*50*50: {computation_time_topo_dense:.2f} seconds")
    
    # Assert that dense takes longer than topo and that the sum of both is close
    assert computation_time_topo_dense > computation_time_dense
    

def test_compute_time_custom_dense_grid():
    geo_model: gp.data.GeoModel = _setup_model()

    # numpy array with random coordinates within the extent of the model
    custom_coordinates = np.random.uniform(
        low=geo_model.grid.extent[:3],
        high=geo_model.grid.extent[3:],
        size=(1000, 3)
    )

    start_time = time.perf_counter()
    gp.compute_model_at(geo_model, custom_coordinates)
    end_time = time.perf_counter()
    computation_time_at = end_time - start_time

    print(f"Computation compute_at with 1000 custom points: {computation_time_at:.2f} seconds")


def _setup_model():
    # Define the path to data
    data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
    path_to_data = data_path + "/data/input_data/jan_models/"
    # Create a GeoModel instance
    geo_model = gp.create_geomodel(
        project_name='EGU_example',
        extent=[0, 2500, 0, 1000, 0, 1000],
        resolution=DENSE_RESOLUTION,
        importer_helper=gp.data.ImporterHelper(
            path_to_orientations=path_to_data + "model7_orientations.csv",
            path_to_surface_points=path_to_data + "model7_surface_points.csv"
        )
    )
    # Map geological series to surfaces
    gp.map_stack_to_surfaces(
        gempy_model=geo_model,
        mapping_object={
                "Fault_Series" : ('fault'),
                "Strat_Series1": ('rock3'),
                "Strat_Series2": ('rock2', 'rock1'),
        }
    )
    # Define youngest structural group as fault
    gp.set_is_fault(geo_model, ["Fault_Series"])
    # Setting a randomly generated topography
    gp.set_topography_from_random(
        grid=geo_model.grid,
        fractal_dimension=2,
        d_z=np.array([700, 950]),
        topography_resolution=np.array([125, 50])
    )
    return geo_model
