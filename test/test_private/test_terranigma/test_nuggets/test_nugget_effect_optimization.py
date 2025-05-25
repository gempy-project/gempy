import os

import dotenv
import numpy as np
import torch
import pyro
import time
import gempy as gp
import gempy_viewer as gpv

from ._aux_func import process_file, initialize_geo_model

dotenv.load_dotenv()

# %%
# Config
seed = 123456
torch.manual_seed(seed)
pyro.set_rng_seed(seed)

# %%
# Start the timer for benchmarking purposes
start_time = time.time()

# %%
# Load necessary configuration and paths from environment variables
path = os.getenv("PATH_TO_NUGGET_TEST_MODEL")


def test_optimize_nugget_effect():
    # Initialize lists to store structural elements for the geological model
    structural_elements = []
    global_extent = None
    color_gen = gp.data.ColorsGenerator()

    for filename in os.listdir(path):
        base, ext = os.path.splitext(filename)
        if ext == '.nc':
            structural_element, global_extent = process_file(os.path.join(path, filename), global_extent, color_gen)
            structural_elements.append(structural_element)

    import xarray as xr
    geo_model: gp.data.GeoModel = initialize_geo_model(
        structural_elements=structural_elements,
        extent=(np.array(global_extent)),
        topography=(xr.open_dataset(os.path.join(path, "Topography.nc")))
    )

    if False:
        gpv.plot_3d(geo_model, show_data=True, image=True)

    geo_model.interpolation_options.cache_mode = gp.data.InterpolationOptions.CacheMode.NO_CACHE

    if True:
        gp.API.compute_API.optimize_and_compute(
            geo_model=geo_model,
            engine_config=gp.data.GemPyEngineConfig(
                backend=gp.data.AvailableBackends.PYTORCH,
            ),
            max_epochs=100,
            convergence_criteria=1e5
        )

        print(f"Final cond number: {geo_model.interpolation_options.kernel_options.condition_number}")
        nugget_effect = geo_model.taped_interpolation_input.surface_points.nugget_effect_scalar.detach().numpy()
        
    else:
        gp.compute_model(
            gempy_model=geo_model,
            engine_config=gp.data.GemPyEngineConfig(
                backend=gp.data.AvailableBackends.PYTORCH,
            ),
            validate_serialization=False
        )


        nugget_effect = geo_model.taped_interpolation_input.surface_points.nugget_effect_scalar.detach().numpy()

    if plot_evaluation := True:
        import matplotlib.pyplot as plt

        plt.hist(nugget_effect, bins=50, color='black', alpha=0.7, log=True)
        plt.xlabel('Eigenvalue')
        plt.ylabel('Frequency')
        plt.title('Histogram of Eigenvalues (nugget-grad)')
        plt.show()

    if plot_result := True:
        import gempy_viewer as gpv
        import pyvista as pv

        gempy_vista = gpv.plot_3d(
            model=geo_model,
            show=False,
            show_boundaries=True,
            show_topography=False,
            kwargs_plot_structured_grid={'opacity': 0.3}
        )

        # Create a point cloud mesh
        surface_points_xyz = geo_model.surface_points_copy.df[['X', 'Y', 'Z']].to_numpy()

        point_cloud = pv.PolyData(surface_points_xyz[0:])
        point_cloud['values'] = nugget_effect

        gempy_vista.p.add_mesh(
            point_cloud,
            scalars='values',
            cmap='inferno',
            point_size=25,
        )

        gempy_vista.p.show()
