import os

import dotenv
import numpy as np
import torch
import time
import gempy as gp

from ._aux_func import process_file, initialize_geo_model

dotenv.load_dotenv()

# %%
# Config
seed = 123456
torch.manual_seed(seed)

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

    geo_model.interpolation_options.cache_mode = gp.data.InterpolationOptions.CacheMode.NO_CACHE
    geo_model.interpolation_options.kernel_options.range = 0.5

    gp.API.compute_API.optimize_nuggets(
        geo_model=geo_model,
        engine_config=gp.data.GemPyEngineConfig(
            backend=gp.data.AvailableBackends.PYTORCH,
        ),
        max_epochs=100,
        convergence_criteria=1e5,
        only_groups=[geo_model.structural_frame.get_group_by_name("Red")]
    )

    gp.compute_model(
        gempy_model=geo_model,
        engine_config=gp.data.GemPyEngineConfig(
            backend=gp.data.AvailableBackends.PYTORCH,
        ),
        validate_serialization=True
    )
    
    print(f"Final cond number: {geo_model.interpolation_options.kernel_options.condition_number}")
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

        image = True
        gempy_vista = gpv.plot_3d(
            model=geo_model,
            show=False,
            show_boundaries=True,
            show_topography=False,
            image=image,
            kwargs_plot_structured_grid={'opacity': 0.3}
        )

        # Create a point cloud mesh
        if image == False:
            surface_points_xyz = geo_model.surface_points_copy.df[['X', 'Y', 'Z']].to_numpy()

            point_cloud = pv.PolyData(surface_points_xyz[0:])
            point_cloud['values'] = nugget_effect

            gempy_vista.p.add_mesh(
                point_cloud,
                scalars='values',
                cmap='inferno',
                point_size=25,
            )

            if False:
                ori_cloud = pv.PolyData(geo_model.orientations_copy.df[['X', 'Y', 'Z']].to_numpy())
                ori_cloud['values2'] = geo_model.taped_interpolation_input.orientations.nugget_effect_grad.detach().numpy()
                
                gempy_vista.p.add_mesh(
                    ori_cloud,
                    scalars='values2',
                    cmap='viridis',
                    point_size=20,
                )

            gempy_vista.p.show()
