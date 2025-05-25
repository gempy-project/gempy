import os

import dotenv
import numpy as np
import torch
import pyro
import time
import gempy as gp
import gempy_viewer as gpv
from gempy_engine.core.backend_tensor import BackendTensor

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


    BackendTensor.change_backend_gempy(
        engine_backend=gp.data.AvailableBackends.PYTORCH,
        dtype="float64"
    )

    import xarray as xr
    geo_model: gp.data.GeoModel = initialize_geo_model(
        structural_elements=structural_elements,
        extent=(np.array(global_extent)),
        topography=(xr.open_dataset(os.path.join(path, "Topography.nc")))
    )
    
    gpv.plot_3d(geo_model, show_data=True, image=True)


