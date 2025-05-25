import dotenv
import os

from typing import Optional

import numpy as np

import gempy_engine
from gempy_engine.core.backend_tensor import BackendTensor
from gempy.API.gp2_gp3_compatibility.gp3_to_gp2_input import gempy3_to_gempy2
from gempy_engine.config import AvailableBackends
from gempy_engine.core.data import Solutions
from gempy_engine.core.data.interpolation_input import InterpolationInput
from .grid_API import set_custom_grid
from ..core.data.gempy_engine_config import GemPyEngineConfig
from ..core.data.geo_model import GeoModel
from ..modules.data_manipulation import interpolation_input_from_structural_frame
from ..modules.optimize_nuggets import nugget_optimizer
from ..optional_dependencies import require_gempy_legacy

dotenv.load_dotenv()


def compute_model(gempy_model: GeoModel, engine_config: Optional[GemPyEngineConfig] = None,
                  **kwargs) -> Solutions:
    """
    Compute the geological model given the provided GemPy model.

    Args:
        gempy_model (GeoModel): The GemPy model to compute.
        engine_config (Optional[GemPyEngineConfig]): Configuration for the computational engine. Defaults to None, in which case a default configuration will be used.

    Raises:
        ValueError: If the provided backend in the engine_config is not supported.

    Returns:
        Solutions: The computed geological model.
    """
    engine_config = engine_config or GemPyEngineConfig(use_gpu=False)

    match engine_config.backend:
        case AvailableBackends.numpy | AvailableBackends.PYTORCH:

            BackendTensor.change_backend_gempy(
                engine_backend=engine_config.backend,
                use_gpu=engine_config.use_gpu,
                dtype=engine_config.dtype
            )

            # TODO: To decide what to do with this.
            interpolation_input = interpolation_input_from_structural_frame(gempy_model)
            gempy_model.taped_interpolation_input = interpolation_input  # * This is used for gradient tape

            gempy_model.solutions = gempy_engine.compute_model(
                interpolation_input=interpolation_input,
                options=gempy_model.interpolation_options,
                data_descriptor=gempy_model.input_data_descriptor,
                geophysics_input=gempy_model.geophysics_input,
            )

        case AvailableBackends.aesara | AvailableBackends.legacy:
            gempy_model.legacy_model = _legacy_compute_model(gempy_model)
        case _:
            raise ValueError(f'Backend {engine_config} not supported')

    if os.getenv("VALIDATE_SERIALIZATION", False) and kwargs.get("validate_serialization", True):
        from ..modules.serialization.save_load import save_model
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w+', delete=True) as tmp:
            save_model(model=gempy_model, path=tmp.name, validate_serialization=True)

    return gempy_model.solutions


def compute_model_at(gempy_model: GeoModel, at: np.ndarray,
                     engine_config: Optional[GemPyEngineConfig] = None) -> np.ndarray:
    """
    Compute the geological model at specific coordinates.
    
    Note: This function sets a custom grid and computes the model so be wary of side effects.

    Args:
        gempy_model (GeoModel): The GemPy model to compute.
        at (np.ndarray): The coordinates at which to compute the model.
        engine_config (Optional[GemPyEngineConfig], optional): Configuration for the computational engine. Defaults to None, in which case a default configuration will be used.

    Returns:
        np.ndarray: The computed geological model at the specified coordinates.
    """
    set_custom_grid(
        grid=gempy_model.grid,
        xyz_coord=at
    )

    sol = compute_model(gempy_model, engine_config, validate_serialization=True)
    return sol.raw_arrays.custom


def optimize_and_compute(geo_model: GeoModel, engine_config: GemPyEngineConfig, max_epochs: int = 10,
                         convergence_criteria: float = 1e5):
    if engine_config.backend != AvailableBackends.PYTORCH:
        raise ValueError(f'Only PyTorch backend is supported for optimization. Received {engine_config.backend}')

    geo_model = nugget_optimizer(
        convergence_criteria=convergence_criteria,
        engine_config=engine_config,
        geo_model=geo_model,
        max_epochs=max_epochs
    )

    geo_model.solutions = gempy_engine.compute_model(
        interpolation_input=geo_model.taped_interpolation_input,
        options=geo_model.interpolation_options,
        data_descriptor=geo_model.input_data_descriptor,
        geophysics_input=geo_model.geophysics_input,
    )
    return geo_model.solutions


def _legacy_compute_model(gempy_model: GeoModel) -> 'gempy_legacy.Project':
    gpl = require_gempy_legacy()
    legacy_model: gpl.Project = gempy3_to_gempy2(gempy_model)
    gpl.set_interpolator(legacy_model)
    gpl.compute_model(legacy_model)
    return legacy_model
