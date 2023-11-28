from typing import Optional

import numpy as np

import gempy_engine
from gempy_engine.core.backend_tensor import BackendTensor
from gempy.API.gp2_gp3_compatibility.gp3_to_gp2_input import gempy3_to_gempy2
from gempy_engine.config import AvailableBackends
from gempy_engine.core.data import Solutions
from .grid_API import set_custom_grid
from ..core.data.gempy_engine_config import GemPyEngineConfig
from ..core.data.geo_model import GeoModel
from ..optional_dependencies import require_gempy_legacy


def compute_model(gempy_model: GeoModel, engine_config: Optional[GemPyEngineConfig] = None) -> Solutions:
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
    engine_config = engine_config or GemPyEngineConfig(
        backend=AvailableBackends.numpy,
        use_gpu=False,
    )

    match engine_config.backend:
        case AvailableBackends.numpy | AvailableBackends.PYTORCH:

            BackendTensor.change_backend_gempy(
                engine_backend=engine_config.backend,
                use_gpu=engine_config.use_gpu,
                dtype=engine_config.dtype
            )

            # TODO: To decide what to do with this.
            interpolation_input = gempy_model.interpolation_input
            gempy_model.taped_interpolation_input = interpolation_input

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

    sol = compute_model(gempy_model, engine_config)
    return sol.raw_arrays.custom


def optimize_and_compute(geo_model: GeoModel, engine_config: GemPyEngineConfig, max_epochs: int = 10):
    if engine_config.backend != AvailableBackends.PYTORCH:
        raise ValueError(f'Only PyTorch backend is supported for optimization. Received {engine_config.backend}')

    BackendTensor.change_backend_gempy(
        engine_backend=engine_config.backend,
        use_gpu=engine_config.use_gpu,
        dtype=engine_config.dtype
    )

    import torch
    from gempy_engine.core.data.continue_epoch import ContinueEpoch

    interpolation_input = geo_model.interpolation_input

    geo_model.taped_interpolation_input = interpolation_input

    nugget_effect_scalar = geo_model.taped_interpolation_input.surface_points.nugget_effect_scalar

    optimizer = torch.optim.Adam(
        params=[nugget_effect_scalar],
        lr=0.01,
    )

    # Optimization loop
    geo_model.interpolation_options.kernel_options.optimizing_condition_number = True

    def _check_convergence_criterion(conditional_number: float, conditional_number_target: float = 1e4):
        return conditional_number < conditional_number_target

    for epoch in range(max_epochs):
        optimizer.zero_grad()
        try:
            geo_model.taped_interpolation_input.grid = geo_model.interpolation_input.grid

            gempy_engine.compute_model(
                interpolation_input=geo_model.taped_interpolation_input,
                options=geo_model.interpolation_options,
                data_descriptor=geo_model.input_data_descriptor,
                geophysics_input=geo_model.geophysics_input,
            )
        except ContinueEpoch:
            # Scale the gradients
            # foo = 1000
            # nugget_effect_scalar.grad *= nugget_effect_scalar * foo
            
            # Update the vector
            optimizer.step()
            nugget_effect_scalar.data = nugget_effect_scalar.data.clamp_(min=1e-7)  # Replace negative values with 0
        # Monitor progress
        if epoch % 1 == 0:
            # print(f"Epoch {epoch}: Condition Number = {condition_number.item()}")
            print(f"Epoch {epoch}")

        if _check_convergence_criterion(geo_model.interpolation_options.kernel_options.condition_number):
            break

        continue

    geo_model.interpolation_options.kernel_options.optimizing_condition_number = False

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
