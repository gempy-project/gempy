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
from ..modules.data_manipulation.engine_factory import interpolation_input_from_structural_frame
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


def optimize_and_compute(geo_model: GeoModel, engine_config: GemPyEngineConfig, max_epochs: int = 10,
                         convergence_criteria: float = 1e5):
    if engine_config.backend != AvailableBackends.PYTORCH:
        raise ValueError(f'Only PyTorch backend is supported for optimization. Received {engine_config.backend}')

    BackendTensor.change_backend_gempy(
        engine_backend=engine_config.backend,
        use_gpu=engine_config.use_gpu,
        dtype=engine_config.dtype
    )

    import torch
    from gempy_engine.core.data.continue_epoch import ContinueEpoch
    interpolation_input: InterpolationInput = interpolation_input_from_structural_frame(geo_model)

    geo_model.taped_interpolation_input = interpolation_input

    nugget_effect_scalar: torch.Tensor = geo_model.taped_interpolation_input.surface_points.nugget_effect_scalar

    optimizer = torch.optim.Adam(
        params=[nugget_effect_scalar],
        lr=0.01,
    )

    # Optimization loop
    geo_model.interpolation_options.kernel_options.optimizing_condition_number = True

    def _check_convergence_criterion(conditional_number: float, condition_number_old: float, conditional_number_target: float = 1e5):
        reached_conditional_target = conditional_number < conditional_number_target
        if reached_conditional_target == False and epoch > 10:
            condition_number_change = torch.abs(conditional_number - condition_number_old) / condition_number_old
            if condition_number_change < 0.01:
                reached_conditional_target = True
        return reached_conditional_target

    previous_condition_number = 0
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        try:
            # geo_model.taped_interpolation_input.grid = geo_model.interpolation_input_copy.grid

            gempy_engine.compute_model(
                interpolation_input=geo_model.taped_interpolation_input,
                options=geo_model.interpolation_options,
                data_descriptor=geo_model.input_data_descriptor,
                geophysics_input=geo_model.geophysics_input,
            )
        except ContinueEpoch:
            # Get absolute values of gradients
            grad_magnitudes = torch.abs(nugget_effect_scalar.grad)

            # Get indices of the 10 largest gradients
            grad_magnitudes.size

            # * This ignores 90 percent of the gradients
            # To int
            n_values = int(grad_magnitudes.size()[0] * 0.9)
            _, indices = torch.topk(grad_magnitudes, n_values, largest=False)

            # Zero out gradients that are not in the top 10
            mask = torch.ones_like(nugget_effect_scalar.grad)
            mask[indices] = 0
            nugget_effect_scalar.grad *= mask

            # Update the vector
            optimizer.step()
            nugget_effect_scalar.data = nugget_effect_scalar.data.clamp_(min=1e-7)  # Replace negative values with 0

            # optimizer.zero_grad()
        # Monitor progress
        if epoch % 1 == 0:
            # print(f"Epoch {epoch}: Condition Number = {condition_number.item()}")
            print(f"Epoch {epoch}")

        if _check_convergence_criterion(
                conditional_number=geo_model.interpolation_options.kernel_options.condition_number,
                condition_number_old=previous_condition_number,
                conditional_number_target=convergence_criteria,
        ):
            break
        previous_condition_number = geo_model.interpolation_options.kernel_options.condition_number
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
