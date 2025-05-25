import gempy_engine
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.interpolation_input import InterpolationInput

from ...modules.data_manipulation import interpolation_input_from_structural_frame


def nugget_optimizer(convergence_criteria, engine_config, geo_model, max_epochs):
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
    nugget_effect_scalar.requires_grad = True
    optimizer = torch.optim.Adam(
        params=[nugget_effect_scalar],
        lr=0.01,
    )
    # Optimization loop
    geo_model.interpolation_options.kernel_options.optimizing_condition_number = True

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
                epoch=epoch
        ):
            break
        previous_condition_number = geo_model.interpolation_options.kernel_options.condition_number
        continue
    geo_model.interpolation_options.kernel_options.optimizing_condition_number = False


def _check_convergence_criterion(conditional_number: float, condition_number_old: float, conditional_number_target: float = 1e5, epoch: int = 0):
    import torch
    reached_conditional_target = conditional_number < conditional_number_target
    if reached_conditional_target == False and epoch > 10:
        condition_number_change = torch.abs(conditional_number - condition_number_old) / condition_number_old
        if condition_number_change < 0.01:
            reached_conditional_target = True
    return reached_conditional_target
