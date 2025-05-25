from torch.nn.utils import clip_grad_norm_

import gempy_engine
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.continue_epoch import ContinueEpoch
from gempy_engine.core.data.interpolation_input import InterpolationInput

from ...modules.data_manipulation import interpolation_input_from_structural_frame
from ...core.data import GeoModel, GemPyEngineConfig

try:
    import torch
except ImportError:
    torch = None


def nugget_optimizer(
        target_cond_num: float,
        engine_cfg: GemPyEngineConfig,
        model: GeoModel,
        max_epochs: int,
        lr: float = .1,
        patience: int = 10,
        min_impr: float = 0.01,
) -> GeoModel:
    """
    Optimize the nugget effect scalar to achieve a target condition number.
    Returns the final nugget effect value.
    """
    # 1) Backend setup (ideally done once, not every call)
    BackendTensor.change_backend_gempy(
        engine_backend=engine_cfg.backend,
        use_gpu=engine_cfg.use_gpu,
        dtype=engine_cfg.dtype,
    )

    # 2) Prepare data

    interp_in: InterpolationInput = interpolation_input_from_structural_frame(model)
    model.taped_interpolation_input = interp_in
    nugget: torch.Tensor = interp_in.surface_points.nugget_effect_scalar
    nugget_ori: torch.Tensor = interp_in.orientations.nugget_effect_grad

    nugget.requires_grad_(True)
    nugget_ori.requires_grad_(True)

    opt = torch.optim.Adam(params=[nugget, nugget_ori], lr=lr)

    model.interpolation_options.kernel_options.optimizing_condition_number = True

    prev_cond = float('inf')
    for epoch in range(max_epochs):
        opt.zero_grad()
        try:
            gempy_engine.compute_model(
                interpolation_input=model.taped_interpolation_input,
                options=model.interpolation_options,
                data_descriptor=model.input_data_descriptor,
                geophysics_input=model.geophysics_input,
            )
        except ContinueEpoch:
            # Keep only top 10% gradients
            if epoch % 5 == 0:
                mask_sp = _mask_iqr(nugget.grad.abs().view(-1))
                mask_ori = _mask_iqr(nugget_ori.grad.abs().view(-1))

                # print the indices of mask
                print(f"Outliers sp: {torch.nonzero(mask_sp)}")
                print(f"Outliers ori: {torch.nonzero(mask_ori)}")

            _apply_outlier_gradients(tensor=nugget, mask=mask_sp)
            _apply_outlier_gradients(tensor=nugget_ori, mask=mask_ori)

            # Step & clamp safely
            opt.step()
            with torch.no_grad():
                nugget.clamp_(min=1e-7)

        # Evaluate condition number
        cur_cond = model.interpolation_options.kernel_options.condition_number
        print(f"[Epoch {epoch}] cond. num. = {cur_cond:.2e}")

        if _has_converged(cur_cond, prev_cond, target_cond_num, epoch, min_impr, patience):
            break
        prev_cond = cur_cond

    model.interpolation_options.kernel_options.optimizing_condition_number = False
    return model


def _mask_iqr(grads, multiplier: float = 1.5) -> torch.BoolTensor:
    q1, q3 = grads.quantile(0.25), grads.quantile(0.75)
    thresh = q3 + multiplier * (q3 - q1)
    return grads > thresh

def _apply_outlier_gradients(
        tensor: torch.Tensor,
        mask: torch.BoolTensor,
        amplification: float = 5.0,
):
    # wrap in no_grad if you prefer, but .grad modifications are fine
    tensor.grad.view(-1)[mask] *= amplification
    tensor.grad.view(-1)[~mask] = 0



def _gradient_masking(nugget, focus=0.01):
    """Old way of avoiding exploding gradients."""
    grads = nugget.grad.abs()
    k = int(grads.numel() * focus)
    top_vals, top_idx = torch.topk(grads, k, largest=True)
    mask = torch.zeros_like(grads)
    mask[top_idx] = 1
    nugget.grad.mul_(mask)


def _has_converged(
        current: float,
        previous: float,
        target: float = 1e5,
        epoch: int = 0,
        min_improvement: float = 0.01,
        patience: int = 10,
) -> bool:
    if current < target:
        return True
    if epoch > patience:
        # relative improvement
        rel_impr = abs(current - previous) / max(previous, 1e-8)
        return rel_impr < min_improvement
    return False


# region legacy
def nugget_optimizer__legacy(target_cond_num, engine_cfg, model, max_epochs):
    geo_model: GeoModel = model
    convergence_criteria = target_cond_num
    engine_config = engine_cfg

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
    return geo_model


def _check_convergence_criterion(conditional_number: float, condition_number_old: float, conditional_number_target: float = 1e5, epoch: int = 0):
    import torch
    reached_conditional_target = conditional_number < conditional_number_target
    if reached_conditional_target == False and epoch > 10:
        condition_number_change = torch.abs(conditional_number - condition_number_old) / condition_number_old
        if condition_number_change < 0.01:
            reached_conditional_target = True
    return reached_conditional_target

# endregion
