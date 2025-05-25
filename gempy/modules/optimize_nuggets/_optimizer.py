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
        lr: float = 1e-2,
        patience: int = 10,
        min_impr: float = 0.01,
) -> float:
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
    nugget.requires_grad_(True)

    opt = torch.optim.Adam(params=[nugget], lr=lr)

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
            if False:
                _gradient_masking(nugget)
            else:
                clip_grad_norm_(parameters=[nugget], max_norm=1.0)

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
    return nugget.item()


def _gradient_masking(nugget):
    """Old way of avoiding exploding gradients."""
    grads = nugget.grad.abs()
    k = int(grads.numel() * 0.1)
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
