import gempy_engine
from gempy.optional_dependencies import require_torch
from gempy_engine.core.data.continue_epoch import ContinueEpoch


def run_optimization(lr, max_epochs, min_impr, model, nugget, patience, target_cond_num):
    torch = require_torch()
    opt = torch.optim.Adam(
        params=[
                nugget,
        ],
        lr=lr
    )
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
            if True:
                # Keep only top 10% gradients
                _gradient_masking(nugget, focus=0.01)
            else:
                if epoch % 1 == 0:
                    mask_sp = _mask_iqr(nugget.grad.abs().view(-1), multiplier=3)
                    print(f"Outliers sp: {torch.nonzero(mask_sp)}")
                _apply_outlier_gradients(tensor=nugget, mask=mask_sp)

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

    # Condition number to numpy
    model.interpolation_options.kernel_options.condition_number = model.interpolation_options.kernel_options.condition_number.detach().numpy()
    return nugget


def _mask_iqr(grads, multiplier: float = 1.5) -> "torch.BoolTensor":
    q1, q3 = grads.quantile(0.25), grads.quantile(0.75)
    thresh = q3 + multiplier * (q3 - q1)
    return grads > thresh

def _apply_outlier_gradients(
        tensor: "torch.Tensor",
        mask: "torch.BoolTensor",
        amplification: float = 1.0,
):
    # wrap in no_grad if you prefer, but .grad modifications are fine
    tensor.grad.view(-1)[mask] *= amplification
    tensor.grad.view(-1)[~mask] = 0



def _gradient_masking(nugget, focus=0.01):
    """Old way of avoiding exploding gradients."""
    torch = require_torch()
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

