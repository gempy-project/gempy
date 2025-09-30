from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.interpolation_input import InterpolationInput
from ._ops import run_optimization
from ..data_manipulation.manipulate_points import modify_surface_points
from ...core.data import GeoModel, GemPyEngineConfig, StructuralGroup
from ...modules.data_manipulation import interpolation_input_from_structural_frame


def nugget_optimizer(
        target_cond_num: float,
        engine_cfg: GemPyEngineConfig,
        model: GeoModel,
        max_epochs: int,
        only_groups: list[StructuralGroup] | None = None,
        lr: float = .01,
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
        grads=True
    )

    # 2) Prepare data

    all_groups: list[StructuralGroup] = model.structural_frame.structural_groups
    if only_groups is not None:
        groups_to_optimize = only_groups
    else:
        groups_to_optimize: list[StructuralGroup] = all_groups
        
    for group in groups_to_optimize:
        model.structural_frame.structural_groups = [group]
    
        interp_in: InterpolationInput = interpolation_input_from_structural_frame(model)
        model.taped_interpolation_input = interp_in
        nugget: "torch.Tensor" = interp_in.surface_points.nugget_effect_scalar
        nugget.requires_grad_(True)

        model.interpolation_options.kernel_options.optimizing_condition_number = True

        nuggets = run_optimization(lr, max_epochs, min_impr, model, nugget, patience, target_cond_num)
        
        modify_surface_points(
            geo_model=model,
            nugget=nuggets.detach().numpy()
        )

        model.interpolation_options.kernel_options.optimizing_condition_number = False
    
    model.structural_frame.structural_groups = all_groups
    
    return model


