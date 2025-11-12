import pytest

import gempy as gp
from gempy.core.data.enumerators import ExampleModel
from gempy.optional_dependencies import require_gempy_viewer
import gempy_engine.core.backend_tensor as BackendTensor

from gempy_engine.modules.weights_cache.weights_cache_interface import WeightCache

PLOT = True

pytest.mark.skip("Run explicitly")


def test_solve_with_cg():
    model = gp.generate_example_model(ExampleModel.GREENSTONE, compute_model=False)
    print(model.structural_frame)

    WeightCache.clear_cache()
    BackendTensor.PYKEOPS = False

    sol = gp.compute_model(
        gempy_model=model,
        engine_config=gp.data.GemPyEngineConfig(
            backend=gp.data.AvailableBackends.PYTORCH,
            use_gpu=False,
            dtype='float64'
        )
    )

    if PLOT:
        gpv = require_gempy_viewer()
        gpv.plot_3d(model, image=True)


def test_save_weights():
    model = gp.generate_example_model(ExampleModel.GREENSTONE, compute_model=False)
    print(model.structural_frame)

    sol = gp.compute_model(
        gempy_model=model,
        engine_config=gp.data.GemPyEngineConfig(
            backend=gp.data.AvailableBackends.PYTORCH,
            use_gpu=False,
            dtype='float32'
        )
    )
    weights1 = sol.octrees_output[0].outputs_centers[0].weights
    weights2 = sol.octrees_output[0].outputs_centers[1].weights
    weights3 = sol.octrees_output[0].outputs_centers[2].weights

    WeightCache.clear_cache()
    BackendTensor.PYKEOPS = False

    sol = gp.compute_model(
        gempy_model=model,
        engine_config=gp.data.GemPyEngineConfig(
            backend=gp.data.AvailableBackends.PYTORCH,
            use_gpu=False,
            dtype='float32'
        )
    )

    if PLOT:
        gpv = require_gempy_viewer()
        gpv.plot_3d(model, image=True)


def test_keops_x_torch():
    model = gp.generate_example_model(ExampleModel.GREENSTONE, compute_model=False)
    print(model.structural_frame)

    WeightCache.clear_cache()
    BackendTensor.PYKEOPS = False
    sol = gp.compute_model(
        gempy_model=model,
        engine_config=gp.data.GemPyEngineConfig(
            backend=gp.data.AvailableBackends.PYTORCH,
            use_gpu=False,
            dtype='float32'
        )
    )

    if PLOT:
        gpv = require_gempy_viewer()
        gpv.plot_3d(model, image=True)
