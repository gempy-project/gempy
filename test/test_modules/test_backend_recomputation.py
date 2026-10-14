import numpy as np
import pytest

import gempy as gp
from gempy.core.data.enumerators import ExampleModel
from gempy.modules.data_manipulation import interpolation_input_from_structural_frame
from gempy_engine.core.backend_tensor import BackendTensor


def test_autograd_weights_survive_numpy_recomputation():
    torch = pytest.importorskip('torch')
    grad_enabled = torch.is_grad_enabled()
    try:
        model = gp.generate_example_model(ExampleModel.ANTICLINE, compute_model=False)
        model.interpolation_options.evaluation_options.mesh_extraction = False
        model.interpolation_options.evaluation_options.number_octree_levels = 1
        original = gp.compute_model(model, gp.data.GemPyEngineConfig(
            backend=gp.data.AvailableBackends.PYTORCH, dtype='float64', compute_grads=True))
        weights = original.root_output.outputs[0].weights
        assert weights.grad_fn is not None
        expected = weights.detach().cpu().numpy().copy()
        input_snapshot = interpolation_input_from_structural_frame(model)
        np.testing.assert_allclose(input_snapshot.weights[0], expected)
        input_snapshot.weights[0][:] = 0
        np.testing.assert_allclose(weights.detach().cpu().numpy(), expected)

        repeated = gp.compute_model(model, gp.data.GemPyEngineConfig(
            backend=gp.data.AvailableBackends.numpy, dtype='float64'))
        np.testing.assert_allclose(repeated.root_output.outputs[0].weights, expected, atol=1e-7)
        # The original graph is still usable after the other backend's computation.
        weights.sum().backward()
        assert weights.grad_fn is not None
    finally:
        BackendTensor._change_backend(gp.data.AvailableBackends.numpy)
        torch.set_grad_enabled(grad_enabled)
