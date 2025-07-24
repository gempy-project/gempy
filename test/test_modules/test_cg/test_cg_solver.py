import gempy as gp
from gempy.core.data.enumerators import ExampleModel
from gempy.optional_dependencies import require_gempy_viewer
from gempy_engine.core.data.interp_output import InterpOutput

from test.verify_helper import gempy_verify_array
import pytest
from test.conftest import TEST_SPEED, TestSpeed

PLOT = True


def test_generate_greenstone_model():
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

    if PLOT:
        gpv = require_gempy_viewer()
        gpv.plot_3d(model, image=True)
