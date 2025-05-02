import gempy as gp
from gempy.core.data.enumerators import ExampleModel
from gempy.optional_dependencies import require_gempy_viewer

PLOT = True


def test_default_options():
    model = gp.generate_example_model(ExampleModel.COMBINATION, compute_model=False)
    
    # TODO: Change options here

    gp.compute_model(
        gempy_model=model,
        engine_config=gp.data.GemPyEngineConfig(
            backend=gp.data.AvailableBackends.PYTORCH
        )
    )

    if PLOT:
        gpv = require_gempy_viewer()
        gpv.plot_3d(model, show_data=True, image=True)


def test_fast_options():
    model = gp.generate_example_model(ExampleModel.COMBINATION, compute_model=False)

    # TODO: Change options here

    gp.compute_model(
        gempy_model=model,
        engine_config=gp.data.GemPyEngineConfig(
            backend=gp.data.AvailableBackends.PYTORCH
        )
    )

    if PLOT:
        gpv = require_gempy_viewer()
        gpv.plot_3d(model, show_data=True, image=True)

    