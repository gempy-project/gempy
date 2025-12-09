import numpy as np
import gempy as gp
from gempy.core.data.enumerators import ExampleModel
from gempy.optional_dependencies import require_gempy_viewer

PLOT = True


def test_fault_relations_implementation():
    # TODO! (Miguel Dec25) These fault description are not serializing!
    model = gp.generate_example_model(ExampleModel.FAULT_RELATION, compute_model=True)

    correct_relations = np.array([
        [0, 1, 1, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ], dtype=bool)

    # Assert
    assert np.array_equal(model.structural_frame.fault_relations, correct_relations) == True

    if PLOT:
        gpv = require_gempy_viewer()
        gtv: gpv.GemPyToVista = gpv.plot_3d(
            model=model,
            show_data=True,
            image=True,
            show=True
        )
