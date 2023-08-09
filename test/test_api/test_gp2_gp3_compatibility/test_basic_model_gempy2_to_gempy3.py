import os
import pytest

#IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
IN_GITHUB_ACTIONS = True

import gempy as gp
from gempy import Project
from gempy_viewer import GemPyToVista
from gempy.API.gp2_gp3_compatibility.gp2_to_gp3_input import gempy_project_to_interpolation_input, gempy_project_to_input_data_descriptor, gempy_project_to_interpolation_options
from gempy.API.gp2_gp3_compatibility.gp3_to_gp2_output import set_gp3_solutions_to_gp2_solution
from test.test_api.test_gp2_gp3_compatibility._gp2togp3_test_utils import load_model, map_sequential_pile

if not IN_GITHUB_ACTIONS:
    import gempy_engine
    from gempy_engine.config import AvailableBackends
    from gempy_engine.core.backend_tensor import BackendTensor
    from gempy_engine.core.data import InterpolationOptions
    from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
    from gempy_engine.core.data.interpolation_input import InterpolationInput
    from gempy_engine.core.data.solutions import Solutions

input_path = os.path.dirname(__file__) + '/../../../test/input_data'
import sys  # These two lines are necessary only if GemPy is not installed

sys.path.append("../../../gempy_3")


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Only runs once gempy 3 is published.")
def test_set_gempy3_gempy2_bridge():
    """
    This test is based on gempy/test/test_core/test_model.py:test_compute_model
    """
    BackendTensor.change_backend(AvailableBackends.numpy, use_gpu=False, pykeops_enabled=False)

    geo_model: Project = load_model()
    geo_model = map_sequential_pile(geo_model)

    # @off
    interpolation_input  : InterpolationInput   = gempy_project_to_interpolation_input(geo_model)
    input_data_descriptor: InputDataDescriptor  = gempy_project_to_input_data_descriptor(geo_model)
    options              : InterpolationOptions = gempy_project_to_interpolation_options(geo_model)
    # @on
            
    print(interpolation_input)
    print(input_data_descriptor)
    print(options)

    solutions: Solutions = gempy_engine.compute_model(
        # @off
        interpolation_input = interpolation_input,
        options             = options,
        data_descriptor     = input_data_descriptor
        # @on
    )

    set_gp3_solutions_to_gp2_solution(gp3_solution=solutions, geo_model=geo_model)
    
    if plot_2d := True:
        gp.plot.plot_2d(geo_model, cell_number=25, direction='y', show_data=True, show_block=True, show_lith=False, series_n=0)
        gp.plot.plot_2d(geo_model, cell_number=25, series_n=1, N=15, show_scalar=True, direction='y', show_data=True)
        gp.plot.plot_2d(geo_model, cell_number=25, direction='y', show_data=True, show_block=False, show_lith=True, series_n=1)
    
    plot_object: GemPyToVista = gp.plot.plot_3d(
        geo_model, show_surfaces=True, show_lith=True, show=False, kwargs_plot_structured_grid={"show_edges": False}
    )