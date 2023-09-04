import gempy as gp
import gempy_viewer
from gempy.core.data import Solutions
from gempy.API.gp2_gp3_compatibility.gp3_to_gp2_input import gempy3_to_gempy2
from gempy.optional_dependencies import require_gempy_legacy
from test.test_api.test_initialization_and_compute_api import _create_data

# Skip pytest if legacy is not installed
try:
    require_gempy_legacy()
except ImportError:
    import pytest
    pytest.skip("Legacy is not installed.", allow_module_level=True)



def test_compare_numpy_with_legacy():
    geo_data = _create_data()

    gp.map_stack_to_surfaces(
        gempy_model=geo_data,
        mapping_object={"Strat_Series": ('rock2', 'rock1')}
    )

    sol: Solutions = gp.compute_model(geo_data)

    gempy_viewer.plot_2d(geo_data, direction=['y'], show_data=True, show_boundaries=False)

    legacy_model = gempy3_to_gempy2(geo_data)

    gl = require_gempy_legacy()
    gl.set_interpolator(legacy_model)
    gl.compute_model(legacy_model)
    gl.plot_2d(legacy_model, direction=['y'])
