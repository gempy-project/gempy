import pytest

import gempy as gp
import gempy_viewer as gpv
from gempy.core.data.gempy_engine_config import GemPyEngineConfig
from gempy_engine.config import AvailableBackends
from gempy.core.data.enumerators import ExampleModel
from test.conftest import TEST_SPEED, TestSpeed


@pytest.mark.skipif(TEST_SPEED.value < TestSpeed.SECONDS.value, reason="Global test speed below this test value.")
class TestBackends:
    @pytest.fixture(scope='class')
    def geo_model(self):
        geo_model: gp.data.GeoModel = gp.generate_example_model(
            example_model=ExampleModel.ONE_FAULT,
            compute_model=False
        )

        geo_model.interpolation_options.number_octree_levels = 6
        return geo_model
    
    def test_backends_numpy(self, geo_model):
        gp.compute_model(geo_model, engine_config=GemPyEngineConfig(backend=AvailableBackends.numpy))
        gpv.plot_3d(
            model=geo_model,
            show_data=True,
            show_boundaries=True,
            show_lith=False,
            image=True
        )

    
    def test_numpy_pykeops(self, geo_model):
        gp.compute_model(
            gempy_model=geo_model,
            engine_config=GemPyEngineConfig(
                backend=AvailableBackends.numpy,
                use_gpu=False,
            ))
        
        gpv.plot_3d(
            model=geo_model,
            show_data=True,
            show_boundaries=True,
            show_lith=False,
            image=True
        )

    @pytest.mark.skip(reason="Not finished yet. Fault data is not passed to legacy properly.")
    @pytest.mark.skipif(TEST_SPEED.value <= 1, reason="Global test speed below this test value.")
    def test_backends_legacy(self, geo_model):
        # TODO: (Sep 2023, Miguel) Pass fault data to legacy properly
        
        gp.compute_model(geo_model, engine_config=GemPyEngineConfig(backend=AvailableBackends.legacy))
        from gempy.optional_dependencies import require_gempy_legacy
        gpl = require_gempy_legacy()
        gpl.plot_3d(
            model=geo_model.legacy_model,
            show_data=True,
            show_boundaries=True,
            show_lith=False,
            image=True
        )