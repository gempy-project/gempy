from typing import Optional

import gempy_engine
from gempy_engine.core.backend_tensor import BackendTensor
from gempy.API.gp2_gp3_compatibility.gp3_to_gp2_input import gempy3_to_gempy2
from gempy_engine.config import AvailableBackends
from gempy_engine.core.data import Solutions
from ..core.data.gempy_engine_config import GempyEngineConfig
from ..core.data.geo_model import GeoModel
from ..optional_dependencies import require_gempy_legacy


def compute_model(gempy_model: GeoModel, engine_config: Optional[GempyEngineConfig] = None) -> Solutions:
    engine_config = engine_config or GempyEngineConfig(
        backend=AvailableBackends.numpy, 
        use_gpu=False, 
        pykeops_enabled=False
    )
    
    match engine_config.backend:
        case AvailableBackends.numpy | AvailableBackends.tensorflow:

            BackendTensor.change_backend(
                engine_backend=engine_config.backend,
                use_gpu=engine_config.use_gpu,
                pykeops_enabled=engine_config.pykeops_enabled)

            gempy_model.solutions = gempy_engine.compute_model(
                interpolation_input=gempy_model.interpolation_input,
                options=gempy_model.interpolation_options,
                data_descriptor=gempy_model.input_data_descriptor
            )
        
        case AvailableBackends.aesara | AvailableBackends.legacy:
            gempy_model.legacy_model = _legacy_compute_model(gempy_model)
        case _:
            raise ValueError(f'Backend {engine_config} not supported')


    return gempy_model.solutions



def _legacy_compute_model(gempy_model: GeoModel) -> 'gempy_legacy.Project':
    gpl = require_gempy_legacy()
    legacy_model: gpl.Project = gempy3_to_gempy2(gempy_model)
    gpl.set_interpolator(legacy_model)
    gpl.compute_model(legacy_model)
    return legacy_model

    