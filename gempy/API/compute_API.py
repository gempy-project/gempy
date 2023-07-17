from typing import Optional

import config
import gempy_engine
from gempy_3.gp3_to_gp2_input import gempy3_to_gempy2
from gempy_engine.config import AvailableBackends
from gempy_engine.core.data import Solutions
from ..core.data.geo_model import GeoModel
from ..optional_dependencies import require_gempy_legacy


def compute_model(gempy_model: GeoModel, backend: Optional[AvailableBackends] = None, output: Optional[list[str]] = None) -> Solutions:
    # TODO: output should be deprecated and use instead interpolation options
    # Make match switch for enumerator BackendTensor.engine_backend
    
    backend = backend or config.DEFAULT_BACKEND
    match backend:
        case AvailableBackends.numpy:
            gempy_model.solutions = gempy_engine.compute_model(
                interpolation_input=gempy_model.interpolation_input,
                options=gempy_model.interpolation_options,
                data_descriptor=gempy_model.input_data_descriptor
            )
        
        case AvailableBackends.aesara | AvailableBackends.legacy:
            gempy_model.legacy_model = _legacy_compute_model(gempy_model)
        case AvailableBackends.jax:
            raise NotImplementedError()
        case AvailableBackends.tensorflow:
            raise NotImplementedError()


    return gempy_model.solutions



def _legacy_compute_model(gempy_model: GeoModel) -> 'gempy_legacy.Project':
    gpl = require_gempy_legacy()
    legacy_model: gpl.Project = gempy3_to_gempy2(gempy_model)
    gpl.set_interpolator(legacy_model)
    gpl.compute_model(legacy_model)
    return legacy_model

    