from typing import Optional

import config
import gempy_engine
from gempy_engine.core.data import Solutions
from gempy_engine.config import AvailableBackends
from ..core.data.geo_model import GeoModel
from ..optional_dependencies import require_gempy_legacy


def compute_model(gempy_model: GeoModel, output: Optional[list[str]] = None) -> Solutions:
    # TODO: output should be deprecated and use instead interpolation options
    # TODO: Probably we will need a switch here for the backends
    # Make match switch for enumerator BackendTensor.engine_backend
    match config.DEFAULT_BACKEND:
        case AvailableBackends.numpy:
            gempy_model.solutions = gempy_engine.compute_model(
                interpolation_input=gempy_model.interpolation_input,
                options=gempy_model.interpolation_options,
                data_descriptor=gempy_model.input_data_descriptor
            )
        case AvailableBackends.aesara:
            _legacy_compute_model(gempy_model)
        case AvailableBackends.jax:
            raise NotImplementedError()
        case AvailableBackends.tensorflow:
            raise NotImplementedError()


    return gempy_model.solutions



def _legacy_compute_model(gempy_model: GeoModel):
    # TODO: import gempy_legacy as optional requirement
    gempy_legacy = require_gempy_legacy()
    pass


    # TODO: Convert GeoModel to geo_data
    