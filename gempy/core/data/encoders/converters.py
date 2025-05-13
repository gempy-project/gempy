from contextlib import contextmanager

from contextvars import ContextVar

import numpy as np
from pydantic import BeforeValidator


def convert_to_arrays(values, keys):
    for key in keys:
        if key in values and not isinstance(values[key], np.ndarray):
            values[key] = np.array(values[key])
    return values


def validate_numpy_array(v):
    return np.array(v) if v is not None else None


numpy_array_short_validator = BeforeValidator(validate_numpy_array)

# First, create a context variable
loading_model_context = ContextVar('loading_model_context', default={})

@contextmanager
def loading_model_injection(surface_points_binary: np.ndarray, orientations_binary: np.ndarray):
    token = loading_model_context.set({
            'surface_points_binary': surface_points_binary,
            'orientations_binary'  : orientations_binary
    })
    try:
        yield
    finally:
        loading_model_context.reset(token)

