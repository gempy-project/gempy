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
