import numpy as np
from pydantic import BeforeValidator


def convert_to_arrays( values, keys):
    for key in keys:
        if key in values and not isinstance(values[key], np.ndarray):
            values[key] = np.array(values[key])
    return values

