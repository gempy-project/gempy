import numpy as np


def encode_numpy_array(array: np.ndarray):
    # Check length 
    if array.size > 10:
        return []
    return array.tolist()
