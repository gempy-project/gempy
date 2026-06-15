import numpy as np


def encode_numpy_array(array: np.ndarray):
    array = np.asarray(array)
    if array.size > 10:
        return []
    return array.tolist()
