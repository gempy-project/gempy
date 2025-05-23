from typing import Annotated

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


short_array_type = Annotated[np.ndarray, (BeforeValidator(lambda v: np.array(v) if v is not None else None))]


def instantiate_if_necessary(data: dict, key: str, type: type) -> None:
    """
    Creates instances of the specified type for a dictionary key if the key exists and its
    current type does not match the specified type. This function modifies the dictionary
    in place by converting the value associated with the key into an instance of the given
    type.

    This is typically used when a dictionary contains data that needs to be represented as
    objects of a specific class type.

    Args:
        data (dict): The dictionary containing the key-value pair to inspect and possibly
            convert.
        key (str): The key in the dictionary whose value should be inspected and converted
            if necessary.
        type (type): The type to which the value of `key` should be converted, if it is not
            already an instance of the type.

    Returns:
        None
    """
    if key in data and not isinstance(data[key], type):
        data[key] = type(**data[key])

numpy_array_short_validator = BeforeValidator(validate_numpy_array)

# First, create a context variable
loading_model_context = ContextVar('loading_model_context', default={})

@contextmanager
def loading_model_from_binary(input_binary: bytes, grid_binary: bytes):
    token = loading_model_context.set({
            'input_binary': input_binary,
            'grid_binary': grid_binary
    })
    try:
        yield
    finally:
        loading_model_context.reset(token)

