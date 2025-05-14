import hashlib

from typing import Sequence

import numpy as np


def structural_element_hasher(i: int, name: str, hash_length: int = 8) -> int:
    # Get the last 'hash_length' digits from the hash
    name_hash = int(hashlib.md5(name.encode('utf-8')).hexdigest(), 16) % (10 ** hash_length)
    return i * (10 ** hash_length) + name_hash


def generate_ids_from_names(name_id_map, names, x):
    name_id_map = name_id_map or {name: structural_element_hasher(i, name) for i, name in enumerate(np.unique(names))}
    if isinstance(names, str):
        ids = np.array([name_id_map[names]] * len(x))
    elif isinstance(names, Sequence) or isinstance(names, np.ndarray):
        ids = np.array([name_id_map[name] for name in names])
    else:
        raise TypeError(f"Names should be a string or a NumPy array, not {type(names)}")
    return ids, name_id_map
