import hashlib
import sys

from typing import Sequence

import numpy as np


def structural_element_hasher(i: int, name: str, hash_length: int = 8) -> int:
    # Get the last 'hash_length' digits from the hash
    name_hash = int(hashlib.md5(name.encode('utf-8')).hexdigest(), 16) % (10 ** hash_length)
    return i * (10 ** hash_length) + name_hash


def resolve_ids_from_names(name_id_map: dict[str, int], names: Sequence[str]) -> np.ndarray:
    """Map each name to its id.

    Args:
        name_id_map (dict[str, int]): Mapping between structural element names and ids.
        names (Sequence[str]): Structural element name of each data point.

    Returns:
        np.ndarray: The id of each data point, in the order of ``names``.

    Raises:
        ValueError: If any name is not in ``name_id_map``, listing the unknown and the available names.
    """
    try:
        return np.array([name_id_map[name] for name in names])
    except KeyError:
        unknown_names = [str(name) for name in dict.fromkeys(names) if name not in name_id_map]
        raise ValueError(
            f"Unknown structural element name(s) {unknown_names}. "
            f"Available names: {list(name_id_map)}. "
            f"Names must match exactly between surface points, orientations and name_id_map."
        ) from None


def generate_ids_from_names(name_id_map, names, x):
    name_id_map = name_id_map or {name: structural_element_hasher(i, name) for i, name in enumerate(np.unique(names))}
    if isinstance(names, str):
        ids = np.repeat(resolve_ids_from_names(name_id_map, [names]), len(x))
    elif isinstance(names, Sequence) or isinstance(names, np.ndarray):
        ids = resolve_ids_from_names(name_id_map, names)
    elif type(names).__module__.startswith('pandas') and type(names).__name__ == 'StringArray':
        ids = resolve_ids_from_names(name_id_map, names)
    else:
        raise TypeError(f"Names should be a string or a NumPy array, not {type(names)}")
    return ids, name_id_map
