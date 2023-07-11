from dataclasses import dataclass
from typing import Optional

import numpy as np

from gempy.optional_dependencies import require_pandas

DEFAULT_NUGGET = 0.01

# ? Maybe we should merge this with the SurfacePoints class from gempy_engine


@dataclass
class OrientationsTable:
    data: np.ndarray
    name_id_map: Optional[dict[str, int]] = None  # ? Do I need this here or this should be a field of StructuralFrame?

    def __len__(self):
        return len(self.data)

    @classmethod
    def from_arrays(cls, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                    G_x: np.ndarray, G_y: np.ndarray, G_z: np.ndarray,
                    names: np.ndarray, nugget: Optional[np.ndarray] = None) -> 'OrientationsTable':

        if nugget is None:
            nugget = np.zeros_like(x) + DEFAULT_NUGGET

        dt = np.dtype([('X', 'f8'), ('Y', 'f8'), ('Z', 'f8'), ('G_x', 'f8'), ('G_y', 'f8'), ('G_z', 'f8'), ('id', 'i4'), ('nugget', 'f8')])
        data = np.zeros(len(x), dtype=dt)

        name_id_map = {name: i for i, name in enumerate(np.unique(names))}
        ids = np.array([name_id_map[name] for name in names])

        data['X'], data['Y'], data['Z'], data['G_x'], data['G_y'], data['G_z'], data['id'], data['nugget'] = x, y, z, G_x, G_y, G_z, ids, nugget
        return cls(data, name_id_map)

    @property
    def xyz(self) -> np.ndarray:
        return np.array([self.data['X'], self.data['Y'], self.data['Z']]).T
    
    @property
    def grads(self) -> np.ndarray:
        return np.array([self.data['G_x'], self.data['G_y'], self.data['G_z']]).T

    def get_orientations_by_name(self, name: str) -> 'OrientationsTable':
        return self.get_orientations_by_id(self.name_id_map[name])

    def get_orientations_by_id(self, id: int) -> 'OrientationsTable':
        return OrientationsTable(self.data[self.data['id'] == id], self.name_id_map)

    def get_orientations_by_id_groups(self) -> list['OrientationsTable']:
        ids = np.unique(self.data['id'])
        return [self.get_orientations_by_id(id) for id in ids]

    @property
    def df(self) -> 'pd.DataFrame':
        pd = require_pandas()
        return pd.DataFrame(self.data)

