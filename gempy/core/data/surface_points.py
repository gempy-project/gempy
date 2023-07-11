from dataclasses import dataclass
from typing import Optional

import numpy as np

from gempy.optional_dependencies import require_pandas

DEFAULT_NUGGET = 0.00001


# ? Maybe we should merge this with the SurfacePoints class from gempy_engine
# ? It does not seem a good a idea because gempy_engine.SurfacePoints is too terse


@dataclass  
class SurfacePointsTable:
    data: np.ndarray
    name_id_map: Optional[dict[str, int]] = None  # ? Do I need this here or this should be a field of StructuralFrame?

    @classmethod
    def from_arrays(cls, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                    names: np.ndarray, nugget: Optional[np.ndarray] = None) -> 'SurfacePointsTable':
        if nugget is None:
            nugget = np.zeros_like(x) + DEFAULT_NUGGET

        dt = np.dtype([('X', 'f8'), ('Y', 'f8'), ('Z', 'f8'), ('id', 'i4'), ('nugget', 'f8')])
        data = np.zeros(len(x), dtype=dt)

        name_id_map = {name: i for i, name in enumerate(np.unique(names))}
        ids = np.array([name_id_map[name] for name in names])

        data['X'], data['Y'], data['Z'], data['id'], data['nugget'] = x, y, z, ids, nugget
        return cls(data, name_id_map)

    @property
    def xyz(self) -> np.ndarray:
        return np.array([self.data['X'], self.data['Y'], self.data['Z']]).T
    
    def __len__(self):
        return len(self.data)
    
    def get_surface_points_by_name(self, name: str) -> 'SurfacePointsTable':
        return self.get_surface_points_by_id(self.name_id_map[name])

    def get_surface_points_by_id(self, id: int) -> 'SurfacePointsTable':
        return SurfacePointsTable(self.data[self.data['id'] == id], self.name_id_map)

    def get_surface_points_by_id_groups(self) -> list['SurfacePointsTable']:
        ids = np.unique(self.data['id'])
        return [self.get_surface_points_by_id(id) for id in ids]
    
    @property
    def df(self) -> 'pd.DataFrame':
        pd = require_pandas()
        return pd.DataFrame(self.data)
    
