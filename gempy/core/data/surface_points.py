from dataclasses import dataclass
from typing import Optional
import numpy as np

from gempy.optional_dependencies import require_pandas

DEFAULT_NUGGET = 0.00001


# ? Maybe we should merge this with the SurfacePoints class from gempy_engine
# ? It does not seem a good a idea because gempy_engine.SurfacePoints is too terse


@dataclass  
class SurfacePointsTable:
    """
    A dataclass to represent a table of surface points in a geological model.
    
    """
    data: np.ndarray  #: A structured NumPy array holding the X, Y, Z coordinates, id, and nugget of each surface point.
    name_id_map: Optional[dict[str, int]] = None  #: A mapping between surface point names and ids.

    dt = np.dtype([('X', 'f8'), ('Y', 'f8'), ('Z', 'f8'), ('id', 'i4'), ('nugget', 'f8')])  #: The custom data type for the data array.

    def __str__(self):
        return "\n" + np.array2string(self.data, precision=2, separator=',', suppress_small=True)

    def __repr__(self):
        return f"SurfacePointsTable(data=\n{np.array2string(self.data, precision=2, separator=',', suppress_small=True)},\nname_id_map={self.name_id_map})"
    
    def _repr_html_(self):
        rows_to_display = 10  # Define the number of rows to display from beginning and end
        html = "<table>"
        html += "<tr><th>X</th><th>Y</th><th>Z</th><th>id</th><th>nugget</th></tr>"
        if len(self.data) > 2*rows_to_display:
            for point in self.data[:rows_to_display]:
                html += "<tr><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td><td>{}</td><td>{:.2f}</td></tr>".format(*point)
            html += "<tr><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td></tr>"
            for point in self.data[-rows_to_display:]:
                html += "<tr><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td><td>{}</td><td>{:.2f}</td></tr>".format(*point)
        else:
            for point in self.data:
                html += "<tr><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td><td>{}</td><td>{:.2f}</td></tr>".format(*point)
        html += "</table>"
        return html
    
    
    @classmethod
    def from_arrays(cls, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                    names: np.ndarray, nugget: Optional[np.ndarray] = None) -> 'SurfacePointsTable':
        if nugget is None:
            nugget = np.zeros_like(x) + DEFAULT_NUGGET

        data = np.zeros(len(x), dtype=SurfacePointsTable.dt)

        name_id_map = {name: i for i, name in enumerate(np.unique(names))}
        ids = np.array([name_id_map[name] for name in names])

        data['X'], data['Y'], data['Z'], data['id'], data['nugget'] = x, y, z, ids, nugget
        return cls(data, name_id_map)

    def id_to_name(self, id: int) -> str:
        return list(self.name_id_map.keys())[id]
    
    @property
    def xyz(self) -> np.ndarray:
        return np.array([self.data['X'], self.data['Y'], self.data['Z']]).T
    
    @property
    def nugget(self) -> np.ndarray:
        return self.data['nugget']
    
    @nugget.setter
    def nugget(self, value: np.ndarray):
        self.data['nugget'] = value
    
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
    def ids(self) -> np.ndarray:
        return self.data['id']
        
    @property
    def id(self) -> int:
        # Check id is the same in the whole column and return it or throw an error
        ids = np.unique(self.data['id'])
        if len(ids) > 1:
            raise ValueError(f"OrientationsTable contains more than one id: {ids}")
        if len(ids) == 0:
            raise ValueError(f"OrientationsTable contains no ids")
        return ids[0]
    
    
    @property
    def df(self) -> 'pd.DataFrame':
        pd = require_pandas()
        return pd.DataFrame(self.data)
    
