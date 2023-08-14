from dataclasses import dataclass
from typing import Optional

import numpy as np

from gempy.optional_dependencies import require_pandas

DEFAULT_ORI_NUGGET = 0.01

# ? Maybe we should merge this with the SurfacePoints class from gempy_engine


@dataclass
class OrientationsTable:
    """
    A dataclass to represent a table of orientations in a geological model.
    
    """ 
    data: np.ndarray  #: A structured NumPy array holding the X, Y, Z coordinates, gradients G_x, G_y, G_z, id, and nugget of each orientation.
    name_id_map: Optional[dict[str, int]] = None  #: A mapping between orientation names and ids.
    
    dt = np.dtype([('X', 'f8'), ('Y', 'f8'), ('Z', 'f8'), ('G_x', 'f8'), ('G_y', 'f8'), ('G_z', 'f8'), ('id', 'i4'), ('nugget', 'f8')])  #: The custom data type for the data array.
    
    @classmethod
    def from_arrays(cls, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                    G_x: np.ndarray, G_y: np.ndarray, G_z: np.ndarray,
                    names: np.ndarray, nugget: Optional[np.ndarray] = None) -> 'OrientationsTable':

        data, name_id_map = cls.data_from_arrays(x, y, z, G_x, G_y, G_z, names, nugget)
        return cls(data, name_id_map)

    @classmethod
    def data_from_arrays(cls, x, y, z, G_x, G_y, G_z, names, nugget,):
        if nugget is None:
            nugget = np.zeros_like(x) + DEFAULT_ORI_NUGGET
        data = np.zeros(len(x), dtype=OrientationsTable.dt)
        name_id_map = {name: i for i, name in enumerate(np.unique(names))}
        ids = np.array([name_id_map[name] for name in names])
        data['X'], data['Y'], data['Z'], data['G_x'], data['G_y'], data['G_z'], data['id'], data['nugget'] = x, y, z, G_x, G_y, G_z, ids, nugget
        return data, name_id_map

    @classmethod
    def initialize_empty(cls) -> 'OrientationsTable':
        return cls(np.zeros(0, dtype=OrientationsTable.dt))
    
    
    @property
    def xyz(self) -> np.ndarray:
        return np.array([self.data['X'], self.data['Y'], self.data['Z']]).T
    
    @property
    def grads(self) -> np.ndarray:
        return np.array([self.data['G_x'], self.data['G_y'], self.data['G_z']]).T
    
    @property
    def nugget(self) -> np.ndarray:
        return self.data['nugget']

    @property
    def ids(self) -> np.ndarray:
        return self.data['id']
    
    def get_orientations_by_name(self, name: str) -> 'OrientationsTable':
        return self.get_orientations_by_id(self.name_id_map[name])

    def get_orientations_by_id(self, id: int) -> 'OrientationsTable':
        return OrientationsTable(self.data[self.data['id'] == id], self.name_id_map)
    
    def get_orientations_by_id_groups(self) -> list['OrientationsTable']:
        ids = np.unique(self.data['id'])
        return [self.get_orientations_by_id(id) for id in ids]

    @classmethod
    def fill_missing_orientations_groups(cls, orientations_groups: list['OrientationsTable'],
                                         surface_points_groups: list['SurfacePointsTable']) -> list['OrientationsTable']:
        # region Deal with elements without orientations
        if len(surface_points_groups) > len(orientations_groups):
            # Check the ids of the surface points and find the missing ones
            surface_points_ids = [surface_points_group.id for surface_points_group in surface_points_groups]
            orientations_ids = [orientations_group.id for orientations_group in orientations_groups]

            missing_ids = list(set(surface_points_ids) - set(orientations_ids))

            empty_orientations = [cls(data=np.zeros(0, dtype=cls.dt)) for id in missing_ids] # Create empty orientations

            for empty_orientation, id in zip(empty_orientations, missing_ids): # Insert the empty orientations in the right position
                orientations_groups.insert(id, empty_orientation)
        # endregion

        return orientations_groups
    
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

    def __str__(self):
        return "\n" + np.array2string(self.data, precision=2, separator=',', suppress_small=True)

    def __repr__(self):
        return f"OrientationsTable(data=\n{np.array2string(self.data, precision=2, separator=',', suppress_small=True)},\nname_id_map={self.name_id_map})"

    def _repr_html_(self):
        rows_to_display = 10  # Define the number of rows to display from beginning and end
        html = "<table>"
        html += "<tr><th>X</th><th>Y</th><th>Z</th><th>G_x</th><th>G_y</th><th>G_z</th><th>id</th><th>nugget</th></tr>"
        if len(self.data) > 2 * rows_to_display:
            for point in self.data[:rows_to_display]:
                html += "<tr><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td><td>{}</td><td>{:.2f}</td></tr>".format(*point)
            html += "<tr><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td></tr>"
            for point in self.data[-rows_to_display:]:
                html += "<tr><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td><td>{}</td><td>{:.2f}</td></tr>".format(*point)
        else:
            for point in self.data:
                html += "<tr><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td><td>{}</td><td>{:.2f}</td></tr>".format(*point)
        html += "</table>"
        return html

    def __len__(self):
        return len(self.data)

