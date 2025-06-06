from dataclasses import dataclass
from typing import Optional, Union, Sequence

import numpy as np
from gempy_engine.core.data.transforms import Transform
from pydantic import Field

from ._data_points_helpers import generate_ids_from_names
from ...optional_dependencies import require_pandas

DEFAULT_SP_NUGGET = 0.00002


# ? Maybe we should merge this with the SurfacePoints class from gempy_engine
# ? It does not seem a good a idea because gempy_engine.SurfacePoints is too terse


# ! ids are not used apparently
@dataclass(init=True)
class SurfacePointsTable:
    """
    A dataclass to represent a table of surface points in a geological model.
    
    """
    dt = np.dtype([('X', 'f8'), ('Y', 'f8'), ('Z', 'f8'), ('id', 'i4'), ('nugget', 'f8')])  #: The custom data type for the data array.
    
    data: np.ndarray = Field(
        default=np.zeros(0, dtype=dt),
        exclude=True,
        description="A structured NumPy array holding the X, Y, Z coordinates, id, and nugget of each surface point."
    )  #: A structured NumPy array holding the X, Y, Z coordinates, id, and nugget of each surface point.
    name_id_map: Optional[dict[str, int]] = None  #: A mapping between surface point names and ids.
    _model_transform: Optional[Transform] = None

    def __post_init__(self):
        # Check if the data array has the correct data type
        if self.data.dtype != SurfacePointsTable.dt:
            raise ValueError(f"Data array must have the following data type: {SurfacePointsTable.dt}")

    def __str__(self):
        return "\n" + np.array2string(self.data, precision=2, separator=',', suppress_small=True)

    def __repr__(self):
        return f"SurfacePointsTable(data=\n{np.array2string(self.data, precision=2, separator=',', suppress_small=True)},\nname_id_map={self.name_id_map})"

    def _repr_html_(self):
        rows_to_display = 10  # Define the number of rows to display from beginning and end
        html = "<table>"
        html += "<tr><th>X</th><th>Y</th><th>Z</th><th>id</th><th>nugget</th></tr>"
        if len(self.data) > 2 * rows_to_display:
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
                    names: Union[Sequence | str], nugget: Optional[np.ndarray] = None,
                    name_id_map: Optional[dict[str, int]] = None
                    ) -> 'SurfacePointsTable':
        """Create a SurfacePointsTable from arrays of coordinates and names.

        Args:
            x (np.ndarray): Array of x-coordinates.
            y (np.ndarray): Array of y-coordinates.
            z (np.ndarray): Array of z-coordinates.
            names (Union[Sequence, str]): Names of the surface points.
            nugget (Optional[np.ndarray]): Nugget values for the surface points.
            name_id_map (Optional[dict[str, int]]): Mapping between surface point names and ids.

        Returns:
            SurfacePointsTable: A new instance of SurfacePointsTable.
        """
        data, name_id_map = cls._data_from_arrays(x, y, z, names, nugget, name_id_map)
        return cls(data, name_id_map)


    @classmethod
    def _data_from_arrays(cls, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                          names: Union[Sequence | str], nugget: Optional[np.ndarray] = None,
                          name_id_map: dict[str, int] = None) -> tuple[np.ndarray, dict[str, int]]:
        if nugget is None:
            nugget = np.zeros_like(x) + DEFAULT_SP_NUGGET

        if name_id_map is None:
            ids, name_id_map = generate_ids_from_names(name_id_map, names, x)
        else:
            ids = np.array([name_id_map[name] for name in names])

        data = np.zeros(len(x), dtype=SurfacePointsTable.dt)
        data['X'], data['Y'], data['Z'], data['id'], data['nugget'] = x, y, z, ids, nugget
        return data, name_id_map

    @classmethod
    def initialize_empty(cls) -> 'SurfacePointsTable':
        """Initialize an empty SurfacePointsTable.

        Returns:
            SurfacePointsTable: An empty instance of SurfacePointsTable.
        """
        return cls(np.zeros(0, dtype=SurfacePointsTable.dt), {})

    def id_to_name(self, id: int) -> str:
        """Get the name corresponding to a given id.

        Args:
            id (int): The id of the surface point.

        Returns:
            str: The name of the surface point.
        """
        return list(self.name_id_map.keys())[id]

    @property
    def xyz(self) -> np.ndarray:
        return np.array([self.data['X'], self.data['Y'], self.data['Z']]).T

    @property
    def xyz_view(self) -> np.ndarray:
        return self.data[['X', 'Y', 'Z']]

    @xyz_view.setter
    def xyz_view(self, value: np.ndarray):
        self.data['X'], self.data['Y'], self.data['Z'] = value.T

    @property
    def nugget(self) -> np.ndarray:
        return self.data['nugget']

    @nugget.setter
    def nugget(self, value: np.ndarray):
        self.data['nugget'] = value

    @property
    def model_transform(self) -> Transform:
        if self._model_transform is None:
            raise ValueError("Model transform is not set. If you want to use this property use GeoModel.surface_points to get the SurfaceTable with transform attached.")
        return self._model_transform

    @model_transform.setter
    def model_transform(self, value: Transform):
        self._model_transform = value

    def __len__(self):
        return len(self.data)

    def get_surface_points_by_name(self, name: str) -> 'SurfacePointsTable':
        """Get a SurfacePointsTable containing points with the specified name.

        Args:
            name (str): The name of the surface points.

        Returns:
            SurfacePointsTable: A new instance containing the specified surface points.
        """
        return self.get_surface_points_by_id(self.name_id_map[name])

    def get_surface_points_by_id(self, id: int) -> 'SurfacePointsTable':
        """Get a SurfacePointsTable containing points with the specified id.

        Args:
            id (int): The id of the surface points.

        Returns:
            SurfacePointsTable: A new instance containing the specified surface points.
        """
        return SurfacePointsTable(self.data[self.data['id'] == id], self.name_id_map)

    def get_surface_points_by_id_groups(self) -> list['SurfacePointsTable']:
        """Get a list of SurfacePointsTable objects, each containing points with a unique id.

        Returns:
            list[SurfacePointsTable]: A list of SurfacePointsTable objects.
        """
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
