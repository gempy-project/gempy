from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
from gempy_engine.core.data.transforms import Transform
from pydantic import Field

from ...optional_dependencies import require_pandas
from ._data_points_helpers import generate_ids_from_names

DEFAULT_ORI_NUGGET = 0.01


# ? Maybe we should merge this with the SurfacePoints class from gempy_engine


@dataclass(init=True)
class OrientationsTable:
    """
    A dataclass to represent a table of orientations in a geological model.
    
    """

    dt = np.dtype([('X', 'f8'), ('Y', 'f8'), ('Z', 'f8'), ('G_x', 'f8'), ('G_y', 'f8'), ('G_z', 'f8'), ('id', 'i4'), ('nugget', 'f8')])  #: The custom data type for the data array.
    data: np.ndarray = Field(
        default=np.zeros(0, dtype=dt),
        exclude=True,
        description="A structured NumPy array holding the X, Y, Z coordinates, gradients G_x, G_y, G_z, id, and nugget of each orientation.",
    )  #: A structured NumPy array holding the X, Y, Z coordinates, id, and nugget of each surface point.
    name_id_map: Optional[dict[str, int]] = None  #: A mapping between orientation names and ids.


    _model_transform: Optional[Transform] = None

    def __post_init__(self):
        # Check if the data array has the correct data type
        if self.data.dtype != OrientationsTable.dt:
            raise ValueError(f"Data array must have the following data type: {OrientationsTable.dt}")

    @classmethod
    def from_arrays(cls, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                    G_x: np.ndarray, G_y: np.ndarray, G_z: np.ndarray,
                    names: Union[Sequence | str], nugget: Optional[np.ndarray] = None,
                    name_id_map: Optional[dict[str, int]] = None) -> 'OrientationsTable':
        """Create an OrientationsTable from arrays.

        Args:
            x (np.ndarray): Array of x-coordinates.
            y (np.ndarray): Array of y-coordinates.
            z (np.ndarray): Array of z-coordinates.
            G_x (np.ndarray): Array of x-components of the gradients.
            G_y (np.ndarray): Array of y-components of the gradients.
            G_z (np.ndarray): Array of z-components of the gradients.
            names (Union[Sequence[str], str]): Sequence of names corresponding to each orientation.
            nugget (Optional[np.ndarray]): Array of nugget values. If None, defaults are used.
            name_id_map (Optional[dict[str, int]]): Mapping between names and ids.

        Returns:
            OrientationsTable: The created OrientationsTable.
        """

        data, name_id_map = cls._data_from_arrays(x, y, z, G_x, G_y, G_z, names, nugget, name_id_map)
        return cls(data, name_id_map)


    @classmethod
    def _data_from_arrays(cls, x, y, z, G_x, G_y, G_z, names, nugget, name_id_map=None) -> tuple[np.ndarray, dict[str, int]]:
        if nugget is None:
            nugget = np.zeros_like(x) + DEFAULT_ORI_NUGGET

        if name_id_map is None:
            ids, name_id_map = generate_ids_from_names(name_id_map, names, x)
        else:
            ids = np.array([name_id_map[name] for name in names])
        data = np.zeros(len(x), dtype=OrientationsTable.dt)
        data['X'], data['Y'], data['Z'], data['G_x'], data['G_y'], data['G_z'], data['id'], data['nugget'] = x, y, z, G_x, G_y, G_z, ids, nugget
        return data, name_id_map

    @classmethod
    def initialize_empty(cls) -> 'OrientationsTable':
        return cls(np.zeros(0, dtype=OrientationsTable.dt))

    @property
    def xyz(self) -> np.ndarray:
        """Get the XYZ coordinates.

        Returns:
            np.ndarray: The XYZ coordinates.
        """
        return np.array([self.data['X'], self.data['Y'], self.data['Z']]).T

    @property
    def xyz_view(self) -> np.ndarray:
        """Get a view of the XYZ coordinates.

        Returns:
            np.ndarray: A view of the XYZ coordinates.
        """
        return self.data[['X', 'Y', 'Z']]

    @xyz_view.setter
    def xyz_view(self, value: np.ndarray):
        """Set the XYZ coordinates.

        Args:
            value (np.ndarray): The new XYZ coordinates.
        """
        self.data['X'], self.data['Y'], self.data['Z'] = value.T

    @property
    def grads(self) -> np.ndarray:
        """Get the gradient components.

        Returns:
            np.ndarray: The gradient components.
        """
        return np.array([self.data['G_x'], self.data['G_y'], self.data['G_z']]).T

    @property
    def grads_view(self) -> np.ndarray:
        """Get a view of the gradient components.

        Returns:
            np.ndarray: A view of the gradient components.
        """
        return self.data[['G_x', 'G_y', 'G_z']]

    @grads_view.setter
    def grads_view(self, value: np.ndarray):
        """Set the gradient components.

        Args:
            value (np.ndarray): The new gradient components.
        """
        self.data['G_x'], self.data['G_y'], self.data['G_z'] = value.T

    @property
    def nugget(self) -> np.ndarray:
        """Get the nugget values.

        Returns:
            np.ndarray: The nugget values.
        """
        return self.data['nugget']

    @property
    def ids(self) -> np.ndarray:
        """Get the IDs.

        Returns:
            np.ndarray: The IDs.
        """
        return self.data['id']

    def get_orientations_by_name(self, name: str) -> 'OrientationsTable':
        """Get orientations by name.

        Args:
            name (str): The name of the orientations.

        Returns:
            OrientationsTable: The orientations corresponding to the given name.
        """
        return self.get_orientations_by_id(self.name_id_map[name])

    def get_orientations_by_id(self, id: int) -> 'OrientationsTable':
        """Get orientations by ID.

        Args:
            id (int): The ID of the orientations.

        Returns:
            OrientationsTable: The orientations corresponding to the given ID.
        """
        return OrientationsTable(self.data[self.data['id'] == id], self.name_id_map)

    def get_orientations_by_id_groups(self) -> list['OrientationsTable']:
        """Get orientations grouped by ID.

        Returns:
            list[OrientationsTable]: A list of OrientationsTable objects, each corresponding to a unique ID.
        """
        ids = np.unique(self.data['id'])
        return [self.get_orientations_by_id(id) for id in ids]

    @classmethod
    def fill_missing_orientations_groups(cls, orientations_groups: list['OrientationsTable'],
                                         surface_points_groups: list['SurfacePointsTable']) -> list['OrientationsTable']:
        """
        Fill Missing Orientations Groups

        Fills in missing orientations in a list of orientations groups based on a list of surface points.

        Args:

        """
        # region Deal with elements without orientations
        if len(surface_points_groups) > len(orientations_groups):
            # Check the ids of the surface points and find the missing ones
            surface_points_ids = [surface_points_group.id for surface_points_group in surface_points_groups]
            orientations_ids = [orientations_group.id for orientations_group in orientations_groups]

            missing_ids = list(set(surface_points_ids) - set(orientations_ids))

            empty_orientations = [cls(data=np.zeros(0, dtype=cls.dt)) for id in missing_ids]  # Create empty orientations

            for empty_orientation, id in zip(empty_orientations, missing_ids):  # Insert the empty orientations in the right position
                orientations_groups.insert(id, empty_orientation)
        # endregion

        return orientations_groups

    @classmethod
    def empty_orientation(cls, id: int) -> 'OrientationsTable':
        zeros = np.zeros(0, dtype=cls.dt)
        zeros['id'] = id
        return cls(data=zeros, name_id_map={})

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
    def model_transform(self) -> Transform:
        if self._model_transform is None:
            raise ValueError("Model transform is not set. If you want to use this property use GeoModel.surface_points to get the SurfaceTable with transform attached.")
        return self._model_transform

    @model_transform.setter
    def model_transform(self, value: Transform):
        self._model_transform = value

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
