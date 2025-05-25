from __future__ import annotations   # Python 3.7+ only
import dataclasses
import numpy as np
from pydantic import Field
from typing import Tuple, Dict, List, Optional, Any

from ..core_utils import calculate_line_coordinates_2points
from ..encoders.converters import short_array_type
from ....optional_dependencies import require_pandas



@dataclasses.dataclass
class Sections:
    """
    Object that creates a grid of cross sections between two points.

    Args:
        regular_grid: Model.grid.regular_grid
        section_dict: {'section name': ([p1_x, p1_y], [p2_x, p2_y], [xyres, zres])}
    """

    """
       Pydantic v2 model of your original Sections class.
       All computed fields are initialized with model_validator.
       """

    # user‐provided inputs
    
    z_ext: Tuple[float, float] | short_array_type
    section_dict: Dict[
        str,
        Tuple[
            Tuple[float, float],  # start
            Tuple[float, float],  # stop
            Tuple[int,   int]     # resolution
        ]
    ]

    # computed/internal (will be serialized too unless excluded)
    names: short_array_type = Field(default=np.array([]), exclude=True)
    points: List[List[Tuple[float, float]]] = Field(default_factory=list, exclude=True)
    resolution: List[Tuple[int, int]] = Field(default_factory=list, exclude=True)
    length: np.ndarray = Field(default_factory=lambda: np.array([0]), exclude=True)
    dist: np.ndarray = Field(default_factory=lambda: np.array([]), exclude=True)
    df: Optional[Any] = Field(default=None, exclude=True)
    values: np.ndarray = Field(default_factory=lambda: np.empty((0, 3)), exclude=True)
    extent: Optional[np.ndarray] = Field(default=None, exclude=True)

    def __post_init__(self):
        self.initialize_computations()
        pass

    def initialize_computations(self):
        # copy names
        self.names = np.array(list(self.section_dict.keys()))

        # build points/resolution/length
        self._get_section_params()
        # compute distances
        self._calculate_all_distances()
        # re-build DataFrame
        pd = require_pandas()
        df = pd.DataFrame.from_dict(
            data=self.section_dict,
            orient="index",
            columns=["start", "stop", "resolution"],
        )
        df["dist"] = self.dist
        self.df = df

        # compute the XYZ grid
        self._compute_section_coordinates()

    def _repr_html_(self):
        return self.df.to_html()

    def __repr__(self):
        return self.df.to_string()

    def show(self):
        pass

    def set_sections(self, section_dict, regular_grid=None, z_ext=None):
        pd = require_pandas()
        self.section_dict = section_dict
        if regular_grid is not None:
            self.z_ext = regular_grid.extent[4:]
            
        self.initialize_computations()

    def _get_section_params(self):
        self.points = []
        self.resolution = []
        self.length = [0]

        for i, section in enumerate(self.names):
            points = [self.section_dict[section][0], self.section_dict[section][1]]
            assert points[0] != points[
                1], 'The start and end points of the section must not be identical.'

            self.points.append(points)
            self.resolution.append(self.section_dict[section][2])
            self.length = np.append(self.length, self.section_dict[section][2][0] *
                                    self.section_dict[section][2][1])
        self.length = np.array(self.length).cumsum()

    def _calculate_all_distances(self):
        self.coordinates = np.array(self.points).ravel().reshape(-1,
                                                                 4)  # axis are x1,y1,x2,y2
        self.dist = np.sqrt(np.diff(self.coordinates[:, [0, 2]]) ** 2 + np.diff(
            self.coordinates[:, [1, 3]]) ** 2)

    def _compute_section_coordinates(self):
        for i in range(len(self.names)):
            xy = calculate_line_coordinates_2points(self.coordinates[i, :2],
                                                    self.coordinates[i, 2:],
                                                    self.resolution[i][0])
            zaxis = np.linspace(self.z_ext[0], self.z_ext[1], self.resolution[i][1],
                                dtype="float64")
            X, Z = np.meshgrid(xy[:, 0], zaxis, indexing='ij')
            Y, _ = np.meshgrid(xy[:, 1], zaxis, indexing='ij')
            xyz = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
            if i == 0:
                self.values = xyz
            else:
                self.values = np.vstack((self.values, xyz))

    def generate_axis_coord(self):
        for i, name in enumerate(self.names):
            xy = calculate_line_coordinates_2points(
                self.coordinates[i, :2],
                self.coordinates[i, 2:],
                self.resolution[i][0]
            )
            yield name, xy

    def get_section_args(self, section_name: str):
        where = np.where(self.names == section_name)[0][0]
        return self.length[where], self.length[where + 1]

    def get_section_grid(self, section_name: str):
        l0, l1 = self.get_section_args(section_name)
        return self.values[l0:l1]
