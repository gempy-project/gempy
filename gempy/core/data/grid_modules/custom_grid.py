import dataclasses
from typing import Annotated, Mapping

import numpy as np
from pydantic import Field


@dataclasses.dataclass
class CustomGrid:
    """Object that contains arbitrary XYZ coordinates.

    Args:
        xyx_coords (numpy.ndarray like): XYZ (in columns) of the desired coordinates

    Attributes:
        values (np.ndarray): XYZ coordinates
        segments (list): Named consecutive runs of ``values``, as ``(name, number of points)``: one per
            object to paint, say. Empty when the points are not named. Saved with the grid's binary metadata.
    """

    values: np.ndarray = Field(
        exclude=True, 
        default_factory=lambda: np.zeros((0, 3)),
        repr=False
    )
    segments: Annotated[list, Field(exclude=True)] = dataclasses.field(default_factory=list, repr=False)
    
       
    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=np.float64)
        custom_grid = np.atleast_2d(self.values)
        assert type(custom_grid) is np.ndarray and custom_grid.shape[1] == 3, \
            'The shape of new grid must be (n,3)  where n is the number of' \
            ' points of the grid'
        self.segments = [(str(name), int(length)) for name, length in self.segments]
        if len({name for name, _ in self.segments}) != len(self.segments):
            raise ValueError("Custom grid segment names must be unique.")

    @classmethod
    def from_segments(cls, segments: Mapping[str, np.ndarray]) -> "CustomGrid":
        """One named run of points per entry, in order."""
        arrays = {str(name): np.asarray(xyz, dtype=np.float64).reshape(-1, 3) for name, xyz in segments.items()}
        values = np.concatenate(list(arrays.values())) if arrays else np.zeros((0, 3))
        return cls(values=values, segments=[(name, len(xyz)) for name, xyz in arrays.items()])

    @property
    def length(self):
        return self.values.shape[0]

    def segment_slice(self, name: str) -> slice:
        """Where the points of segment ``name`` are in ``values`` and in any result on this grid."""
        self._check_segments()
        start = 0
        for segment, length in self.segments:
            if segment == name:
                return slice(start, start + length)
            start += length
        raise KeyError(f"The custom grid has no segment '{name}'.")

    def split(self, values) -> dict:
        """``values`` on this grid (one row per point), by segment name."""
        self._check_segments()
        if len(values) != self.length:
            raise ValueError(f"Expected {self.length} values on the custom grid, got {len(values)}.")
        return {name: values[self.segment_slice(name)] for name, _ in self.segments}

    def _check_segments(self):
        if sum(length for _, length in self.segments) != self.length:
            raise ValueError("The custom grid's segments do not cover its points.")
