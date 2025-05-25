import dataclasses
import numpy as np
from pydantic import Field


@dataclasses.dataclass
class CustomGrid:
    """Object that contains arbitrary XYZ coordinates.

    Args:
        xyx_coords (numpy.ndarray like): XYZ (in columns) of the desired coordinates

    Attributes:
        values (np.ndarray): XYZ coordinates
    """

    values: np.ndarray = Field(
        exclude=True, 
        default_factory=lambda: np.zeros((0, 3)),
        repr=False
    )
    
       
    def __post_init__(self):
        custom_grid = np.atleast_2d(self.values)
        assert type(custom_grid) is np.ndarray and custom_grid.shape[1] == 3, \
            'The shape of new grid must be (n,3)  where n is the number of' \
            ' points of the grid'

    
    @property
    def length(self):
        return self.values.shape[0]
