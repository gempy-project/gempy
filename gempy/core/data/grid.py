import dataclasses

import enum
import warnings
from typing import Union, Optional

import numpy as np

from .grid_modules import RegularGrid, CustomGrid, Sections
from .grid_modules.topography import Topography
from gempy_engine.core.data.centered_grid import CenteredGrid


class GridTypes(enum.Enum):
    REGULAR = 0
    CUSTOM = 1
    TOPOGRAPHY = 2
    SECTIONS = 3
    CENTERED = 4


@dataclasses.dataclass
class Grid:
    """ Class to generate grids.

    This class is used to create points to evaluate the geological model. 
    This class serves a container which transmits the XYZ coordinates to the
    interpolator. There are several type of grid objects feeding into the Grid class

    Args:
         **kwargs: See below

    Keyword Args:
         regular (:class:`gempy.core.grid_modules.grid_types.RegularGrid`): [s0]
         custom (:class:`gempy.core.grid_modules.grid_types.CustomGrid`): [s1]
         topography (:class:`gempy.core.grid_modules.grid_types.Topography`): [s2]
         sections (:class:`gempy.core.grid_modules.grid_types.Sections`): [s3]
         gravity (:class:`gempy.core.grid_modules.grid_types.Gravity`):

    Attributes:
        values (np.ndarray): coordinates where the model is going to be evaluated. This is the coordinates
         concatenation of all active grids.
        values_r (np.ndarray): rescaled coordinates where the model is going to be evaluated
        length (np.ndarray):I a array which contain the slicing index for each grid type in order. The first element will
         be 0, the second the length of the regular grid; the third custom and so on. This can be used to slice the
         solutions correspondent to each of the grids
        grid_types(np.ndarray[str]): names of the current grids of GemPy
        active_grids_bool(np.ndarray[bool]): boolean array which controls which type of grid is going to be computed and
         hence on the property `values`.
        regular_grid (:class:`gempy.core.grid_modules.grid_types.RegularGrid`)
        custom_grid (:class:`gempy.core.grid_modules.grid_types.CustomGrid`)
        topography (:class:`gempy.core.grid_modules.grid_types.Topography`)
        sections (:class:`gempy.core.grid_modules.grid_types.Sections`)
        gravity_grid (:class:`gempy.core.grid_modules.grid_types.Gravity`)
    """

    class GridTypes(enum.Flag):
        REGULAR = 2**1
        CUSTOM = 2**2
        TOPOGRAPHY = 2**3
        SECTIONS = 2**4
        CENTERED = 2**5


    # ? What should we do with the extent?
    extent: Optional[np.ndarray]  # * Model extent should be cross grid

    active_grids_bool: np.ndarray

    _dense_grid: Optional[RegularGrid] = None
    _octree_grid: Optional[RegularGrid] = None
    _custom_grid: Optional[CustomGrid] = None
    _topography: Optional[Topography] = None
    _sections: Optional[Sections] = None
    centered_grid: Optional[CenteredGrid] = None

    values: np.ndarray = np.empty((0, 3))
    length: np.ndarray = np.empty(0)
    
    _octree_levels: int = -1
    
    def __init__(self, extent=None, resolution=None):
        self.extent = extent
        self.active_grids_bool = np.zeros(6, dtype=bool)
        # Init basic grid empty
        if extent is not None and resolution is not None:
            self.dense_grid = RegularGrid(extent, resolution)

        # Init optional sections
        self.sections = None


    @property
    def dense_grid(self):
        return self._dense_grid
    
    @dense_grid.setter
    def dense_grid(self, value):
        self._dense_grid = value
        self.active_grids_bool[0] = True
        self._update_values()
    
    @property
    def octree_grid(self):
        return self._octree_grid
    
    @octree_grid.setter
    def octree_grid(self, value):
        self._octree_grid = value
        self.active_grids_bool[1] = True
        self._update_values()
        
    @property
    def custom_grid(self):
        return self._custom_grid
    
    @custom_grid.setter
    def custom_grid(self, value):
        self._custom_grid = value
        self.active_grids_bool[2] = True
        self._update_values()
        
    @property
    def topography(self):
        return self._topography
    
    @topography.setter
    def topography(self, value):
        self._topography = value
        self.active_grids_bool[3] = True
        self._update_values()
        
    @property
    def sections(self):
        return self._sections
    
    @sections.setter
    def sections(self, value):
        self._sections = value
        self.active_grids_bool[4] = True
        self._update_values()
        
    
    # ? Do we need a active_regular grid property for backwards compatibility?
    # @property
    # def regular_grid(self):
    #     warnings.warn('This property is deprecated. Use the dense_grid or octree_grid instead', DeprecationWarning)
    #     if self.dense_grid is not None and self.octree_grid is not None:
    #         raise AttributeError('Both dense_grid and octree_grid are active. This is not possible.')
    #     elif self.dense_grid is not None:
    #         return self.dense_grid
    #     elif self.octree_grid is not None:
    #         return self.octree_grid
    #     else:
    #         return None
    # 
    # @regular_grid.setter
    # def regular_grid(self, value):
    #     warnings.warn('This property is deprecated. Use the dense_grid property instead', DeprecationWarning)
    #     self.dense_grid = value

    @property
    def octree_levels(self):
        return self._octree_levels

    @octree_levels.setter
    def octree_levels(self, value):
        self._octree_levels = value
        self.octree_grid = RegularGrid(
            extent=self.extent,
            resolution=np.array([2 ** value] * 3),
        )
        self.active_grids_bool[5] = True
        self._update_values()

    def __str__(self):
        grid_summary = [f"{g_type} (active: {getattr(self, g_type + '_grid_active')}): {len(getattr(self, g_type + '_grid').values)} points"
                        for g_type in self.grid_types]
        grid_summary_str = "\n".join(grid_summary)
        return f"Grid Object:\n{grid_summary_str}"

    @property
    def active_grids(self) -> np.ndarray:
        # TODO: I think we need to update this
        return self.grid_types[self.active_grids_bool]


    def deactivate_all_grids(self):
        """
        Deactivates the active grids array
        :return:
        """
        self.active_grids_bool = np.zeros(5, dtype=bool)
        self._update_values()
        return self.active_grids_bool

    def set_active(self, grid_name: Union[str, np.ndarray]):
        """
        Set active a given or several grids
        Args:
            grid_name (str, list):

        """
        warnings.warn('This function is deprecated. Use gempy.set_active_grid instead', DeprecationWarning)

        where = self.grid_types == grid_name
        self.active_grids_bool[where] = True
        self._update_values()
        return self.active_grids_bool

    def set_inactive(self, grid_name: str):
        where = self.grid_types == grid_name
        self.active_grids_bool *= ~where
        self._update_values()
        return self.active_grids_bool

    def _update_values(self):
        """
        Copy XYZ coordinates from each specific grid to Grid.values for those which are active.

        Returns:
            values

        """
        self.length = np.empty(0)
        self.values = np.empty((0, 3))
        lengths = [0]
        all_grids = [self.regular_grid, self.custom_grid, self.topography, self.sections, self.centered_grid]
        try:
            for e, grid_types in enumerate(all_grids):
                if self.active_grids_bool[e]:
                    self.values = np.vstack((self.values, grid_types.values))
                    lengths.append(grid_types.values.shape[0])
                else:
                    lengths.append(0)
        except AttributeError:
            raise AttributeError('Grid type does not exist yet. Set the grid before activating it.')

        self.length = np.array(lengths).cumsum()
        return self.values

    def get_grid_args(self, grid_name: str):
        assert type(grid_name) is str, 'Only one grid type can be retrieved'
        assert grid_name in self.grid_types, 'possible grid types are ' + str(self.grid_types)
        where = np.where(self.grid_types == grid_name)[0][0]
        return self.length[where], self.length[where + 1]

    def get_grid(self, grid_name: str):
        assert type(grid_name) is str, 'Only one grid type can be retrieved'

        l_0, l_1 = self.get_grid_args(grid_name)
        return self.values[l_0:l_1]

    def get_section_args(self, section_name: str):
        # assert type(section_name) is str, 'Only one section type can be retrieved'
        l0, l1 = self.get_grid_args('sections')
        where = np.where(self.sections.names == section_name)[0][0]
        return l0 + self.sections.length[where], l0 + self.sections.length[where + 1]
