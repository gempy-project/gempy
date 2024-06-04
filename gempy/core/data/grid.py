import dataclasses
import enum
import numpy as np
from typing import Optional

from gempy_engine.core.data.centered_grid import CenteredGrid
from .grid_modules import RegularGrid, CustomGrid, Sections
from .grid_modules.topography import Topography


@dataclasses.dataclass
class Grid:
    class GridTypes(enum.Flag):
        OCTREE = 2**0
        DENSE = 2**1
        CUSTOM = 2**2
        TOPOGRAPHY = 2**3
        SECTIONS = 2**4
        CENTERED = 2**5
        NONE = 2**10

    # ? What should we do with the extent?
    _extent: Optional[np.ndarray]  # * Model extent should be cross grid

    _octree_grid: Optional[RegularGrid] = None
    _dense_grid: Optional[RegularGrid] = None
    _custom_grid: Optional[CustomGrid] = None
    _topography: Optional[Topography] = None
    _sections: Optional[Sections] = None
    _centered_grid: Optional[CenteredGrid] = None

    values: np.ndarray = np.empty((0, 3))
    length: np.ndarray = np.empty(0)
    
    _active_grids = GridTypes.NONE
    
    _octree_levels: int = -1
    
    def __init__(self, extent=None, resolution=None):
        self.extent = extent
        # Init basic grid empty
        if extent is not None and resolution is not None:
            self.dense_grid = RegularGrid(extent, resolution)

    def __str__(self):
        active_grid_types_str = [g_type for g_type in self.GridTypes if self.active_grids & g_type]

        grid_summary = [f"{g_type} (active: {getattr(self, g_type + '_grid_active')}): {len(getattr(self, g_type + '_grid').values)} points"
                        for g_type in active_grid_types_str]
        grid_summary_str = "\n".join(grid_summary)
        return f"Grid Object:\n{grid_summary_str}"

    @property
    def extent(self):
        if self._extent is None:
            # Try to get the extent from the dense or octree grid if those are also none raise an error
            if self.dense_grid is not None:
                return self.dense_grid.extent
            elif self.octree_grid is not None:
                return self.octree_grid.extent
            else:
                raise AttributeError('Extent is not defined')
        else:
            return self._extent
        
    @extent.setter
    def extent(self, value):
        self._extent = value
        
    
    @property
    def active_grids(self):
        return self._active_grids
    
    @active_grids.setter
    def active_grids(self, value):
        self._active_grids = value
        self._update_values()

    @property
    def dense_grid(self):
        return self._dense_grid
    
    @dense_grid.setter
    def dense_grid(self, value):
        self._dense_grid = value
        self.active_grids |= self.GridTypes.DENSE
        self._update_values()
    
    @property
    def octree_grid(self):
        return self._octree_grid
    
    @octree_grid.setter
    def octree_grid(self, value):
        self._octree_grid = value
        self.active_grids |= self.GridTypes.OCTREE
        self._update_values()
        
    @property
    def custom_grid(self):
        return self._custom_grid
    
    @custom_grid.setter
    def custom_grid(self, value):
        self._custom_grid = value
        self.active_grids |= self.GridTypes.CUSTOM
        self._update_values()
        
    @property
    def topography(self):
        return self._topography
    
    @topography.setter
    def topography(self, value):
        self._topography = value
        self.active_grids |= self.GridTypes.TOPOGRAPHY
        self._update_values()
        
    @property
    def sections(self):
        return self._sections
    
    @sections.setter
    def sections(self, value):
        self._sections = value
        self.active_grids |= self.GridTypes.SECTIONS
        self._update_values()
        
    @property
    def centered_grid(self):
        return self._centered_grid
    
    @centered_grid.setter
    def centered_grid(self, value):
        self._centered_grid = value
        self.active_grids |= self.GridTypes.CENTERED
        self._update_values()
    
    
    @property
    def regular_grid(self):
        raise AttributeError('This property is deprecated. Use the dense_grid or octree_grid instead')

    @regular_grid.setter
    def regular_grid(self, value):
        raise AttributeError('This property is deprecated. Use the dense_grid or octree_grid instead')

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
        self.active_grids |= self.GridTypes.OCTREE

    def _update_values(self):
        values = []
        
        if self.GridTypes.OCTREE in self.active_grids:
            values.append(self.octree_grid.values)
        if self.GridTypes.DENSE in self.active_grids:
            values.append(self.dense_grid.values)
        if self.GridTypes.CUSTOM in self.active_grids:
            values.append(self.custom_grid.values)
        if self.GridTypes.TOPOGRAPHY in self.active_grids:
            values.append(self.topography.values)
        if self.GridTypes.SECTIONS in self.active_grids:
            values.append(self.sections.values)
        if self.GridTypes.CENTERED in self.active_grids:
            values.append(self.centered_grid.values)
        
        self.values = np.concatenate(values)

        return self.values

  
    def get_section_args(self, section_name: str):
        # TODO: This method should be part of the sections
        # assert type(section_name) is str, 'Only one section type can be retrieved'
        l0, l1 = self.get_grid_args('sections')
        where = np.where(self.sections.names == section_name)[0][0]
        return l0 + self.sections.length[where], l0 + self.sections.length[where + 1]
