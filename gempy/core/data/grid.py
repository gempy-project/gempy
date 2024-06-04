import dataclasses

import enum
import warnings
from typing import Union, Optional

import numpy as np

from .grid_modules import topography, RegularGrid, CustomGrid, Sections
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

    dense_grid: Optional[RegularGrid] = None
    octree_grid: Optional[RegularGrid] = None
    custom_grid: Optional[CustomGrid] = None
    topography: Optional[Topography] = None
    sections: Optional[Sections] = None
    centered_grid: Optional[CenteredGrid] = None

    values: np.ndarray = np.empty((0, 3))
    length: np.ndarray = np.empty(0)
    
    _octree_levels: int = -1
    
    def __init__(self, extent=None, resolution=None):
        self.extent = extent
        self.active_grids_bool = np.zeros(6, dtype=bool)
        # Init basic grid empty
        if extent is not None and resolution is not None:
            self.regular_grid = RegularGrid(extent, resolution)
            self.active_grids_bool[0] = True

        # Init optional sections
        self.sections = None

        self.update_grid_values()

    # ? Do we need a active_regular grid property for backwards compatibility?
    @property
    def regular_grid(self):
        warnings.warn('This property is deprecated. Use the dense_grid or octree_grid instead', DeprecationWarning)
        if self.dense_grid is not None and self.octree_grid is not None:
            raise AttributeError('Both dense_grid and octree_grid are active. This is not possible.')
        elif self.dense_grid is not None:
            return self.dense_grid
        elif self.octree_grid is not None:
            return self.octree_grid
        else:
            return None

    @regular_grid.setter
    def regular_grid(self, value):
        warnings.warn('This property is deprecated. Use the dense_grid property instead', DeprecationWarning)
        self.dense_grid = value

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
        self.update_grid_values()

    def __str__(self):
        grid_summary = [f"{g_type} (active: {getattr(self, g_type + '_grid_active')}): {len(getattr(self, g_type + '_grid').values)} points"
                        for g_type in self.grid_types]
        grid_summary_str = "\n".join(grid_summary)
        return f"Grid Object:\n{grid_summary_str}"

    @property
    def active_grids(self) -> np.ndarray:
        return self.grid_types[self.active_grids_bool]

    def create_regular_grid(self, extent=None, resolution=None, set_active=True, *args, **kwargs):
        """
        Set a new regular grid and activate it.

        Args:
            extent (np.ndarray): [x_min, x_max, y_min, y_max, z_min, z_max]
            resolution (np.ndarray): [nx, ny, nz]

        RegularGrid Docs
        """
        self.regular_grid = RegularGrid(extent, resolution)
        return self.regular_grid

    # ? DEP?
    def create_custom_grid_(self, custom_grid: np.ndarray):
        """
        Set a new regular grid and activate it.

        Args:
            custom_grid (np.array): [s0]

        """
        self.custom_grid = CustomGrid(custom_grid)
        self.set_active('custom')

    # ! (miguel, Sep 2023) This has to change a lot
    # ? DEP? 
    def create_topography_(self, source='random', **kwargs):
        """Create a topography grid and activate it.

        Args:
            source:
                * 'gdal':  Load topography from a raster file.
                * 'random': Generate random topography (based on a fractal grid).
                * 'saved': Load topography that was saved with the topography.save() function.
                  This is useful after loading and saving a heavy raster file with gdal once or after saving a
                  random topography with the save() function. This .npy file can then be set as topography.

        Keyword Args:
            source = 'gdal':
                * filepath:   path to raster file, e.g. '.tif', (for all file formats see
                  https://gdal.org/drivers/raster/index.html)

            source = 'random':
                * fd:         fractal dimension, defaults to 2.0
                * d_z:        maximum height difference. If none, last 20% of the model in z direction
                * extent:     extent in xy direction. If none, geo_model.grid.extent
                * resolution: desired resolution of the topography array. If none, geo_model.grid.resoution

            source = 'saved':
                * filepath:   path to the .npy file that was created using the topography.save() function

        Returns:
             :class:gempy.core.data.Topography
        """
        self.topography = topography.Topography(self.regular_grid)

        if source == 'random':
            self.topography.load_random_hills(**kwargs)
        elif source == 'gdal':
            filepath = kwargs.get('filepath', None)
            if filepath is not None:
                self.topography.load_from_gdal(filepath)
            else:
                print('to load a raster file, a path to the file must be provided')
        elif source == 'saved':
            filepath = kwargs.get('filepath', None)
            if filepath is not None:
                self.topography.load_from_saved(filepath)
            else:
                print('path to .npy file must be provided')
        elif source == 'numpy':
            array = kwargs.get('array', None)
            self.topography.set_values(array)
        else:
            raise AttributeError('source must be random, gdal or saved')

        self.set_active('topography')

    def create_section_grid(self, section_dict):
        self.sections = Sections(regular_grid=self.regular_grid, section_dict=section_dict)
        self.set_active('sections')
        return self.sections

    # ? DEP?
    def create_centered_grid(self, centers, radius, resolution=None):
        """Initialize gravity grid. Deactivate the rest of the grids"""
        self.centered_grid = CenteredGrid(centers, radius, resolution)
        # self.active_grids = np.zeros(4, dtype=bool)
        self.set_active('centered')

    def deactivate_all_grids(self):
        """
        Deactivates the active grids array
        :return:
        """
        self.active_grids_bool = np.zeros(5, dtype=bool)
        self.update_grid_values()
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
        self.update_grid_values()
        return self.active_grids_bool

    def set_inactive(self, grid_name: str):
        where = self.grid_types == grid_name
        self.active_grids_bool *= ~where
        self.update_grid_values()
        return self.active_grids_bool

    def update_grid_values(self):
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
