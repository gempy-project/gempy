import enum
import warnings
from typing import Union, Optional

import numpy as np

from gempy.core.grid_modules import grid_types, topography
from gempy.core.grid_modules.topography import Topography


class GridTypes(enum.Enum):
    REGULAR = 0
    CUSTOM = 1
    TOPOGRAPHY = 2
    SECTIONS = 3
    CENTERED = 4



class Grid(object):
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
        active_grids(np.ndarray[bool]): boolean array which controls which type of grid is going to be computed and
         hence on the property `values`.
        regular_grid (:class:`gempy.core.grid_modules.grid_types.RegularGrid`)
        custom_grid (:class:`gempy.core.grid_modules.grid_types.CustomGrid`)
        topography (:class:`gempy.core.grid_modules.grid_types.Topography`)
        sections (:class:`gempy.core.grid_modules.grid_types.Sections`)
        gravity_grid (:class:`gempy.core.grid_modules.grid_types.Gravity`)
    """

    def __init__(self, **kwargs):

        self.values = np.empty((0, 3))
        self.values_r = np.empty((0, 3))
        self.length = np.empty(0)
        self.grid_types = np.array(['regular', 'custom', 'topography', 'sections', 'centered'])
        self.active_grids = np.zeros(5, dtype=bool)
        # All grid types must have values

        # Init optional grids
        self.custom_grid = None
        self.custom_grid_grid_active = False
        self.topography: Optional[Topography] = None
        self.topography_grid_active = False
        self.sections_grid_active = False
        self.centered_grid = None
        self.centered_grid_active = False

        # Init basic grid empty
        self.regular_grid = self.create_regular_grid(set_active=False, **kwargs)
        self.regular_grid_active = False

        # Init optional sections
        self.sections = grid_types.Sections(regular_grid=self.regular_grid)

        self.update_grid_values()

    def __str__(self):
        return 'Grid Object. Values: \n' + np.array2string(self.values)

    def __repr__(self):
        return 'Grid Object. Values: \n' + np.array_repr(self.values)

    def create_regular_grid(self, extent=None, resolution=None, set_active=True, *args, **kwargs):
        """
        Set a new regular grid and activate it.

        Args:
            extent (np.ndarray): [x_min, x_max, y_min, y_max, z_min, z_max]
            resolution (np.ndarray): [nx, ny, nz]

        RegularGrid Docs
        """
        self.regular_grid = grid_types.RegularGrid(extent, resolution, **kwargs)
        if set_active is True:
            self.set_active('regular')
        return self.regular_grid

    def create_custom_grid(self, custom_grid: np.ndarray):
        """
        Set a new regular grid and activate it.

        Args:
            custom_grid (np.array): [s0]

        """
        self.custom_grid = grid_types.CustomGrid(custom_grid)
        self.set_active('custom')

    def create_topography(self, source='random', **kwargs):
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
        self.sections = grid_types.Sections(regular_grid=self.regular_grid, section_dict=section_dict)
        self.set_active('sections')
        return self.sections

    def create_centered_grid(self, centers, radius, resolution=None):
        """Initialize gravity grid. Deactivate the rest of the grids"""
        self.centered_grid = grid_types.CenteredGrid(centers, radius, resolution)
        # self.active_grids = np.zeros(4, dtype=bool)
        self.set_active('centered')

    def deactivate_all_grids(self):
        """
        Deactivates the active grids array
        :return:
        """
        self.active_grids = np.zeros(5, dtype=bool)
        self.update_grid_values()
        return self.active_grids

    def set_active(self, grid_name: Union[str, np.ndarray]):
        """
        Set active a given or several grids
        Args:
            grid_name (str, list):

        """
        warnings.warn('This function is deprecated. Use gempy.set_active_grid instead', DeprecationWarning)
        
        where = self.grid_types == grid_name
        self.active_grids[where] = True
        self.update_grid_values()
        return self.active_grids

    def set_inactive(self, grid_name: str):
        where = self.grid_types == grid_name
        self.active_grids *= ~where
        self.update_grid_values()
        return self.active_grids

    def update_grid_values(self):
        """
        Copy XYZ coordinates from each specific grid to Grid.values for those which are active.

        Returns:
            values

        """
        self.length = np.empty(0)
        self.values = np.empty((0, 3))
        lengths = [0]
        try:
            for e, grid_types in enumerate(
                    [self.regular_grid, self.custom_grid, self.topography, self.sections, self.centered_grid]):
                if self.active_grids[e]:
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
