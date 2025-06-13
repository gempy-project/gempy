from typing import Union, Sequence

import numpy as np

from ..core.data import Grid
from ..core.data.grid_modules import CustomGrid, Sections
from ..core.data.grid_modules.topography import Topography
from ..modules.grids.create_topography import create_random_topography
from ..optional_dependencies import require_subsurface


def set_section_grid(grid: Grid, section_dict: dict):
    if grid.sections is None:
        grid.sections = Sections(
            z_ext=grid.regular_grid.extent[4:],
            section_dict=section_dict
        )
    else:
        grid.sections.set_sections(section_dict,
                                   regular_grid=grid.regular_grid)

    set_active_grid(grid, [Grid.GridTypes.SECTIONS])
    return grid.sections


def set_topography_from_random(grid: Grid, fractal_dimension: float = 2.0, d_z: Union[Sequence, None] = None,
                               topography_resolution: Union[Sequence, None] = None):
    """
    Sets the topography of the grid using a randomly generated topography.

    Args:
        grid (Grid): The grid object on which to set the topography.
        fractal_dimension (float, optional): The fractal dimension of the random topography. Defaults to 2.0.
        d_z (Union[Sequence, None], optional): The sequence of elevation increments for the random topography.
            If None, a default sequence will be used. Defaults to None.
        topography_resolution (Union[Sequence, None], optional): The resolution of the random topography.
            If None, the resolution of the grid's regular grid will be used. Defaults to None.

    Returns:
        The topography object that was set on the grid.

    Example:
        >>> grid = Grid()
        >>> set_topography_from_random(grid, fractal_dimension=1.5, d_z=[0.1, 0.2, 0.3], topography_resolution=[10, 10])

    Note:
        If topography_resolution is None, the resolution of the grid's regular grid will be used.
        If d_z is None, a default sequence of elevation increments will be used.
    """

    if topography_resolution is None:
        topography_resolution = grid.regular_grid.resolution

    random_topography: np.ndarray = create_random_topography(
        extent=grid.regular_grid.extent,
        resolution=topography_resolution,
        dz=d_z,
        fractal_dimension=fractal_dimension
    )

    grid.topography = Topography(
        _regular_grid=grid.regular_grid,
        values_2d=random_topography
    )

    set_active_grid(grid, [Grid.GridTypes.TOPOGRAPHY])
    return grid.topography


def set_topography_from_subsurface_structured_grid(grid: Grid, struct: "subsurface.StructuredData"):
    grid.topography = Topography.from_subsurface_structured_data(struct, grid.regular_grid)
    set_active_grid(grid, [Grid.GridTypes.TOPOGRAPHY])
    return grid.topography


def set_topography_from_arrays(grid: Grid, xyz_vertices: np.ndarray):
    grid.topography = Topography.from_unstructured_mesh(grid.regular_grid, xyz_vertices)
    set_active_grid(grid, [Grid.GridTypes.TOPOGRAPHY])
    return grid.topography


def set_topography_from_file(grid: Grid, filepath: str, crop_to_extent: Union[Sequence, None] = None):
    ss = require_subsurface()
    struct: ss.StructuredData = ss.modules.reader.read_structured_topography(
        path=filepath,
        crop_to_extent=crop_to_extent
    )
    return set_topography_from_subsurface_structured_grid(grid, struct)


def set_custom_grid(grid: Grid, xyz_coord: np.ndarray, reset: bool = False):
    custom_grid = CustomGrid(values=xyz_coord)
    grid.custom_grid = custom_grid

    set_active_grid(grid, grid_type=[Grid.GridTypes.CUSTOM], reset=reset)
    return grid.custom_grid


def set_centered_grid(grid: Grid, centers: np.ndarray, resolution: Sequence[float], radius: Union[float, Sequence[float]]):
    from gempy_engine.core.data.centered_grid import CenteredGrid
    centered_grid = CenteredGrid(
        centers=centers,
        resolution=resolution,
        radius=radius
    )
    grid.centered_grid = centered_grid
    set_active_grid(grid, [Grid.GridTypes.CENTERED])
    return grid.centered_grid


def set_topography_from_gdal():
    raise NotImplementedError("This is not implemented yet")


def set_topography_from_array():
    raise NotImplementedError("This is not implemented yet")


def set_active_grid(grid: Grid, grid_type: list[Grid.GridTypes], reset: bool = False):
    if reset is True:
        grid.active_grids = Grid.GridTypes.NONE
    for grid_type in grid_type:
        grid.active_grids |= grid_type

    print(f'Active grids: {grid.active_grids}')

    return grid
