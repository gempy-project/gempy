from typing import Union

from gempy import GeoModel, Grid
from gempy.core.grid import GridTypes


def set_section_grid(grid: Grid, section_dict: dict):
    if grid.sections is None:
        grid.create_section_grid(section_dict=section_dict)
    else:
        grid.sections.set_sections(section_dict,
                                   regular_grid=grid.regular_grid)

    set_active_grid(grid, [GridTypes.SECTIONS])
    return grid.sections



def set_active_grid(grid: Grid, grid_type: list[GridTypes], reset: bool = False):
    if reset is True:
        grid.deactivate_all_grids()
    for grid_type in grid_type:
        grid.active_grids[grid_type.value] = True

    print(f'Active grids: {grid.grid_types[grid.active_grids]}')

    return grid
