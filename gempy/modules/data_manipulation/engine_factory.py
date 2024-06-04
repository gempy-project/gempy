from typing import Optional

import numpy as np

from ...core.data.grid import Grid
from ...core.data.structural_frame import StructuralFrame

from gempy_engine.core.data import SurfacePoints, Orientations
from gempy_engine.core.data import engine_grid
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.transforms import Transform


def interpolation_input_from_structural_frame(structural_frame: StructuralFrame, grid: Grid,
                                              input_transform: Transform) -> InterpolationInput:
    _legacy_factor = 0

    if LEGACY_COORDS := False:
        _legacy_factor = 0.5

    total_transform: Transform = input_transform + grid.dense_grid.transform

    surface_points_copy = structural_frame.surface_points_copy
    surface_points: SurfacePoints = SurfacePoints(
        sp_coords=total_transform.apply(surface_points_copy.xyz) + _legacy_factor,
        nugget_effect_scalar=surface_points_copy.nugget
    )

    orientations_copy = structural_frame.orientations_copy
    orientations: Orientations = Orientations(
        dip_positions=total_transform.apply(orientations_copy.xyz) + _legacy_factor,
        dip_gradients=total_transform.transform_gradient(orientations_copy.grads),
        nugget_effect_grad=orientations_copy.nugget
    )

    grid: engine_grid.EngineGrid = _apply_input_transform_to_grids(grid, input_transform)


    interpolation_input: InterpolationInput = InterpolationInput(
        surface_points=surface_points,
        orientations=orientations,
        grid=grid,
        unit_values=structural_frame.elements_ids  # TODO: Here we will need to pass densities etc.
    )

    return interpolation_input


def _apply_input_transform_to_grids(grid: Grid, input_transform: Transform) -> engine_grid.EngineGrid:
    transformed = input_transform.apply(grid.bounding_box)  # ! grid already has the grid transform applied
    grid.regular_grid.input_transform = input_transform
    new_extents = np.array([transformed[:, 0].min(), transformed[:, 0].max(),
                            transformed[:, 1].min(), transformed[:, 1].max(),
                            transformed[:, 2].min(), transformed[:, 2].max()])

    # Initialize all variables to None
    octree_grid: Optional[engine_grid.RegularGrid] = None
    regular_grid: Optional[engine_grid.RegularGrid] = None
    custom_values: Optional[engine_grid.GenericGrid] = None
    topography_values: Optional[engine_grid.GenericGrid] = None
    section_values: Optional[engine_grid.GenericGrid] = None
    centered_grid: Optional[engine_grid.CenteredGrid] = None

    if grid.GridTypes.DENSE in grid.active_grids:
        regular_grid = engine_grid.RegularGrid(
            orthogonal_extent=new_extents,
            regular_grid_shape=grid.dense_grid.resolution,
        )
    if grid.GridTypes.CUSTOM in grid.active_grids and grid.custom_grid is not None:
        custom_values = engine_grid.GenericGrid(values=input_transform.apply(grid.custom_grid.values))
    if grid.GridTypes.TOPOGRAPHY in grid.active_grids and grid.topography is not None:
        topography_values = engine_grid.GenericGrid(values=input_transform.apply(grid.topography.values))
    if grid.GridTypes.SECTIONS in grid.active_grids and grid.sections is not None:
        section_values = engine_grid.GenericGrid(values=input_transform.apply(grid.sections.values))
    if grid.GridTypes.CENTERED in grid.active_grids and grid.centered_grid is not None:
        centered_grid = engine_grid.CenteredGrid(
            centers=input_transform.apply(grid.centered_grid.centers),
            radius=input_transform.scale_points(np.atleast_2d(grid.centered_grid.radius))[0],
            resolution=grid.centered_grid.resolution
        )
    octree_grid = engine_grid.RegularGrid(
        orthogonal_extent=new_extents,
        regular_grid_shape=np.array([2, 2, 2])
    )
    grid: engine_grid.EngineGrid = engine_grid.EngineGrid(  # * Here we convert the GemPy grid to the
        octree_grid=octree_grid,  # BUG: Adapt the engine to deal with this
        dense_grid=regular_grid,
        topography=topography_values,
        sections=section_values,
        custom_grid=custom_values,
        geophysics_grid=centered_grid
    )
    return grid
