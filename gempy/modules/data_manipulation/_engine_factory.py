from typing import Optional

import numpy as np

from ...core.data.grid import Grid
from ...core.data.structural_frame import StructuralFrame

from gempy_engine.core.data import SurfacePoints, Orientations
from gempy_engine.core.data import engine_grid
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.transforms import Transform


def interpolation_input_from_structural_frame(geo_model: "gempy.data.GeoModel") -> InterpolationInput:
    import gempy # ! This is important for type safety
    geo_model: gempy.data.GeoModel = geo_model

    _legacy_factor = 0

    structural_frame: StructuralFrame = geo_model.structural_frame
    input_transform: Transform = geo_model.input_transform
    grid: Grid = geo_model.grid

    total_transform: Transform = input_transform + grid.transform

    surface_points_copy_transformed = geo_model.surface_points_copy_transformed
    surface_points: SurfacePoints = SurfacePoints(
        sp_coords=geo_model.surface_points_copy_transformed.xyz,
        nugget_effect_scalar=surface_points_copy_transformed.nugget
    )

    orientations_copy_transformed = geo_model.orientations_copy_transformed
    orientations: Orientations = Orientations(
        dip_positions=orientations_copy_transformed.xyz,
        dip_gradients=orientations_copy_transformed.grads,
        nugget_effect_grad=orientations_copy_transformed.nugget
    )

    grid: engine_grid.EngineGrid = _apply_input_transform_to_grids(
        grid=grid,
        input_transform=input_transform,
        extent_transformed=geo_model.extent_transformed_transformed_by_input
    )

    weights = []
    if geo_model.solutions is not None:
        for stack_sol in geo_model.solutions.root_output.outputs_centers:
            weights.append(stack_sol.weights)
    

    interpolation_input: InterpolationInput = InterpolationInput(
        surface_points=surface_points,
        orientations=orientations,
        grid=grid,
        unit_values=structural_frame.elements_ids,  # TODO: Here we will need to pass densities etc.
        weights=weights
    )

    return interpolation_input


def _apply_input_transform_to_grids(grid: Grid, input_transform: Transform, extent_transformed: np.ndarray) -> engine_grid.EngineGrid:
    new_extents = extent_transformed
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
