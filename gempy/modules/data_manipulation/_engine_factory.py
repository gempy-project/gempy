from typing import Optional

import numpy as np

from ...core.data.grid import Grid
from ...core.data.structural_frame import StructuralFrame

from gempy_engine.core.data import FiniteFault, Orientations, SurfacePoints
from gempy_engine.core.data import engine_grid
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.kernel_classes.faults import FaultsData
from gempy_engine.core.data.transforms import Transform


def interpolation_input_from_structural_frame(geo_model: "gempy.data.GeoModel") -> InterpolationInput:
    import gempy  # ! This is important for type safety
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
        for stack_sol in geo_model.solutions.root_output.outputs:
            weights.append(stack_sol.weights)

    interpolation_input: InterpolationInput = InterpolationInput(
        surface_points=surface_points,
        orientations=orientations,
        grid=grid,
        unit_values=structural_frame.elements_enumerator,  # TODO: Here we will need to pass densities etc.
        weights=weights
    )

    return interpolation_input


def input_data_descriptor_from_geo_model(geo_model: "gempy.data.GeoModel") -> InputDataDescriptor:
    """Build an engine descriptor with finite-fault geometry in engine coordinates."""
    descriptor = geo_model.input_data_descriptor
    faults_input_data = descriptor.stack_structure.faults_input_data
    if faults_input_data is None or not any(
            fault_data is not None and fault_data.finite_fault_defined
            for fault_data in faults_input_data
    ):
        return descriptor

    total_scale = geo_model.input_transform.scale * geo_model.grid.transform.scale
    if not np.allclose(total_scale, total_scale[0]):
        raise ValueError("Finite faults require an isotropic model transform")
    if not (
            np.allclose(geo_model.input_transform.rotation[:2], 0.0)
            and np.allclose(geo_model.grid.transform.rotation[:2], 0.0)
    ):
        raise ValueError("Finite faults do not support model transforms that tilt the vertical axis")

    scale = float(total_scale[0])
    transformed_faults_input_data = []
    for fault_data in faults_input_data:
        if fault_data is None or not fault_data.finite_fault_defined:
            transformed_faults_input_data.append(fault_data)
            continue

        finite_fault = fault_data.finite_fault
        center = np.atleast_2d(finite_fault.center)
        center = geo_model.grid.transform.apply_with_cached_pivot(center)
        center = geo_model.input_transform.apply(center)[0]
        transformed_finite_fault = FiniteFault(
            center=tuple(center),
            strike_radius=_scale_radius(finite_fault.strike_radius, scale),
            dip_radius=_scale_radius(finite_fault.dip_radius, scale),
            taper=finite_fault.taper,
            rotation_deg=finite_fault.rotation_deg,
            spline_control_points=finite_fault.spline_control_points,
        )
        transformed_faults_input_data.append(FaultsData.from_user_input(
            thickness=fault_data.thickness,
            finite_fault=transformed_finite_fault,
        ))

    descriptor.stack_structure.faults_input_data = transformed_faults_input_data
    return descriptor


def _scale_radius(radius: float | tuple[float, float], scale: float) -> float | tuple[float, float]:
    if isinstance(radius, tuple):
        return tuple(value * scale for value in radius)
    return radius * scale


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
    if grid.octree_grid is not None:
        base_grid_resolution = grid.octree_grid.base_resolution
    else:
        base_grid_resolution = np.array([2, 2, 2])

    octree_grid = engine_grid.RegularGrid(
        orthogonal_extent=new_extents,
        regular_grid_shape=base_grid_resolution
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
