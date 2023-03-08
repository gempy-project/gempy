import numpy as np

from gempy import Project
from gempy_engine.core.data import SurfacePoints, Orientations, InterpolationOptions
from gempy_engine.core.data.grid import RegularGrid, Grid
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor, StacksStructure, StackRelationType, TensorsStructure
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.kernel_classes.kernel_functions import AvailableKernelFunctions


def gempy_project_to_interpolation_input(geo_model: Project) -> InterpolationInput:
    surface_points: SurfacePoints = SurfacePoints(
        sp_coords=geo_model._surface_points.df[['X_c', 'Y_c', 'Z_c']].values,
    )

    orientations: Orientations = Orientations(
        dip_positions=geo_model._orientations.df[['X_c', 'Y_c', 'Z_c']].values,
        dip_gradients=geo_model._orientations.df[['G_x', 'G_y', 'G_z']].values
    )

    regular_grid: RegularGrid = RegularGrid(
        extent=geo_model.grid.regular_grid.extent_r,
        regular_grid_shape=geo_model.grid.regular_grid.resolution,
    )

    grid: Grid = Grid(
        values=regular_grid.values,
        regular_grid=regular_grid
    )

    interpolation_input: InterpolationInput = InterpolationInput(
        surface_points=surface_points,
        orientations=orientations,
        grid=grid,
        unit_values=geo_model._surfaces.df['id'].values,  # * This can be optional
    )
    
    return interpolation_input


def gempy_project_to_input_data_descriptor(geo_model: Project) -> InputDataDescriptor:
    # @formatter:off
    stack_structure: StacksStructure = StacksStructure(
        number_of_points_per_stack       = geo_model.additional_data.structure_data.df.loc['values', 'len series surface_points'],
        number_of_orientations_per_stack = geo_model.additional_data.structure_data.df.loc['values', 'len series orientations'],
        number_of_surfaces_per_stack     = geo_model.additional_data.structure_data.df.loc['values', 'number surfaces per series'],
        masking_descriptor               = [StackRelationType.FAULT                                , StackRelationType.ERODE      , False],
        faults_relations                 = geo_model._faults.faults_relations_df.values
    )
    # @formatter:on

    print(stack_structure)

    tensor_struct = TensorsStructure(
        number_of_points_per_surface=geo_model.additional_data.structure_data.df.loc['values', 'len surfaces surface_points']
    )

    input_data_descriptor = InputDataDescriptor(
        tensors_structure=tensor_struct,
        stack_structure=stack_structure
    )
    
    return input_data_descriptor


def gempy_project_to_interpolation_options(geo_model: Project) -> InterpolationOptions:
    rescaling_factor: float = geo_model._additional_data.rescaling_data.df.loc['values', 'rescaling factor']
    shift: np.array = geo_model._additional_data.rescaling_data.df.loc['values', 'centers']

    # @formatter:off
    options                     = InterpolationOptions(
        range                   = geo_model._additional_data.kriging_data.df.loc['values', 'range'] / rescaling_factor,
        c_o                     = geo_model._additional_data.kriging_data.df.loc['values', '$C_o$'] / rescaling_factor,
        uni_degree              = 1,
        number_dimensions       = 3,
        kernel_function         = AvailableKernelFunctions.cubic,
        dual_contouring         = True,
        compute_scalar_gradient = False,
        number_octree_levels    = 1
    )
    # @formatter:on

    return options
    