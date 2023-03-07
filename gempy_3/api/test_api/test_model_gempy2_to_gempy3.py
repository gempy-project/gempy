import gempy as gp
import pandas as pn
import numpy as np
import os

import gempy_engine
from gempy.core.grid_modules.grid_types import RegularGrid
from gempy_engine.core.data import SurfacePoints, Orientations, InterpolationOptions
from gempy_engine.core.data.grid import Grid
from gempy_engine.core.data.input_data_descriptor import StacksStructure, StackRelationType, TensorsStructure, InputDataDescriptor
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.kernel_classes.kernel_functions import AvailableKernelFunctions
from gempy_engine.core.data.solutions import Solutions

input_path = os.path.dirname(__file__) + '/../../../test/input_data'

# ## Preparing the Python environment
#
# For modeling with GemPy, we first need to import it. We should also import any other packages we want to
# utilize in our Python environment.Typically, we will also require `NumPy` and `Matplotlib` when working
# with GemPy. At this point, we can further customize some settings as desired, e.g. the size of figures or,
# as we do here, the way that `Matplotlib` figures are displayed in our notebook (`%matplotlib inline`).


# These two lines are necessary only if GemPy is not installed
import sys, os

sys.path.append("../..")


def create_interpolator():
    m = gp.create_model('JustInterpolator')
    return gp.set_interpolator(m, theano_optimizer='fast_run')


def load_model():
    verbose = False
    geo_model = gp.create_model('Model_Tuto1-1')

    # Importing the data from CSV-files and setting extent and resolution
    gp.init_data(geo_model, [0, 2000., 0, 2000., 0, 2000.], [50, 50, 50],
                 path_o=input_path + "/simple_fault_model_orientations.csv",
                 path_i=input_path + "/simple_fault_model_points.csv", default_values=True)

    df_cmp_i = gp.get_data(geo_model, 'surface_points')
    df_cmp_o = gp.get_data(geo_model, 'orientations')

    df_o = pn.read_csv(input_path + "/simple_fault_model_orientations.csv")
    df_i = pn.read_csv(input_path + "/simple_fault_model_points.csv")

    assert not df_cmp_i.empty, 'data was not set to dataframe'
    assert not df_cmp_o.empty, 'data was not set to dataframe'
    assert df_cmp_i.shape[0] == df_i.shape[0], 'data was not set to dataframe'
    assert df_cmp_o.shape[0] == df_o.shape[0], 'data was not set to dataframe'

    if verbose:
        gp.get_data(geo_model, 'surface_points').head()

    return geo_model


def map_sequential_pile(load_model):
    geo_model = load_model

    # TODO decide what I do with the layer order

    gp.map_stack_to_surfaces(geo_model, {"Fault_Series": 'Main_Fault',
                                         "Strat_Series": ('Sandstone_2', 'Siltstone',
                                                          'Shale', 'Sandstone_1', 'basement')},
                             remove_unused_series=True)

    geo_model.set_is_fault(['Fault_Series'])
    return geo_model


def test_set_gempy3_input():
    geo_model = load_model()
    geo_model = map_sequential_pile(geo_model)

    surface_points: SurfacePoints = SurfacePoints(
        sp_coords=geo_model._surface_points.df[['X_c', 'Y_c', 'Z_c']].values,
    )

    orientations: Orientations = Orientations(
        dip_positions=geo_model._orientations.df[['X_c', 'Y_c', 'Z_c']].values,
        dip_gradients=geo_model._orientations.df[['G_x', 'G_y', 'G_z']].values
    )

    regular_grid: RegularGrid = RegularGrid(
        extent=geo_model.grid.regular_grid.extent_r,
        resolution=geo_model.grid.regular_grid.resolution,
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

    print(interpolation_input)

    faults_relations = geo_model._faults.faults_relations_df.values
    # @formatter:off
    stack_structure: StacksStructure = StacksStructure(
        number_of_points_per_stack       = geo_model.additional_data.structure_data.df.loc['values', 'len series surface_points'],
        number_of_orientations_per_stack = geo_model.additional_data.structure_data.df.loc['values', 'len series orientations'],
        number_of_surfaces_per_stack     = geo_model.additional_data.structure_data.df.loc['values', 'number surfaces per series'],
        masking_descriptor               = [StackRelationType.FAULT                                , StackRelationType.ERODE      , False],
        faults_relations                 = faults_relations
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

    print(input_data_descriptor)
    rescaling_factor: float = geo_model._additional_data.rescaling_data.df.loc['values', 'rescaling factor']

    # @formatter:off
    options = InterpolationOptions(
        range             = geo_model._additional_data.kriging_data.df.loc['values', 'range'] / rescaling_factor,
        c_o               = geo_model._additional_data.kriging_data.df.loc['values', '$C_o$'] / rescaling_factor,
        uni_degree        = 1,
        number_dimensions = 3,
        kernel_function   = AvailableKernelFunctions.cubic,
        dual_contouring=False,
        compute_scalar_gradient=False,
        number_octree_levels=3
    )
    # @formatter:on

    print(options)

    solutions: Solutions = gempy_engine.compute_model(
        interpolation_input=interpolation_input,
        options=options,
        data_descriptor=input_data_descriptor
    )


def test_compute_model():
    geo_model = load_model()
    geo_model = map_sequential_pile(geo_model)
    interpolator = create_interpolator()
    geo_model.set_aesara_graph(interpolator)

    gp.compute_model(geo_model, compute_mesh=False)

    test_values = [45, 150, 2500]
    if False:
        np.save(input_path + '/test_integration_lith_block.npy', geo_model.solutions.lith_block[test_values])

    # Load model
    real_sol = np.load(input_path + '/test_integration_lith_block.npy')

    # We only compare the block because the absolute pot field I changed it
    np.testing.assert_array_almost_equal(np.round(geo_model.solutions.lith_block[test_values]), real_sol, decimal=0)

    gp.plot.plot_2d(geo_model, cell_number=25,
                    direction='y', show_data=True)

    gp.plot.plot_2d(geo_model, cell_number=25, series_n=1, N=15, show_scalar=True,
                    direction='y', show_data=True)
