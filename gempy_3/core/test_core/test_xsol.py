import pytest
from gempy.core.data import Surfaces

from gempy.core.data_modules.stack import Stack

import gempy as gp
import numpy as np

from gempy.core.solution import Solution
from gempy.core.xsolution import XSolution


@pytest.fixture(scope='module')
def a_grid():
    # Or we can init one of the default grids since the beginning by passing
    # the correspondant attributes
    grid = gp.Grid(extent=[0, 2000, 0, 2000, -2000, 0],
                   resolution=[50, 50, 50])
    grid.set_active('regular')

    grid.create_custom_grid(np.arange(12).reshape(-1, 3))
    grid.set_active('custom')

    grid.create_topography()
    grid.set_active('topography')

    section_dict = {'section_SW-NE': ([250, 250], [1750, 1750], [100, 100]),
                    'section_NW-SE': ([250, 1750], [1750, 250], [100, 100])}
    grid.create_section_grid(section_dict)
    grid.set_active('sections')

    return grid


@pytest.fixture(scope='module')
def stack_eg():
    series = Stack()
    series.set_series_index(['foo', 'foo2', 'foo5', 'foo7'])
    series.add_series('foo3')
    series.delete_series('foo2')
    series.rename_series({'foo': 'boo'})
    series.reorder_series(['foo3', 'boo', 'foo7', 'foo5'])

    series.set_is_fault(['boo'])

    fr = np.zeros((4, 4))
    fr[2, 2] = True
    series.set_fault_relation(fr)

    series.add_series('foo20')

    # Mock
    series.df['isActive'] = True
    return series


@pytest.fixture(scope='module')
def surface_eg(stack_eg):
    surfaces = Surfaces(stack_eg)
    surfaces.set_surfaces_names(['foo', 'foo2', 'foo5', 'fee'])
    surfaces.add_surfaces_values([[2, 2, 2, 6], [2, 2, 1, 8]],
                                 ['val_foo', 'val2_foo'])
    return surfaces


@pytest.fixture(scope='module')
def sol_values(a_grid):
    rg_s = a_grid.values.shape[0]
    n_input = 100
    len_x = rg_s + n_input

    n_features = 5
    n_properties = 3
    # Generate random solution
    values = list()
    values_matrix = np.random.randint(0, 10, (n_properties, len_x))
    block_matrix = np.random.randint(
        0, 10, (n_features, n_properties, len_x)
    )

    fault_block = np.random.randint(40, 50, (n_features, len_x))
    weights = None
    scalar_field = np.random.randint(20, 30, (n_features, len_x))
    unknows = None
    mask_matrix = None
    fault_mask = None

    values.append(values_matrix)
    values.append(block_matrix)
    for i in [fault_block, weights, scalar_field, unknows, mask_matrix, fault_mask]:
        values.append(i)
    return values


def test_xsol(sol_values, a_grid, stack_eg, surface_eg):
    sol = XSolution(a_grid, stack=stack_eg, surfaces=surface_eg)
    sol.set_values(sol_values)
    print('\n regular', sol.s_regular_grid)
    print('\n custom', sol.s_custom_grid)
    print('\n topo', sol.s_topography)
    print('\n sect', sol.s_sections['section_SW-NE'])
    print('\n sect', sol.s_sections['section_NW-SE'])

    sol.set_values(sol_values)
    print('\n custom2', sol.s_custom_grid)


def test_xsol_unstructured(sol_values, a_grid, stack_eg, surface_eg):
    sol = XSolution(a_grid, stack=stack_eg, surfaces=surface_eg)
    sol.set_values(sol_values)
    print('\n custom', sol.s_custom_grid)


def test_xsol_structured(sol_values, a_grid, stack_eg, surface_eg):
    sol = XSolution(a_grid, stack=stack_eg, surfaces=surface_eg)
    sol.set_values(sol_values)
    print('\n custom', sol.s_regular_grid)


def test_xsol_to_disk(sol_values, a_grid, stack_eg, surface_eg, tmpdir):
    sol = XSolution(a_grid, stack=stack_eg, surfaces=surface_eg)
    sol.set_values(sol_values)
    print('\n custom', sol.s_regular_grid)
    sol.to_netcdf(tmpdir, 'bar')


@pytest.mark.skip('Test for gempy_lite')
def test_property(sol_values, a_grid, stack_eg, surface_eg):
    sol = XSolution(a_grid, stack=stack_eg, surfaces=surface_eg)
    sol.set_values(sol_values)
    print('scalar', sol.scalar_field_matrix)
    print('lith', sol.lith_block)
    print('values', sol.values_matrix)

    print('scalar_asp', sol.scalar_field_at_surface_points)


@pytest.mark.skip('Test for gempy_lite')
def test_scalar_field_matrix_property_full(model_horizontal_two_layers):
    sol_vals = gp.compute_model(model_horizontal_two_layers, set_solutions=False)
    sol = XSolution(
        model_horizontal_two_layers._grid,
        stack=model_horizontal_two_layers._stack,
        surfaces=model_horizontal_two_layers._surfaces)
    sol.set_values(sol_vals)
    print('scalar', sol.scalar_field_matrix)

    old_sol = Solution(model_horizontal_two_layers._grid,
        series=model_horizontal_two_layers._stack,
        surfaces=model_horizontal_two_layers._surfaces)
    old_sol.set_solution_to_regular_grid(sol_vals, False, None)
    print(old_sol.scalar_field_matrix)

    np.testing.assert_array_almost_equal(sol.scalar_field_matrix,
                                         old_sol.scalar_field_matrix)


def test_xsol_full(model_horizontal_two_layers):
    model_horizontal_two_layers.update_to_interpolator()
    vals = gp.compute_model(model_horizontal_two_layers, set_solutions=False)
    sol = XSolution(
        model_horizontal_two_layers._grid,
        stack=model_horizontal_two_layers._stack,
        surfaces=model_horizontal_two_layers._surfaces)
    sol.set_values(vals)
    print('\n regular', sol.s_regular_grid)
    print('\n custom', sol.s_custom_grid)
    print('\n topo', sol.s_topography)


def test_xsol_inherit(model_horizontal_two_layers):
    vals = gp.compute_model(model_horizontal_two_layers, set_solutions=True)


def test_set_meshes(model_horizontal_two_layers):
    import subsurface
    m = model_horizontal_two_layers
    vals = gp.compute_model(m, set_solutions=True)
    unstruct = m.solutions.set_meshes(m.surfaces)
    ts = subsurface.TriSurf(unstruct)
    s  = subsurface.visualization.to_pyvista_mesh(ts)
    subsurface.visualization.pv_plot([s], image_2d=True)

    print(unstruct)


def test_save_solutions(model_horizontal_two_layers, tmpdir):
    gp.save_model(model_horizontal_two_layers, solution=True, compress=False,
                  path=tmpdir)
