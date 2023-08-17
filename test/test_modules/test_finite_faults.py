import numpy as np

import gempy as gp
import gempy_viewer as gpv
from gempy.core.data.enumerators import ExampleModel
from gempy_viewer.optional_dependencies import require_pyvista


center = np.array([500, 500, 500])
radius = np.array([100, 200, 200])
k = np.array([1, 1, 1]) * 2


def test_finite_fault_scalar_field():

    geo_model: gp.data.GeoModel = gp.generate_example_model(
        example_model=ExampleModel.ONE_FAULT,
        compute_model=False
    )
    
    regular_grid = geo_model.grid.regular_grid
    
    # TODO: Extract grid from the model
    scalar_funtion: callable = gp.implicit_functions.ellipsoid_3d_factory(  # * This paints the 3d regular grid
        center=center,
        radius=radius,
        max_slope=k  # * This controls the speed of the transition
    )

    transform = gp.data.transforms.Transform(
        position=np.array([0, 0, 0]),
        rotation=np.array([0, 0, 30]),
        scale=np.ones(3)
    )

    transformed_points = transform.apply_inverse_with_pivot(
        points=regular_grid.values,
        pivot=center
    )
    scalar_block = scalar_funtion(transformed_points)
    
    
    # TODO: Try to do this afterwards
    # scalar_fault = scalar_funtion(regular_grid.values)
    
    if plot_pyvista:= True:
        _plot_scalar_field(regular_grid, scalar_block)
        
        
def test_finite_fault_scalar_field_on_fault():
    geo_model: gp.data.GeoModel = gp.generate_example_model(
        example_model=ExampleModel.ONE_FAULT,
        compute_model=True
    )

    regular_grid = geo_model.grid.regular_grid

    # TODO: Extract grid from the model
    scalar_funtion: callable = gp.implicit_functions.ellipsoid_3d_factory(  # * This paints the 3d regular grid
        center=center,
        radius=radius,
        max_slope=k  # * This controls the speed of the transition
    )

    transform = gp.data.transforms.Transform(
        position=np.array([0, 0, 0]),
        rotation=np.array([0, 0, 30]),
        scale=np.ones(3)
    )

    transformed_points = transform.apply_inverse_with_pivot(
        points=regular_grid.values,
        pivot=center
    )
    scalar_block = scalar_funtion(transformed_points)

    # TODO: Try to do this afterwards
    # scalar_fault = scalar_funtion(regular_grid.values)

    if plot_pyvista := True:
        plot3d = gpv.plot_3d(
            geo_model,
            show=False
        )
        _plot_scalar_field(regular_grid, scalar_block, plot3d.p)
        


def _plot_scalar_field(regular_grid, scalar_block, plotter=None):
    pv = require_pyvista()
    p = plotter or pv.Plotter()
    regular_grid_values = regular_grid.values_vtk_format
    grid_3d = regular_grid_values.reshape(*(regular_grid.resolution + 1), 3).T
    regular_grid_mesh = pv.StructuredGrid(*grid_3d)
    regular_grid_mesh["lith"] = scalar_block
    if True:
        area_of_effect = regular_grid_mesh.threshold([.000000001, 1.1])
        p.add_mesh(area_of_effect, show_edges=True, opacity=.4)

        area_of_effect_2 = regular_grid_mesh.threshold([.2, 1.1])
        p.add_mesh(area_of_effect_2, show_edges=True, opacity=.8)
    p.add_mesh(regular_grid_mesh, show_edges=False, opacity=.2)
    p.show_bounds(bounds=regular_grid.extent)
    # * Add the fault
    if False:
        dual_mesh = pv.PolyData(fault_mesh.vertices, np.insert(fault_mesh.edges, 0, 3, axis=1).ravel())
        dual_mesh["bar"] = scalar_fault
        p.add_mesh(dual_mesh, opacity=1, silhouette=True, show_edges=True)
    p.show()
    
