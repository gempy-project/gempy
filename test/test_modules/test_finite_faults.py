import numpy as np

import gempy as gp
from gempy.core.data.enumerators import ExampleModel
from gempy_viewer.optional_dependencies import require_pyvista


def test_finite_fault_scalar_field():

    geo_model: gp.data.GeoModel = gp.generate_example_model(
        example_model=ExampleModel.ANTICLINE,
        compute_model=False
    )
    
    regular_grid = geo_model.grid.regular_grid
    
    # TODO: Extract grid from the model
    center = np.array([0, 0, 0])
    radius = np.array([1000, 1000, 1000])
    
    scalar_funtion: callable = gp.implicit_functions.ellipsoid_3d_factory(  # * This paints the 3d regular grid
        center=center,
        radius=radius,
        max_slope=1000  # * This controls the speed of the transition
    )
    
    
    scalar_block = scalar_funtion(regular_grid.values)
    
    # TODO: Try to do this afterwards
    # scalar_fault = scalar_funtion(regular_grid.values)
    
    if plot_pyvista:= True:
        pv = require_pyvista()
        p = pv.Plotter()
        regular_grid_values = regular_grid.values_vtk_format

        grid_3d = regular_grid_values.reshape(*(regular_grid.resolution + 1), 3).T
        regular_grid_mesh = pv.StructuredGrid(*grid_3d)
        
        regular_grid_mesh["lith"] = scalar_block
        p.add_mesh(regular_grid_mesh, show_edges=False, opacity=.5)

        # * Add the fault
        if False:
            dual_mesh = pv.PolyData(fault_mesh.vertices, np.insert(fault_mesh.edges, 0, 3, axis=1).ravel())
            dual_mesh["bar"] = scalar_fault
            p.add_mesh(dual_mesh, opacity=1, silhouette=True, show_edges=True)

        p.show()

