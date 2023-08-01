import numpy as np
import pytest
import gempy as gp
import gempy_viewer as gpv
from gempy.core.data.enumerators import ExampleModel
from gempy_engine.modules.octrees_topology.octrees_topology_interface import ValueType
from gempy_engine.plugins.plotting.helper_functions import plot_block_and_input_2d


def test_section_grids():
    geo_model: gp.GeoModel = gp.generate_example_model(
        example_model=ExampleModel.ANTICLINE,
        compute_model=False
    )

    geo_model.interpolation_options.number_octree_levels = 2

    gp.set_section_grid(
        grid=geo_model.grid,
        section_dict={'section_SW-NE': ([250, 250], [1750, 1750], [100, 100]),
                      'section_NW-SE': ([250, 1750], [1750, 250], [100, 100])}
    )

    gp.set_topography_from_random(
        grid=geo_model.grid,
        fractal_dimension=1.2,
        d_z=np.array([200, 1000]),
        topography_resolution=np.array([60, 60])
    )
    gp.compute_model(geo_model)

    gpv.plot_2d(
        model=geo_model,
        section_names=['section_SW-NE', 'section_NW-SE', 'topography'],
        direction=['x', 'y', 'y'], cell_number=['mid', 'mid', 'mid'],
        show_lith=[False, False, False, True, True, True],
        show_boundaries=[False, False, False, True, True, True],
        show_scalar=[False, False, False, False, True, True],
        series_n=[0, 0, 0, 0, 0, 0],
        show_topography=True,
        show_section_traces=True  # TODO: Test this one
    )

    gpv.plot_2d(geo_model, show_topography=True, show_boundaries=False, section_names=['topography'])
    gpv.plot_2d(geo_model, show_boundaries=False, section_names=['topography'])
    gpv.plot_3d(geo_model, show_data=True, show_boundaries=True, show_lith=False, image=True)
