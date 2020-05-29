import pytest
import gempy as gp
import matplotlib.pyplot as plt
import numpy as np


def test_sort_surfaces_by_solution(one_fault_model_topo_solution):
    geo_model = one_fault_model_topo_solution
    section_dict = {'section_SW-NE': ([250, 250], [1750, 1750], [100, 100]),
                    'section_NW-SE': ([250, 1750], [1750, 250], [100, 100])}
    geo_model.set_section_grid(section_dict)

    geo_model.set_active_grid('sections', reset=True)

    s1 = geo_model.solutions.scalar_field_at_surface_points
    geo_model.update_additional_data()
    geo_model.update_to_interpolator()
    gp.compute_model(geo_model, sort_surfaces=True)
    gp.plot_2d(geo_model, section_names=['section_NW-SE'],
               show_topography=True)
    plt.show()
    s2 = geo_model.solutions.scalar_field_at_surface_points

    gp.compute_model(geo_model, sort_surfaces=True)
    gp.plot_2d(geo_model, section_names=['section_NW-SE'],
               show_topography=True)
    plt.show()
    s3 = geo_model.solutions.scalar_field_at_surface_points
    np.testing.assert_array_equal(s2, s3)

    gp.compute_model(geo_model, sort_surfaces=True)
    gp.plot_2d(geo_model, section_names=['section_NW-SE'],
               show_topography=True)
    plt.show()

    return geo_model
