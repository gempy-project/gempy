import gempy as gp
import matplotlib.pyplot as plt
import numpy as np
import pytest


def test_plot_2d_data_default(one_fault_model_no_interp):
    gp.plot.plot_2d(one_fault_model_no_interp)
    plt.show()


def test_plot_2d_data_default_all_none(one_fault_model_no_interp):
    gp.plot.plot_2d(one_fault_model_no_interp, show_data=True, show_results=False)
    plt.show()


def test_basic(one_fault_model_no_interp):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(np.arange(25).reshape(5,5))
    fig.show()


def test_plot_2d_test_labels(one_fault_model_no_interp):
    geo_model = one_fault_model_no_interp
    section_dict = {'section_SW-NE': ([250, 250], [1750, 1750], [100, 100]),
                    'section_NW-SE': ([250, 1750], [1750, 250], [100, 100])}
    geo_model.set_section_grid(section_dict)
    geo_model.set_topography(fd=1.2, d_z=np.array([600, 2000]), resolution=np.array([60, 60]))

    gp.plot_2d(geo_model,
               section_names=['section_NW-SE', 'section_NW-SE', 'topography'],
               direction=['x'], cell_number=['mid'],
               show_topography=True,
               )
    plt.show()

    gp.plot_2d(geo_model,
               section_names=['section_NW-SE', 'section_NW-SE', 'topography'],
               direction=['x'], cell_number=['mid'],
               show_topography=True,
               projection_distance=100)
    plt.show()

    gp.plot_2d(geo_model,
               section_names=['section_NW-SE', 'section_NW-SE', 'topography'],
               direction=['x'], cell_number=['mid'],
               show_topography=True,
               projection_distance=1000)
    plt.show()

    gp.plot.plot_section_traces(geo_model)
    plt.show()


@pytest.fixture(scope='module')
def section_model(one_fault_model_topo_solution):
    geo_model = one_fault_model_topo_solution
    section_dict = {'section_SW-NE': ([250, 250], [1750, 1750], [100, 100]),
                    'section_NW-SE': ([250, 1750], [1750, 250], [100, 100])}
    geo_model.set_section_grid(section_dict)

    geo_model.set_active_grid('sections', reset=False)

    one_fault_model_topo_solution.update_additional_data()
    one_fault_model_topo_solution.update_to_interpolator()
    gp.compute_model(geo_model, sort_surfaces=False)
    return geo_model


def test_topo_sections_iterp2(section_model):
    # Test 1 single
    gp.plot_2d(section_model, section_names=['section_NW-SE'],
               show_topography=True)
    plt.show()

    # Test 2 plots
    gp.plot_2d(section_model, section_names=['section_NW-SE', 'section_NW-SE'],
               show_topography=True)
    plt.show()

    # Test 3 plots
    gp.plot_2d(section_model, section_names=['section_NW-SE', 'section_NW-SE', 'topography'],
               show_topography=True)
    plt.show()

    # Test 4
    gp.plot_2d(section_model, section_names=['section_NW-SE', 'section_NW-SE', 'topography'],
               direction=['x'], cell_number=['mid'],
               show_topography=True)
    plt.show()

    gp.plot.plot_section_traces(section_model)
    plt.show()


def test_show_results(section_model):
    p2d = gp.plot_2d(section_model,
                     show_data=False,
                     show_topography=False,
                     show_results=True,
                     show=True)


def test_ve(section_model):
    # Test ve
    p2d = gp.plot_2d(section_model, section_names=['section_NW-SE', 'section_NW-SE'],
                     show_topography=True, ve=3)

    plt.show()


def test_topo_resize(one_fault_model_topo_solution):
    geo_model = one_fault_model_topo_solution
    geo_model.set_topography(fd=1.2, d_z=np.array([600, 2000]), resolution=np.array([60, 60]))
    sol = gp.compute_model(geo_model, compute_mesh=False)
    gp.plot_2d(geo_model, section_names=['topography'])
    plt.show()
