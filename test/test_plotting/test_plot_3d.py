import gempy as gp
import pyvista as pv
import matplotlib.pyplot as plt


def test_plot_3d_data_default(one_fault_model_no_interp):
    gpv = gp.plot.plot_3d(one_fault_model_no_interp,
                          plotter_type='basic', off_screen=True, notebook=False)
    img = gpv.p.show(screenshot=True)
    plt.imshow(img[1])
    plt.show()


def test_plot_3d_geo_map(unconformity_model):
    gpv = gp.plot.plot_3d(unconformity_model,
                          plotter_type='basic', off_screen=True,
                          show_topography=True,
                          show_scalar=False,
                          show_lith=False,
                          kwargs_plot_structured_grid={'opacity': .5})
    img = gpv.p.show(screenshot=True)
    plt.imshow(img[1])
    plt.show()


def test_plot_3d_geo_map2(one_fault_model_topo_solution):
    gpv = gp.plot.plot_3d(one_fault_model_topo_solution,
                          plotter_type='basic', off_screen=True,
                          show_topography=True,
                          show_scalar=False,
                          show_lith=False,
                          kwargs_plot_structured_grid={'opacity': .5})
    img = gpv.p.show(screenshot=True)
    plt.imshow(img[1])
    plt.show()


def test_plot_3d_structure_topo(one_fault_model_topo_solution):
    one_fault_model_topo_solution._grid.regular_grid.set_topography_mask(one_fault_model_topo_solution._grid.topography)
    gpv = gp.plot.plot_3d(one_fault_model_topo_solution,
                          plotter_type='basic', off_screen=True,
                          show_topography=True,
                          show_scalar=False,
                          show_lith=True,
                          kwargs_plot_structured_grid={'opacity': .5})
    img = gpv.p.show(screenshot=True)
    plt.imshow(img[1])
    plt.show()


def test_plot_3d_structure_topo2(unconformity_model_topo):
    gpv = gp.plot.plot_3d(unconformity_model_topo,
                          plotter_type='basic', off_screen=True,
                          show_topography=True,
                          show_scalar=False,
                          show_lith=True,
                          kwargs_plot_structured_grid={'opacity': .5})
    img = gpv.p.show(screenshot=True)
    plt.imshow(img[1])
    plt.show()
