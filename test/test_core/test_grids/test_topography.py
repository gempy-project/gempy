import gempy as gp
import matplotlib.pyplot as plt
from gempy.core.grid_modules.topography import Topography
import pytest


@pytest.fixture(scope='module')
def artificial_grid(one_fault_model_no_interp):
    geo_model = one_fault_model_no_interp
    topo = Topography(geo_model._grid.regular_grid)
    topo.load_random_hills()
    print(topo.values, topo.values_2d)
    return topo


def test_plot_2d_topography(one_fault_model_no_interp, artificial_grid):
    geo_model = one_fault_model_no_interp
    geo_model._grid.topography = artificial_grid
    p2d = gp.plot_2d(geo_model, section_names=['topography'], show_topography=True,
                     kwargs_topography={'hillshade': False})
    plt.show()

    p2d = gp.plot_2d(geo_model, section_names=['topography'], show_topography=True,
                     kwargs_topography={'hillshade': True, 'fill_contour': False})
    plt.show()

    p2d = gp.plot_2d(geo_model, section_names=['topography'], show_topography=True,
                     kwargs_topography={'hillshade': True})
    plt.show()


def test_plot_3d_structure_topo2(unconformity_model_topo, artificial_grid):
    geo_model = unconformity_model_topo
    with pytest.raises(AssertionError):
        geo_model._grid.regular_grid.set_topography_mask(artificial_grid)

    geo_model._grid.regular_grid.set_topography_mask(geo_model._grid.topography)

    gpv = gp.plot.plot_3d(unconformity_model_topo,
                          plotter_type='basic', off_screen=False,
                          show_topography=True,
                          show_scalar=False,
                          show_lith=True,
                          kwargs_plot_structured_grid={'opacity': .5})
    img = gpv.p.show(screenshot=True)
    plt.imshow(img[1])
    plt.show()