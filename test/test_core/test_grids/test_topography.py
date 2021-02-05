import os
import sys, os
sys.path.append("../../..")

os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu"
import warnings

try:
    import faulthandler
    faulthandler.enable()
except Exception as e:  # pragma: no cover
    warnings.warn('Unable to enable faulthandler:\n%s' % str(e))

import gempy as gp
import matplotlib.pyplot as plt
from gempy.core.grid_modules.topography import Topography
import pytest

import numpy as np
data_path = os.path.dirname(__file__)+'/../../input_data'


@pytest.fixture(scope='module')
def artificial_grid(one_fault_model_no_interp):
    geo_model = one_fault_model_no_interp
    topo = Topography(geo_model._grid.regular_grid)
    topo.load_random_hills()
    print(topo.values, topo.values_2d)
    return topo


@pytest.mark.skipif("TRAVIS" in os.environ and os.environ["TRAVIS"] == "true",
                    reason="Skipping this test on Travis CI because gdal.")
def test_real_grid_ales():
    resolution = [30, 20, 50]
    extent = np.array([729550.0, 751500.0, 1913500.0, 1923650.0, -50, 800.0])
    path_interf = data_path + "/2018_interf.csv"
    path_orient = data_path + "/2018_orient_clust_n_init5_0.csv"
    path_dem = data_path + "/_cropped_DEM_coarse.tif"

    geo_model = gp.create_model('Alesmodel')
    gp.init_data(geo_model, extent=extent, resolution=resolution,
                 path_i=path_interf,
                 path_o=path_orient)

    geo_model.set_topography(source='gdal', filepath=path_dem)

    p2d = gp.plot_2d(geo_model, section_names=['topography'], show_topography=True,
                     kwargs_topography={'hillshade': False, 'fill_contour': False})
    plt.show()

    p2d = gp.plot_2d(geo_model, section_names=['topography'], show_topography=True,
                     kwargs_topography={'hillshade': False, 'fill_contour': True})
    plt.show()

    p2d = gp.plot_2d(geo_model, section_names=['topography'], show_topography=True,
                     kwargs_topography={'hillshade': True, 'fill_contour': True})
    plt.show()

    if False:
        gp.map_stack_to_surfaces(geo_model, {'fault_left': ('fault_left'),
                                              'fault_right': ('fault_right'),
                                              'fault_lr': ('fault_lr'),
                                              'Trias_Series': ('TRIAS', 'LIAS'),
                                              'Carbon_Series': ('CARBO'),
                                              'Basement_Series': ('basement')}, remove_unused_series=True)

        geo_model.set_is_fault(['fault_right', 'fault_left', 'fault_lr'], change_color=True)
        gp.set_interpolator(geo_model,
                                  output=['geology'], compile_theano=True,
                                  theano_optimizer='fast_run', dtype='float64',
                                  verbose=[])

        gp.compute_model(geo_model, compute_mesh=True)

        geo_model._grid.regular_grid.set_topography_mask(geo_model._grid.topography)

    gpv = gp.plot.plot_3d(geo_model,
                          plotter_type='basic', off_screen=True,
                          show_topography=True,
                          show_scalar=False,
                          show_lith=True,
                          show_surfaces=False,
                          kwargs_plot_structured_grid={'opacity': 1,
                                                       'show_edges': False},
                          ve=10,
                          image=True,
                          kwargs_plot_topography={'scalars': 'topography'})
    #
    # gpv.p.set_scale(zscale=10)
    #
    # img = gpv.p.show(screenshot=True)
    # plt.imshow(img[1])
    # plt.show()


def test_plot_2d_topography(one_fault_model_no_interp, artificial_grid):
    geo_model = one_fault_model_no_interp
    #geo_model._grid.topography = artificial_grid
    geo_model.set_topography()
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

    # geo_model._grid.regular_grid.set_topography_mask(geo_model._grid.topography)

    p2d = gp.plot_2d(geo_model, section_names=['topography'], show_topography=True,
                     show_lith=False,
                     kwargs_topography={'hillshade': True})
    plt.show()

    gpv = gp.plot.plot_3d(unconformity_model_topo,
                          plotter_type='basic', off_screen=True,
                          show_topography=True,
                          show_scalar=False,
                          show_lith=True,
                          show_surfaces=True,
                          kwargs_plot_structured_grid={'opacity': .5,
                                                       'show_edges': True},
                          image=True,
                          kwargs_plot_topography={'scalars': 'topography'})

    # img = gpv.p.show(screenshot=True)
    # plt.imshow(img[1])
    # plt.show()