import numpy as np
import pytest

import gempy as gp
import matplotlib.pyplot as plt
import os

# Input files
root = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/data/input_data/turner_syncline/'
path = os.path.dirname(__file__) + '/../input_data/'
orientations_file = root + 'orientations_clean.csv'
contacts_file = root + 'contacts_clean.csv'
fp = path + 'dtm_rp.tif'
series_file = root + 'all_sorts_clean.csv'

bbox = (500000, 7490000, 545000, 7520000)
model_base = -1500  # Original 3200
model_top = 800

gdal = pytest.importorskip("gdal")


@pytest.fixture(scope='module')
def model():
    geo_model = gp.create_model('test_map2Loop')
    gp.init_data(
        geo_model,
        extent=[bbox[0], bbox[2], bbox[1], bbox[3], model_base, model_top],
        resolution=[50, 50, 80],
        path_o=orientations_file,
        path_i=contacts_file
    )

    # Load Topology

    geo_model.set_topography(source='gdal', filepath=fp)

    # Stack Processing
    contents = np.genfromtxt(series_file,
                             delimiter=',', dtype='U100')[1:, 4:-1]

    map_series_to_surfaces = {}
    for pair in contents:
        map_series_to_surfaces.setdefault(pair[1], []).append(pair[0])

    gp.map_stack_to_surfaces(geo_model, map_series_to_surfaces,
                             remove_unused_series=False)

    gp.plot_3d(geo_model, ve=None, show_topography=False
               , image=True, show_lith=False,
               kwargs_plot_data={'arrow_size': 300})

    return geo_model


def test_axial_anisotropy_type_data(model):
    geo_model = model

    geo_model._rescaling.toggle_axial_anisotropy()
    # gp.compute_model(geo_model, compute_mesh_options={'mask_topography': False})

    geo_model.surface_points.df[['X', 'Y', 'Z']] = geo_model.surface_points.df[['X_c', 'Y_c',
                                                                                'Z_c']]
    geo_model.orientations.df[['X', 'Y', 'Z']] = geo_model.orientations.df[['X_c', 'Y_c',
                                                                            'Z_c']]

    # This is a hack
    geo_model._grid.topography.extent = geo_model._grid.extent_c

    geo_model.set_regular_grid(geo_model._grid.extent_c, [50, 50, 50])

    gp.plot_3d(geo_model, ve=None, show_topography=False
               , image=True, show_lith=False,
               kwargs_plot_data={'arrow_size': 10})


def test_axial_anisotropy_type_extent(model):
    geo_model = model

    geo_model._rescaling.toggle_axial_anisotropy(type='extent')
    # gp.compute_model(geo_model, compute_mesh_options={'mask_topography': False})

    geo_model.surface_points.df[['X', 'Y', 'Z']] = geo_model.surface_points.df[['X_c', 'Y_c',
                                                                                'Z_c']]
    geo_model.orientations.df[['X', 'Y', 'Z']] = geo_model.orientations.df[['X_c', 'Y_c',
                                                                            'Z_c']]

    # This is a hack
    geo_model._grid.topography.extent = geo_model._grid.extent_c

    geo_model.set_regular_grid(geo_model._grid.extent_c, [50, 50, 50])

    gp.plot_3d(geo_model, ve=None, show_topography=False
               , image=True, show_lith=False,
               kwargs_plot_data={'arrow_size': 10})


def test_axial_anisotropy(model):
    # Location box

    geo_model = model
    geo_model._rescaling.toggle_axial_anisotropy()
    # gp.compute_model(geo_model, compute_mesh_options={'mask_topography': False})

    geo_model.surface_points.df[['X', 'Y', 'Z']] = geo_model.surface_points.df[['X_c', 'Y_c',
                                                                                'Z_c']]
    geo_model.orientations.df[['X', 'Y', 'Z']] = geo_model.orientations.df[['X_c', 'Y_c',
                                                                            'Z_c']]

    # This is a hack
    geo_model._grid.topography.extent = geo_model._grid.extent_c

    geo_model.set_regular_grid(geo_model._grid.extent_c, [50, 50, 50])
    geo_model.modify_kriging_parameters('range', 0.1)
    geo_model.modify_kriging_parameters('drift equations', [9, 9, 9, 9, 9])

    geo_model.modify_surface_points(
        geo_model.surface_points.df.index,
        smooth=0.001
    )

    gp.set_interpolator(geo_model, theano_optimizer='fast_run', dtype='float64')
    gp.compute_model(geo_model, compute_mesh_options={'mask_topography': False,
                                                      'masked_marching_cubes': False})

    gp.plot_2d(geo_model,
               section_names=['topography'],
               show_topography=True,
               )
    plt.show()

    gp.plot_3d(geo_model, ve=None, show_topography=False, image=True, show_lith=False,
               kwargs_plot_data={'arrow_size': 10}
               )
