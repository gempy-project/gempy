import gempy as gp
import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
import os

# Input files
from gempy.addons.map2gempy import loop2gempy

root = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/data/input_data/turner_syncline/'
root2 = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/data/input_data/australia/'
path = os.path.dirname(__file__) + '/../input_data/'

orientations_file = root + 'orientations_clean.csv'
orientations_file2 = root2 + 'orientations_clean.csv'

contacts_file = root + 'contacts_clean.csv'
contacts_file2 = root2 + 'contacts_clean.csv'
faults_contact = root + 'faults.csv'
faults_contact2 = root2 + 'faults.csv'
faults_orientations = root + 'fault_orientations.csv'
faults_orientations2 = root2 + 'fault_orientations.csv'
series_file = root + 'all_sorts_clean.csv'
series_file2 = root2 + 'all_sorts_clean.csv'

faults_rel_matrix = root + 'fault-fault-relationships.csv'
faults_rel_matrix2 = root2 + 'fault-fault-relationships.csv'
series_rel_matrix = root + 'group-fault-relationships.csv'
series_rel_matrix2 = root2 + 'group-fault-relationships.csv'

ff = root2 + 'fault-fault-relationships.csv'
fg = root2 + 'group-fault-relationships.csv'

fp = path + 'dtm_rp.tif'
fp2 = path + 'dtm_rp2.tif'

bbox = (500000, 7490000, 545000, 7520000)
model_base = -0  # Original 3200
model_top = 800
extent_g = [515687.3100586407, 562666.8601065436,
            7473446.765934078, 7521273.574077863,
            -3200, 1200.0]

extent = [515687.3100586407, 7473446.765934078,
          562666.8601065436, 7521273.574077863,
          -3200, 1200.0]


@pytest.mark.skipif("TRAVIS" in os.environ and os.environ["TRAVIS"] == "true",
                    reason="Skipping this test on Travis CI. For some reason there is a linalg "
                           "error.")
def test_loop2gempy():
    topo = fp
    topo = None

    loop2gempy(contacts_file, orientations_file, bbox, series_file, model_base,
               model_top, topo, faults_contact, faults_orientations, 'testing_map',
               vtk=True, vtk_path=None, image_2d=True)


def test_map2loop2relmatrix():
    ff_ = pd.read_csv(ff).set_index('fault_id')
    fg_ = pd.read_csv(fg).set_index('group')
    p = pd.concat((ff_, fg_.T), axis=1)
    print(p)


@pytest.mark.skipif("TRAVIS" in os.environ and os.environ["TRAVIS"] == "true",
                    reason="Skipping this test on Travis CI beacuse travis.")
def test_loop2gempy2():
    topo = fp2
    # topo = None

    loop2gempy(contacts_file2, orientations_file2, extent[:4], series_file2,
               extent[4],
               extent[5],
               dtm_reproj_file=topo,
               faults_contact=faults_contact2,
               faults_orientations=faults_orientations2,
               faults_faults_rel=ff,
               faults_groups_rel=fg,
               model_name='testing_map',
               compute=True,
               vtk=True, vtk_path=None, image_2d=True)


@pytest.mark.skipif("TRAVIS" in os.environ and os.environ["TRAVIS"] == "true",
                    reason="Skipping this test on Travis CI beacuse travis.")
def test_map2loop_model_import_data():
    geo_model = gp.create_model('test_map2Loop')
    gp.init_data(
        geo_model,
        extent=[bbox[0], bbox[2], bbox[1], bbox[3], model_base, model_top],
        resolution=[50, 50, 50],
        path_o=orientations_file,
        path_i=contacts_file
    )

    # Load Topology
    geo_model.set_topography(source='gdal', filepath=fp)

    gp.plot_2d(geo_model, ve=10, show_topography=True)
    plt.show()

    # Plot in 3D
    gp.plot_3d(geo_model, ve=None, show_topography=False, image=True,
               kwargs_plot_data={'arrow_size': 400}
               )
    print(geo_model.orientations)


@pytest.mark.skipif("TRAVIS" in os.environ and os.environ["TRAVIS"] == "true",
                    reason="Skipping this test on Travis CI beacuse travis.")
def test_map2loop_model_no_faults():
    # Location box
    bbox = (500000, 7490000, 545000, 7520000)
    model_base = -3200  # Original 3200
    model_top = 800

    # Input files
    geo_model = gp.create_model('test_map2Loop')
    gp.init_data(
        geo_model,
        extent=[bbox[0], bbox[2], bbox[1], bbox[3], model_base, model_top],
        resolution=[50, 50, 80],
        path_o=orientations_file,
        path_i=contacts_file
    )

    gp.set_interpolator(geo_model)

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

    gp.plot_2d(geo_model, ve=10, show_topography=False)
    plt.show()

    # Plot in 3D
    gp.plot_3d(geo_model, ve=10, show_topography=False, image=True)

    # Stack Processing
    contents = np.genfromtxt(series_file,
                             delimiter=',', dtype='U100')[1:, 4:-1]

    map_series_to_surfaces = {}
    for pair in contents:
        map_series_to_surfaces.setdefault(pair[1], []).append(pair[0])

    gp.map_stack_to_surfaces(geo_model, map_series_to_surfaces,
                             remove_unused_series=False)

    # Adding axial rescale
    # geo_model._rescaling.toggle_axial_anisotropy()

    # Increasing nugget effect
    geo_model.modify_surface_points(
        geo_model.surface_points.df.index,
        smooth=0.001
    )

    geo_model.modify_kriging_parameters('drift equations', [9, 9, 9, 9, 9])

    gp.compute_model(geo_model)

    gp.plot_2d(geo_model,
               section_names=['topography'],
               show_topography=True,
               )
    plt.show()

    gp.plot_3d(geo_model, ve=10, show_topography=True,
               image=True,
               show_lith=False,
               )
