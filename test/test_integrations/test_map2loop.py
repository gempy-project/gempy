import gempy as gp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Input files
root = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/data/input_data/turner_syncline/'

orientations_file = root + 'orientations_clean.csv'
contacts_file = root + 'contacts_clean.csv'
faults_contact = root + 'faults.csv'
faults_orientations = root + 'fault_orientations.csv'

fp = root + 'dtm_rp.tif'
series_file = root + 'all_sorts_clean.csv'

faults_rel_matrix = root + 'fault-fault-relationships.csv'
series_rel_matrix = root + 'group-fault-relationships.csv'

bbox = (500000, 7490000, 545000, 7520000)
model_base = -1500  # Original 3200
model_top = 800


def test_map2loop_model_input_data_preparation():
    contacts = []
    orientations = []

    contacts.append(
        pd.read_csv(
            contacts_file,
            sep=',',
            names=['X', 'Y', 'Z', 'formation'],
            header=1
        )
    )

    contacts.append(
        pd.read_csv(
            faults_contact,
            sep=',',
            names=['X', 'Y', 'Z', 'formation'],
            header=1
        )
    )

    orientations.append(
        pd.read_csv(
            orientations_file,
            sep=',',
            names=['X', 'Y', 'Z', 'azimuth', 'dip', 'polarity', 'formation'],
            header=1
        )
    )

    orientations.append(
        pd.read_csv(
            faults_orientations,
            sep=',',
            names=['X', 'Y', 'Z', 'azimuth', 'dip', 'polarity', 'formation'],
            header=1
        )
    )

    surface_points_ready = pd.concat(contacts, sort=True)
    surface_points_ready.reset_index(inplace=True, drop=False)
    orientation_ready = pd.concat(orientations, sort=True)
    orientation_ready.reset_index(inplace=True, drop=False)

    geo_model = gp.create_model('test_map2Loop')
    gp.init_data(
        geo_model,
        extent=[bbox[0], bbox[2], bbox[1], bbox[3], model_base, model_top],
        resolution=[50, 50, 50],
        orientations_df=orientation_ready,
        surface_points_df=surface_points_ready
    )

    # Stack Processing
    contents = np.genfromtxt(series_file,
                             delimiter=',', dtype='U100')[1:, 4:-1]

    map_series_to_surfaces = {}
    for pair in contents:
        map_series_to_surfaces.setdefault(pair[1], []).append(pair[0])

    gp.map_stack_to_surfaces(geo_model, map_series_to_surfaces,
                             remove_unused_series=False)

    # Get the unassigned series
    faults_pair = geo_model._surfaces.df.groupby('series').get_group('Default series')[[
        'surface']].values[:, 0]
    faults_pair_dict = dict(zip(faults_pair, faults_pair))
    gp.map_stack_to_surfaces(geo_model, faults_pair_dict)

    # Grabbing the order of faults and Formations

    # Sorting series

    ordered_features = np.append()
    geo_model.reorder_features([faults_pair])

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
    gp.plot_3d(geo_model, ve=None, show_topography=False, image=False,
               kwargs_plot_data={'arrow_size': 400}
               )
    print(geo_model.orientations)


def test_map2loop_model():
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
    geo_model._rescaling.toggle_axial_anisotropy()

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
               image=False,
               show_lith=False,
               )


def test_map2loop_model_faults():
    pass
