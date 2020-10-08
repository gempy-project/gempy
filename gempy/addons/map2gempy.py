import gempy as gp
import pandas as pd
import numpy as np


def loop2gempy(
        contacts_file: str,
        orientations_file: str,
        bbox: tuple,
        groups_file: str,
        model_base: float,
        model_top: float, dtm_reproj_file: str = None,
        faults_contact: str = None,
        faults_orientations: str = None,
        model_name: str = None,
        compute: bool = True,
        vtk: bool = False,
        vtk_path: str = None,
        image_2d: bool = False
):

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

    if faults_contact is not None:
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

    if faults_orientations is not None:
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

    if model_name is None:
        model_name = 'loop2gempy'

    geo_model = gp.create_model(model_name)
    gp.init_data(
        geo_model,
        extent=[bbox[0], bbox[2], bbox[1], bbox[3], model_base, model_top],
        resolution=[50, 50, 50],
        orientations_df=orientation_ready,
        surface_points_df=surface_points_ready
    )

    # Load Topology
    if dtm_reproj_file is not None:
        geo_model.set_topography(source='gdal', filepath=dtm_reproj_file)

    # Stack Processing
    contents = np.genfromtxt(groups_file, delimiter=',', dtype='U100')[1:, 4:-1]

    map_series_to_surfaces = {}
    for pair in contents:
        map_series_to_surfaces.setdefault(pair[1], []).append(pair[0])

    gp.map_stack_to_surfaces(geo_model, map_series_to_surfaces,
                             remove_unused_series=False)

    order_formations = geo_model.stack.df.index.drop('Default series')

    # Get the unassigned series as faults
    if faults_contact is not None and faults_orientations is not None:
        faults_pair = geo_model._surfaces.df.groupby('series').get_group('Default series')[[
            'surface']].values[:, 0]
        faults_pair_dict = dict(zip(faults_pair, faults_pair))
        gp.map_stack_to_surfaces(geo_model, faults_pair_dict, remove_unused_series=True)
        # Grabbing the order of faults and Formations
        ordered_features = np.append(faults_pair, order_formations)
        # Sorting series
        geo_model.reorder_features(ordered_features)
        geo_model.set_is_fault(faults_pair)

    geo_model.add_surfaces('basement')

    # Check if there is features without data and delete it
    try:
        f_ = geo_model.surfaces.df.groupby('isActive').get_group(False)
        features_without_data = f_['series']
        geo_model.delete_features(features_without_data, remove_surfaces=True, remove_data=True)
    except KeyError:
        pass

    if faults_contact is not None and faults_orientations is not None:
        get_fault_names = geo_model.stack.df.groupby(['isActive', 'isFault']).get_group(
            (True, True)).index
        geo_model._surfaces.colors.make_faults_black(get_fault_names)

    if compute is True:
        gp.set_interpolator(geo_model)

        # Increasing nugget effect
        geo_model.modify_surface_points(
            geo_model.surface_points.df.index,
            smooth=0.1
        )
        new_range = geo_model.get_additional_data().loc[('Kriging', 'range'), 'values'] * 0.3
        geo_model.modify_kriging_parameters('range', new_range)

        gp.compute_model(geo_model)

    if vtk is True:
        gp.plot_3d(geo_model, ve=10, show_topography=True,
                   image=image_2d,
                   show_lith=False,
                   )

    if vtk_path is not None:
        gp._plot.export_to_vtk(geo_model, path=vtk_path, name=model_name + '.vtk',
                               voxels=False, block=None, surfaces=True)

    return geo_model