from typing import Iterable
import gempy as gp
import pandas as pd
import numpy as np


def loop2gempy(
        contacts_file: str,
        orientations_file: str,
        bbox: Iterable,
        groups_file: str,
        model_base: float,
        model_top: float, dtm_reproj_file: str = None,
        faults_contact: str = None,
        faults_orientations: str = None,
        faults_faults_rel: str = None,
        faults_groups_rel: str = None,
        faults_rel_matrix = None,
        model_name: str = None,
        compute: bool = True,
        vtk: bool = False,
        vtk_path: str = None,
        image_2d: bool = False,
        plot_3d_kwargs=None
):
    """ Calculate the model using gempy as backend.

        At the moment there is not support for finite faults since gempy does not
         accept passing the ellipsoid parameters directly.

        :param contacts_file (str): path of contacts file
        :param orientations_file: path of orientations file
        :param bbox: model bounding box
        :param groups_file: path of groups file
        :param model_base: z value of base of model
        :param model_top: z value of top of model
        :param dtm_reproj_file: path of dtm file
        :param faults_contact: path of contacts file with fault data
        :param faults_orientations: path of orientations file with fault data
        :param faults_rel_matrix: bool matrix describing the interaction between groups. Rows offset columns
        :param faults_groups_rel: bool matrix describing the interaction between faults and features
        :param faults_faults_rel: bool matrix describing the interaction between faults and faults
        :param model_name: name of the model
        :param compute (bool): Default True. Whether or not compute the model
        :param vtk (bool): Default False. Whether or not visualize the model
        :param vtk_path (str): Default None. Path of vtk output directory
        :param plot_3d_kwargs (dict): kwargs for `gempy.plot_3d`
        :return: gempy.Project
    """
    if plot_3d_kwargs is None:
        plot_3d_kwargs = {}

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

    if faults_faults_rel is not None and faults_groups_rel is not None:
        ff_ = pd.read_csv(faults_faults_rel).set_index('fault_id')
        fg_ = pd.read_csv(faults_groups_rel).set_index('group')
        p_ = pd.concat((ff_, fg_), axis=0, sort=True)
        faults_rel_matrix = pd.concat((p_, fg_.T), axis=1, sort=True).fillna(0).values

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
        if type(dtm_reproj_file) is str:
            source = 'gdal'
            topo_kwarg = {'filepath': dtm_reproj_file}
        elif type(dtm_reproj_file) is np.ndarray:
            source = 'numpy'
            topo_kwarg = {'array': dtm_reproj_file}
        else:
            raise AttributeError('dtm_proj_file must be either a path to gdal or a'
                                 'numpy array with values')
        geo_model.set_topography(source=source, **topo_kwarg)

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

        # Faults relation
        geo_model.set_fault_relation(faults_rel_matrix)

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

    try:
        colours = pd.read_csv(groups_file).set_index('code')['colour']
        # Drop formations that do not exist in surfaces
        colours = colours.loc[colours.index.isin(geo_model.surfaces.df['surface'])].to_dict()

        geo_model._surfaces.colors.change_colors(colours)
    except KeyError:
        pass

    if compute is True:
        gp.set_interpolator(geo_model, dtype='float64',
                            # verbose=['solve_kriging']
                            )

        # Increasing nugget effect
        geo_model.modify_surface_points(
            geo_model.surface_points.df.index,
            smooth=0.1
        )

        geo_model.modify_orientations(
            geo_model.orientations.df.index,
            smooth=0.01
        )

        new_range = geo_model.get_additional_data().loc[('Kriging', 'range'), 'values'] * 0.5
        geo_model.modify_kriging_parameters('range', new_range)

        gp.compute_model(geo_model)

    if vtk is True:
        gp.plot_3d(geo_model, show_topography=True,
                   image=image_2d,
                   show_lith=True,
                   **plot_3d_kwargs
                   )

    if vtk_path is not None:
        gp._plot.export_to_vtk(geo_model, path=vtk_path, name=model_name + '.vtk',
                               voxels=False, block=None, surfaces=True)

    return geo_model