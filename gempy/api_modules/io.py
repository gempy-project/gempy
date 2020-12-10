"""
The aim of this module is to encapsulate the loading functionality. Also this will enable
 better error handling when some of the data files are missing
"""
import os
import pathlib
import shutil

import pandas as pn
import numpy as np

# region Save
from gempy import Project, create_model, init_data
from gempy.utils.meta import _setdoc


@_setdoc(Project.save_model_pickle.__doc__)
def save_model_to_pickle(model: Project, path=None):
    model.save_model_pickle(path)
    return True


@_setdoc(Project.save_model.__doc__)
def save_model(model: Project, name=None, path=None, compress=True,
               solution=False, **kwargs):
    name, path = default_path_and_name(model, name, path)
    model.save_model(name, path, compress)
    if solution is True:
        solution_to_netcdf(kwargs, model, name, path)
    if compress is True:
        shutil.make_archive(name, 'zip', path)
        shutil.rmtree(path)
    return True


def default_path_and_name(model, name, path):
    if name is None:
        name = model.meta.project_name
    if not path:
        path = './'
    path = f'{path}/{name}'
    if os.path.isdir(path):
        print("Directory already exists, files will be overwritten")
    else:
        os.makedirs(f'{path}')
    return name, path


def solution_to_netcdf(kwargs, model, name, path):
    try:
        model.solutions.to_netcdf(path, name, **kwargs)
    except AttributeError:
        raise AttributeError('You need to install Subsurface to be able to '
                             'write the solutions.')


@_setdoc(Project.load_model_pickle.__doc__)
def load_model_pickle(path):
    """
    Read InputData object from python pickle.

    Args:
       path (str): path where save the pickle

    Returns:
        :class:`Project`

    """
    return Project.load_model_pickle(path)


def load_model(name=None, path=None, recompile=False):
    """
    Loading model saved with model.save_model function.

    Args:
        name: name of folder with saved files
        path (str): path to folder directory or the zip file
        recompile (bool): if true, theano functions will be recompiled

    Returns:
        :class:`Project`

    """
    # TODO: Divide each dataframe in its own function and move them as
    #  method of the class
    # TODO: Include try except in case some of the datafiles is missing
    #

    # Default path
    is_compressed = False
    if path is None:
        path = f'./{name}'
    p = pathlib.Path(path)

    # If the path includes .zip
    if p.suffix == '.zip':
        is_compressed, path = _unpack_model_if_compressed_includes_zip(is_compressed, path)

    # if the path does not include .zip but exist
    elif os.path.isfile(f'{path}.zip'):
        is_compressed = _unpack_model_if_compressed_no_zip(is_compressed, path)

    # create model with extent and resolution from csv - check
    geo_model = create_model()
    init_data(
        geo_model,
        np.load(f'{path}/{name}_extent.npy'),
        np.load(f'{path}/{name}_resolution.npy')
    )

    _load_files_into_geo_model(geo_model, name, path)

    if recompile is True:
        from gempy.api_modules.setters import set_interpolator
        set_interpolator(geo_model, verbose=[0])

    # Cleaning temp files
    if is_compressed:
        shutil.rmtree(path)

    return geo_model


def _load_files_into_geo_model(geo_model, name, path):
    _load_topography(geo_model, name, path)
    _load_additional_data(geo_model, name, path)
    # Load series
    cat_series = _load_stack(geo_model, name, path)
    # do surfaces properly
    cat_surfaces = _load_surfaces(cat_series, geo_model, name, path)
    # do orientations properly, reset all dtypes
    _load_orientations(cat_series, cat_surfaces, geo_model, name, path)
    # do surface_points properly, reset all dtypes
    _load_surface_points(cat_series, cat_surfaces, geo_model, name, path)
    # update structure from loaded input
    _update_structure_and_mapping(geo_model)


def _update_structure_and_mapping(geo_model):
    geo_model._additional_data.structure_data.update_structure_from_input()
    geo_model._rescaling.rescale_data()
    geo_model.update_from_series()
    geo_model.update_from_surfaces()
    geo_model.update_structure()


def _load_surface_points(cat_series, cat_surfaces, geo_model, name, path):
    geo_model._surface_points.df = pn.read_csv(f'{path}/{name}_surface_points.csv',
                                               index_col=0,
                                               dtype={'X': 'float64', 'Y': 'float64',
                                                      'Z': 'float64',
                                                      'X_r': 'float64',
                                                      'Y_r': 'float64',
                                                      'Z_r': 'float64',
                                                      'surface': 'category',
                                                      'series': 'category',
                                                      'id': 'int64',
                                                      'order_series': 'int64'})
    geo_model._surface_points.df['surface'].cat.set_categories(cat_surfaces,
                                                               inplace=True)
    geo_model._surface_points.df['series'].cat.set_categories(cat_series,
                                                              inplace=True)
    # Code to add smooth columns for models saved before gempy 2.0bdev4
    try:
        geo_model._surface_points.df['smooth']
    except KeyError:
        geo_model._surface_points.df['smooth'] = 1e-7


def _load_orientations(cat_series, cat_surfaces, geo_model, name, path):
    geo_model._orientations.df = pn.read_csv(f'{path}/{name}_orientations.csv',
                                             index_col=0,
                                             dtype={'X': 'float64', 'Y': 'float64',
                                                    'Z': 'float64',
                                                    'X_r': 'float64',
                                                    'Y_r': 'float64',
                                                    'Z_r': 'float64',
                                                    'dip': 'float64',
                                                    'azimuth': 'float64',
                                                    'polarity': 'float64',
                                                    'surface': 'category',
                                                    'series': 'category',
                                                    'id': 'int64',
                                                    'order_series': 'int64'})
    geo_model._orientations.df['surface'].cat.set_categories(cat_surfaces,
                                                             inplace=True)
    geo_model._orientations.df['series'].cat.set_categories(cat_series, inplace=True)

    try:
        geo_model._orientations.df['smooth']
    except KeyError:
        geo_model._orientations.df['smooth'] = 0.01


def _load_surfaces(cat_series, geo_model, name, path):
    surf_df = pn.read_csv(f'{path}/{name}_surfaces.csv', index_col=0,
                          dtype={'surface': 'str', 'series': 'category',
                                 'order_surfaces': 'int64', 'isBasement': 'bool',
                                 'id': 'int64',
                                 'color': 'str'})
    c_ = surf_df.columns[
        ~(surf_df.columns.isin(geo_model._surfaces._columns_vis_drop))]
    geo_model._surfaces.df[c_] = surf_df[c_]
    geo_model._surfaces.df['series'].cat.reorder_categories(
        np.asarray(geo_model._stack.df.index),
        ordered=False, inplace=True)
    geo_model._surfaces.sort_surfaces()
    geo_model._surfaces.colors.generate_colordict()
    geo_model._surfaces.df['series'].cat.set_categories(cat_series, inplace=True)
    try:
        geo_model._surfaces.df['isActive']
    except KeyError:
        geo_model._surfaces.df['isActive'] = False
    cat_surfaces = geo_model._surfaces.df['surface'].values
    return cat_surfaces


def _load_stack(geo_model, name, path):
    s = pn.read_csv(f'{path}/{name}_series.csv', index_col=0,
                    dtype={'order_series': 'int32',
                           'BottomRelation': 'category'})
    f = pn.read_csv(f'{path}/{name}_faults.csv', index_col=0,
                    dtype={'isFault': 'bool', 'isFinite': 'bool'})
    stack = pn.concat([s, f], axis=1, sort=False)
    stack = stack.loc[:, ~stack.columns.duplicated()]
    geo_model._stack.df = stack
    series_index = pn.CategoricalIndex(geo_model._stack.df.index.values)
    # geo_model.series.df.index = pn.CategoricalIndex(series_index)
    geo_model._stack.df.index = series_index
    geo_model._stack.df['BottomRelation'].cat.set_categories(
        ['Erosion', 'Onlap', 'Fault'], inplace=True)
    try:
        geo_model._stack.df['isActive']
    except KeyError:
        geo_model._stack.df['isActive'] = False
    cat_series = geo_model._stack.df.index.values
    # # do faults relations properly - this is where I struggle
    geo_model._faults.faults_relations_df = pn.read_csv(
        f'{path}/{name}_faults_relations.csv', index_col=0)
    geo_model._faults.faults_relations_df.index = series_index
    geo_model._faults.faults_relations_df.columns = series_index
    geo_model._faults.faults_relations_df.fillna(False, inplace=True)
    return cat_series


def _load_additional_data(geo_model, name, path):
    geo_model._additional_data.kriging_data.df = pn.read_csv(
        f'{path}/{name}_kriging_data.csv', index_col=0,
        dtype={'range': 'float64', '$C_o$': 'float64',
               'drift equations': object,
               'nugget grad': 'float64',
               'nugget scalar': 'float64'})
    geo_model._additional_data.kriging_data.str2int_u_grade()
    geo_model._additional_data.options.df = pn.read_csv(f'{path}/{name}_options.csv',
                                                        index_col=0,
                                                        dtype={'dtype': 'category',
                                                               'output': 'category',
                                                               'theano_optimizer': 'category',
                                                               'device': 'category',
                                                               'verbosity': object})
    geo_model._additional_data.options.df['dtype'].cat.set_categories(
        ['float32', 'float64'], inplace=True)
    geo_model._additional_data.options.df['theano_optimizer'].cat.set_categories(
        ['fast_run', 'fast_compile'],
        inplace=True)
    geo_model._additional_data.options.df['device'].cat.set_categories(
        ['cpu', 'cuda'], inplace=True)
    geo_model._additional_data.options.df['output'].cat.set_categories(
        ['geology', 'gradients'], inplace=True)
    geo_model._additional_data.options.df.loc['values', 'verbosity'] = None


def _load_topography(geo_model, name, path):
    try:
        geo_model.set_topography(source='saved',
                                 filepath=f'{path}/{name}_topography.npy')
    except FileNotFoundError:
        pass


def _unpack_model_if_compressed_no_zip(is_compressed, path):
    try:
        shutil.unpack_archive(path + '.zip', extract_dir=path)
    except ValueError as e:
        raise ValueError(e)
    is_compressed = True
    return is_compressed


def _unpack_model_if_compressed_includes_zip(is_compressed, path):
    path = path[:-4]
    print("is path", path)
    try:
        shutil.unpack_archive(path + '.zip', extract_dir=path)
    except ValueError as e:
        raise ValueError(e)
    is_compressed = True
    return is_compressed, path

# endregion
