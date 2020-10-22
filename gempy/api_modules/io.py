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
def save_model(model: Project, name=None, path=None, compress=True):
    # try:
    #     model._grid.topography.topo = None
    # except AttributeError:
    #     pass
    model.save_model(name, path, compress)
    return True


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
        path = path[:-4]
        print("is path", path )
        try:
            shutil.unpack_archive(path + '.zip', extract_dir=path)
        except ValueError as e:
            raise ValueError(e)
        is_compressed = True

    # if the path does not include .zip but exist
    elif os.path.isfile(f'{path}.zip'):

        try:
            shutil.unpack_archive(path + '.zip', extract_dir=path)
        except ValueError as e:
            raise ValueError(e)
        is_compressed = True

    # create model with extent and resolution from csv - check
    geo_model = create_model()
    init_data(geo_model, np.load(f'{path}/{name}_extent.npy'),
              np.load(f'{path}/{name}_resolution.npy'))

    try:
        geo_model.set_topography(source='saved',
                                 filepath=f'{path}/{name}_topography.npy')
    except FileNotFoundError:
        pass

    geo_model._additional_data.kriging_data.df = pn.read_csv(f'{path}/{name}_kriging_data.csv', index_col=0,
                                                             dtype={'range': 'float64', '$C_o$': 'float64',
                                                                    'drift equations': object,
                                                                    'nugget grad': 'float64',
                                                                    'nugget scalar': 'float64'})

    geo_model._additional_data.kriging_data.str2int_u_grade()

    geo_model._additional_data.options.df = pn.read_csv(f'{path}/{name}_options.csv', index_col=0,
                                                        dtype={'dtype': 'category', 'output': 'category',
                                                               'theano_optimizer': 'category', 'device': 'category',
                                                               'verbosity': object})
    geo_model._additional_data.options.df['dtype'].cat.set_categories(['float32', 'float64'], inplace=True)
    geo_model._additional_data.options.df['theano_optimizer'].cat.set_categories(['fast_run', 'fast_compile'],
                                                                                 inplace=True)
    geo_model._additional_data.options.df['device'].cat.set_categories(['cpu', 'cuda'], inplace=True)
    geo_model._additional_data.options.df['output'].cat.set_categories(['geology', 'gradients'], inplace=True)
    geo_model._additional_data.options.df.loc['values', 'verbosity'] = None
    # do series properly - this needs proper check

    # Load series
    s = pn.read_csv(f'{path}/{name}_series.csv', index_col=0, dtype={'order_series': 'int32',
                                                                     'BottomRelation': 'category'})

    f = pn.read_csv(f'{path}/{name}_faults.csv', index_col=0,
                                       dtype={'isFault': 'bool', 'isFinite': 'bool'})

    stack = pn.concat([s, f], axis=1, sort=False)
    stack = stack.loc[:, ~stack.columns.duplicated()]
    geo_model._stack.df = stack
    series_index = pn.CategoricalIndex(geo_model._stack.df.index.values)
    # geo_model.series.df.index = pn.CategoricalIndex(series_index)
    geo_model._stack.df.index = series_index
    geo_model._stack.df['BottomRelation'].cat.set_categories(['Erosion', 'Onlap', 'Fault'], inplace=True)
    try:
        geo_model._stack.df['isActive']
    except KeyError:
        geo_model._stack.df['isActive'] = False

    cat_series = geo_model._stack.df.index.values
    # # do faults relations properly - this is where I struggle
    geo_model._faults.faults_relations_df = pn.read_csv(f'{path}/{name}_faults_relations.csv', index_col=0)
    geo_model._faults.faults_relations_df.index = series_index
    geo_model._faults.faults_relations_df.columns = series_index

    geo_model._faults.faults_relations_df.fillna(False, inplace=True)

    # do surfaces properly
    surf_df = pn.read_csv(f'{path}/{name}_surfaces.csv', index_col=0,
                          dtype={'surface': 'str', 'series': 'category',
                                 'order_surfaces': 'int64', 'isBasement': 'bool', 'id': 'int64',
                                 'color': 'str'})
    c_ = surf_df.columns[~(surf_df.columns.isin(geo_model._surfaces._columns_vis_drop))]
    geo_model._surfaces.df[c_] = surf_df[c_]
    geo_model._surfaces.df['series'].cat.reorder_categories(np.asarray(geo_model._stack.df.index),
                                                            ordered=False, inplace=True)
    geo_model._surfaces.sort_surfaces()

    geo_model._surfaces.colors.generate_colordict()
    geo_model._surfaces.df['series'].cat.set_categories(cat_series, inplace=True)

    try:
        geo_model._surfaces.df['isActive']
    except KeyError:
        geo_model._surfaces.df['isActive'] = False

    cat_surfaces = geo_model._surfaces.df['surface'].values

    # do orientations properly, reset all dtypes
    geo_model._orientations.df = pn.read_csv(f'{path}/{name}_orientations.csv', index_col=0,
                                             dtype={'X': 'float64', 'Y': 'float64', 'Z': 'float64',
                                                    'X_r': 'float64', 'Y_r': 'float64', 'Z_r': 'float64',
                                                    'dip': 'float64', 'azimuth': 'float64', 'polarity': 'float64',
                                                    'surface': 'category', 'series': 'category',
                                                    'id': 'int64', 'order_series': 'int64'})
    geo_model._orientations.df['surface'].cat.set_categories(cat_surfaces, inplace=True)
    geo_model._orientations.df['series'].cat.set_categories(cat_series, inplace=True)

    # do surface_points properly, reset all dtypes
    geo_model._surface_points.df = pn.read_csv(f'{path}/{name}_surface_points.csv', index_col=0,
                                               dtype={'X': 'float64', 'Y': 'float64', 'Z': 'float64',
                                                      'X_r': 'float64', 'Y_r': 'float64', 'Z_r': 'float64',
                                                      'surface': 'category', 'series': 'category',
                                                      'id': 'int64', 'order_series': 'int64'})
    geo_model._surface_points.df['surface'].cat.set_categories(cat_surfaces, inplace=True)
    geo_model._surface_points.df['series'].cat.set_categories(cat_series, inplace=True)

    # Code to add smooth columns for models saved before gempy 2.0bdev4
    try:
        geo_model._surface_points.df['smooth']
    except KeyError:
        geo_model._surface_points.df['smooth'] = 1e-7

    try:
        geo_model._orientations.df['smooth']
    except KeyError:
        geo_model._orientations.df['smooth'] = 0.01

    # update structure from loaded input
    geo_model._additional_data.structure_data.update_structure_from_input()
    geo_model._rescaling.rescale_data()
    geo_model.update_from_series()
    geo_model.update_from_surfaces()
    geo_model.update_structure()

    if recompile is True:
        from gempy.api_modules.setters import set_interpolator
        set_interpolator(geo_model, verbose=[0])

    # Cleaning temp files
    if is_compressed:
        shutil.rmtree(path)

    return geo_model


# endregion


# From here down it needs to be reworked to make load more reasonable
# ----------------

def load_kriging_data(geo_model, path, name):
    geo_model._additional_data.kriging_data.df = pn.read_csv(
        f'{path}/{name}_kriging_data.csv', index_col=0,
        dtype={'range': 'float64', '$C_o$': 'float64', 'drift equations': object,
               'nugget grad': 'float64', 'nugget scalar': 'float64'})


def load_options(geo_model, path, name):
    geo_model._additional_data.options.df = pn.read_csv(
        f'{path}/{name}_options.csv', index_col=0,
        dtype={'dtype': 'category', 'output': 'category',
               'theano_optimizer': 'category', 'device': 'category',
               'verbosity': object})
    geo_model._additional_data.options.df['dtype'].cat.set_categories(
        ['float32', 'float64'], inplace=True)
    geo_model._additional_data.options.df['theano_optimizer'].cat.set_categories(
        ['fast_run', 'fast_compile'],
        inplace=True)
    geo_model._additional_data.options.df['device'].cat.set_categories(['cpu', 'cuda'], inplace=True)
    geo_model._additional_data.options.df['output'].cat.set_categories(['geology', 'gradients'], inplace=True)


def load_series(geo_model, path, name):
    # do series properly - this needs proper check
    geo_model._stack.df = pn.read_csv(f'{path}/{name}_series.csv', index_col=0,
                                      dtype={'order_series': 'int32', 'BottomRelation': 'category'})
    series_index = pn.CategoricalIndex(geo_model._stack.df.index.values)
    # geo_model.series.df.index = pn.CategoricalIndex(series_index)
    geo_model._stack.df.index = series_index
    geo_model._stack.df['BottomRelation'].cat.set_categories(['Erosion', 'Onlap'], inplace=True)

    cat_series = geo_model._stack.df.index.values


def load_faults(geo_model, path, name):
    # do faults properly - check
    geo_model._faults.df = pn.read_csv(f'{path}/{name}_faults.csv', index_col=0,
                                       dtype={'isFault': 'bool', 'isFinite': 'bool'})
    # geo_model._faults.df.index = series_index


def load_faults_relations(geo_model, path, name):
    # do faults relations properly - this is where I struggle
    geo_model._faults.faults_relations_df = pn.read_csv(f'{path}/{name}_faults_relations.csv', index_col=0)
    geo_model._faults.faults_relations_df.index = series_index
    geo_model._faults.faults_relations_df.columns = series_index

    geo_model._faults.faults_relations_df.fillna(False, inplace=True)


def load_surfaces(geo_model, path, name):
    # do surfaces properly
    geo_model._surfaces.df = pn.read_csv(f'{path}/{name}_surfaces.csv', index_col=0,
                                         dtype={'surface': 'str', 'series': 'category',
                                                'order_surfaces': 'int64', 'isBasement': 'bool', 'id': 'int64'})
    geo_model._surfaces.df['series'].cat.set_categories(cat_series, inplace=True)

    cat_surfaces = geo_model._surfaces.df['surface'].values


def load_orientations(geo_model, path, name):
    # do orientations properly, reset all dtypes
    geo_model._orientations.df = pn.read_csv(f'{path}/{name}_orientations.csv', index_col=0,
                                             dtype={'X': 'float64', 'Y': 'float64', 'Z': 'float64',
                                                    'X_r': 'float64', 'Y_r': 'float64', 'Z_r': 'float64',
                                                    'dip': 'float64', 'azimuth': 'float64', 'polarity': 'float64',
                                                    'surface': 'category', 'series': 'category',
                                                    'id': 'int64', 'order_series': 'int64'})
    geo_model._orientations.df['surface'].cat.set_categories(cat_surfaces, inplace=True)
    geo_model._orientations.df['series'].cat.set_categories(cat_series, inplace=True)


def load_surface_points(geo_model, path, name):
    # do surface_points properly, reset all dtypes
    geo_model._surface_points.df = pn.read_csv(f'{path}/{name}_surface_points.csv', index_col=0,
                                               dtype={'X': 'float64', 'Y': 'float64', 'Z': 'float64',
                                                      'X_r': 'float64', 'Y_r': 'float64', 'Z_r': 'float64',
                                                      'surface': 'category', 'series': 'category',
                                                      'id': 'int64', 'order_series': 'int64'})
    geo_model._surface_points.df['surface'].cat.set_categories(cat_surfaces, inplace=True)
    geo_model._surface_points.df['series'].cat.set_categories(cat_series, inplace=True)

    # update structure from loaded input
    geo_model._additional_data.structure_data.update_structure_from_input()


def load_solutions(geo_model, path, name):
    # load solutions in npy files
    geo_model.solutions.lith_block = np.load(f'{path}/{name}_lith_block.npy')
    geo_model.solutions.scalar_field_lith = np.load(f"{path}/{name}_scalar_field_lith.npy")
    geo_model.solutions.fault_blocks = np.load(f'{path}/{name}_fault_blocks.npy')
    geo_model.solutions.scalar_field_faults = np.load(f'{path}/{name}_scalar_field_faults.npy')
    geo_model.solutions.gradient = np.load(f'{path}/{name}_gradient.npy')
    geo_model.solutions.values_block = np.load(f'{path}/{name}_values_block.npy')

    geo_model.solutions._additional_data.kriging_data.df = geo_model._additional_data.kriging_data.df
    geo_model.solutions._additional_data.options.df = geo_model._additional_data.options.df
    geo_model.solutions._additional_data.rescaling_data.df = geo_model._additional_data.rescaling_data.df
