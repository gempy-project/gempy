"""
The aim of this module is to encapsulate the loading functionality. Also this will enable better error handling
when some of the data files are missing
"""

import pandas as pn
import numpy as np




def load_kriging_data(geo_model, path, name):
    geo_model.additional_data.kriging_data.df = pn.read_csv(
        f'{path}/{name}_kriging_data.csv', index_col=0,
        dtype={'range': 'float64', '$C_o$': 'float64', 'drift equations': object,
               'nugget grad': 'float64', 'nugget scalar': 'float64'})

def load_options(geo_model, path, name):
    geo_model.additional_data.options.df = pn.read_csv(f'{path}/{name}_options.csv', index_col=0,
                                            dtype={'dtype': 'category', 'output': 'category',
                                            'theano_optimizer': 'category', 'device': 'category',
                                            'verbosity': object})
    geo_model.additional_data.options.df['dtype'].cat.set_categories(['float32', 'float64'], inplace=True)
    geo_model.additional_data.options.df['theano_optimizer'].cat.set_categories(['fast_run', 'fast_compile'], inplace=True)
    geo_model.additional_data.options.df['device'].cat.set_categories(['cpu', 'cuda'], inplace=True)
    geo_model.additional_data.options.df['output'].cat.set_categories(['geology', 'gradients'], inplace=True)

def load_series(geo_model, path, name):
    # do series properly - this needs proper check
    geo_model.series.df = pn.read_csv(f'{path}/{name}_series.csv', index_col=0,
                                            dtype={'order_series': 'int32', 'BottomRelation': 'category'})
    series_index = pn.CategoricalIndex(geo_model.series.df.index.values)
    # geo_model.series.df.index = pn.CategoricalIndex(series_index)
    geo_model.series.df.index = series_index
    geo_model.series.df['BottomRelation'].cat.set_categories(['Erosion', 'Onlap'], inplace=True)

    cat_series = geo_model.series.df.index.values


def load_faults(geo_model, path, name):
    # do faults properly - check
    geo_model.faults.df = pn.read_csv(f'{path}/{name}_faults.csv', index_col=0,
                                            dtype={'isFault': 'bool', 'isFinite': 'bool'})
    geo_model.faults.df.index = series_index


def load_faults_relations(geo_model, path, name):

    # do faults relations properly - this is where I struggle
    geo_model.faults.faults_relations_df = pn.read_csv(f'{path}/{name}_faults_relations.csv', index_col=0)
    geo_model.faults.faults_relations_df.index = series_index
    geo_model.faults.faults_relations_df.columns = series_index

    geo_model.faults.faults_relations_df.fillna(False, inplace=True)


def load_surfaces(geo_model, path, name):

    # do surfaces properly
    geo_model.surfaces.df = pn.read_csv(f'{path}/{name}_surfaces.csv', index_col=0,
                                            dtype={'surface': 'str', 'series': 'category',
                                                   'order_surfaces': 'int64', 'isBasement': 'bool', 'id': 'int64'})
    geo_model.surfaces.df['series'].cat.set_categories(cat_series, inplace=True)

    cat_surfaces = geo_model.surfaces.df['surface'].values


def load_orientations(geo_model, path, name):

    # do orientations properly, reset all dtypes
    geo_model.orientations.df = pn.read_csv(f'{path}/{name}_orientations.csv', index_col=0,
                                            dtype={'X': 'float64', 'Y': 'float64', 'Z': 'float64',
                                                   'X_r': 'float64', 'Y_r': 'float64', 'Z_r': 'float64',
                                                   'dip': 'float64', 'azimuth': 'float64', 'polarity': 'float64',
                                                   'surface': 'category', 'series': 'category',
                                                   'id': 'int64', 'order_series': 'int64'})
    geo_model.orientations.df['surface'].cat.set_categories(cat_surfaces, inplace=True)
    geo_model.orientations.df['series'].cat.set_categories(cat_series, inplace=True)


def load_surface_points(geo_model, path, name):

    # do surface_points properly, reset all dtypes
    geo_model.surface_points.df = pn.read_csv(f'{path}/{name}_surface_points.csv', index_col=0,
                                              dtype={'X': 'float64', 'Y': 'float64', 'Z': 'float64',
                                                     'X_r': 'float64', 'Y_r': 'float64', 'Z_r': 'float64',
                                                     'surface': 'category', 'series': 'category',
                                                     'id': 'int64', 'order_series': 'int64'})
    geo_model.surface_points.df['surface'].cat.set_categories(cat_surfaces, inplace=True)
    geo_model.surface_points.df['series'].cat.set_categories(cat_series, inplace=True)

    # update structure from loaded input
    geo_model.additional_data.structure_data.update_structure_from_input()


def load_solutions(geo_model, path, name):

    # load solutions in npy files
    geo_model.solutions.lith_block = np.load(f'{path}/{name}_lith_block.npy')
    geo_model.solutions.scalar_field_lith = np.load(f"{path}/{name}_scalar_field_lith.npy")
    geo_model.solutions.fault_blocks = np.load(f'{path}/{name}_fault_blocks.npy')
    geo_model.solutions.scalar_field_faults = np.load(f'{path}/{name}_scalar_field_faults.npy')
    geo_model.solutions.gradient = np.load(f'{path}/{name}_gradient.npy')
    geo_model.solutions.values_block = np.load(f'{path}/{name}_values_block.npy')

    geo_model.solutions.additional_data.kriging_data.df = geo_model.additional_data.kriging_data.df
    geo_model.solutions.additional_data.options.df = geo_model.additional_data.options.df
    geo_model.solutions.additional_data.rescaling_data.df = geo_model.additional_data.rescaling_data.df