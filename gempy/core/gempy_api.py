"""
    This file is part of gempy.

    gempy is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Foobar is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with gempy.  If not, see <http://www.gnu.org/licenses/>.


    Created on 10/10 /2018

    @author: Miguel de la Varga
"""

from os import path
import sys
import numpy as np
import pandas as pn
from numpy import ndarray
from typing import Union
import warnings

# This is for sphenix to find the packages
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from gempy.core.model import Model, DataMutation, AdditionalData, Faults, Grid, MetaData, Orientations, RescaledData, Series, SurfacePoints,\
    Surfaces, Options, Structure, KrigingParameters
from gempy.core.solution import Solution
from gempy.utils.meta import _setdoc
from gempy.core.interpolator import InterpolatorGravity, InterpolatorModel


# This warning comes from numpy complaining about a theano optimization
warnings.filterwarnings("ignore",
                        message='.* a non-tuple sequence for multidimensional indexing is deprecated; use*.',
                        append=True)


# region Model
@_setdoc(Model.__doc__)
def create_model(project_name='default_project'):
    """
    Create Model Object


    Returns:
        Model
    """

    return Model(project_name)


@_setdoc(Model.save_model_pickle.__doc__)
def save_model_to_pickle(model: Model, path=None):

    model.save_model_pickle(path)
    return True


@_setdoc(Model.save_model.__doc__)
def save_model(model: Model, name=None, path=None):

    model.save_model(name, path)
    return True


@_setdoc(Model.load_model_pickle.__doc__)
def load_model_pickle(path):
    """
    Read InputData object from python pickle.

    Args:
       path (str): path where save the pickle

    Returns:
        :class:`gempy.data_management.InputData`

    """
    return Model.load_model_pickle(path)


def load_model(name, path=None, recompile=False):
    """
    Loading model saved with model.save_model function.

    Args:
        name: name of folder with saved files
        path (str): path to folder directory
        recompile (bool): if true, theano functions will be recompiled

    Returns:
        :class:`gempy.core.model

    """
    # TODO: Divide each dataframe in its own function
    # TODO: Include try except in case some of the datafiles is missing
    #

    if not path:
        path = './'
    path = f'{path}/{name}'

    # create model with extent and resolution from csv - check
    geo_model = create_model()
    init_data(geo_model, np.load(f'{path}/{name}_extent.npy'), np.load(f'{path}/{name}_resolution.npy'))
    # rel_matrix = np.load()
    # set additonal data
    geo_model.additional_data.kriging_data.df = pn.read_csv(f'{path}/{name}_kriging_data.csv', index_col=0,
                                            dtype={'range': 'float64', '$C_o$': 'float64', 'drift equations': object,
                                            'nugget grad': 'float64', 'nugget scalar': 'float64'})

    geo_model.additional_data.kriging_data.str2int_u_grage()

    geo_model.additional_data.options.df = pn.read_csv(f'{path}/{name}_options.csv', index_col=0,
                                            dtype={'dtype': 'category', 'output': 'category',
                                            'theano_optimizer': 'category', 'device': 'category',
                                            'verbosity': object})
    geo_model.additional_data.options.df['dtype'].cat.set_categories(['float32', 'float64'], inplace=True)
    geo_model.additional_data.options.df['theano_optimizer'].cat.set_categories(['fast_run', 'fast_compile'], inplace=True)
    geo_model.additional_data.options.df['device'].cat.set_categories(['cpu', 'cuda'], inplace=True)
    geo_model.additional_data.options.df['output'].cat.set_categories(['geology', 'gradients'], inplace=True)

    # do series properly - this needs proper check
    geo_model.series.df = pn.read_csv(f'{path}/{name}_series.csv', index_col=0,
                                            dtype={'order_series': 'int32', 'BottomRelation': 'category'})
    series_index = pn.CategoricalIndex(geo_model.series.df.index.values)
    # geo_model.series.df.index = pn.CategoricalIndex(series_index)
    geo_model.series.df.index = series_index
    geo_model.series.df['BottomRelation'].cat.set_categories(['Erosion', 'Onlap', 'Fault'], inplace=True)

    cat_series = geo_model.series.df.index.values

    # do faults properly - check
    geo_model.faults.df = pn.read_csv(f'{path}/{name}_faults.csv', index_col=0,
                                            dtype={'isFault': 'bool', 'isFinite': 'bool'})
    geo_model.faults.df.index = series_index

    # # do faults relations properly - this is where I struggle
    geo_model.faults.faults_relations_df = pn.read_csv(f'{path}/{name}_faults_relations.csv', index_col=0)
    geo_model.faults.faults_relations_df.index = series_index
    geo_model.faults.faults_relations_df.columns = series_index

    geo_model.faults.faults_relations_df.fillna(False, inplace=True)

    # do surfaces properly
    surf_df = pn.read_csv(f'{path}/{name}_surfaces.csv', index_col=0,
                          dtype={'surface': 'str', 'series': 'category',
                                 'order_surfaces': 'int64', 'isBasement': 'bool', 'id': 'int64',
                                 'color': 'str'})
    c_ = surf_df.columns[~(surf_df.columns.isin(geo_model.surfaces._columns_vis_drop))]
    geo_model.surfaces.df[c_] = surf_df[c_]

    geo_model.surfaces.colors.generate_colordict()
    geo_model.surfaces.df['series'].cat.set_categories(cat_series, inplace=True)

    cat_surfaces = geo_model.surfaces.df['surface'].values

    # do orientations properly, reset all dtypes
    geo_model.orientations.df = pn.read_csv(f'{path}/{name}_orientations.csv', index_col=0,
                                            dtype={'X': 'float64', 'Y': 'float64', 'Z': 'float64',
                                                   'X_r': 'float64', 'Y_r': 'float64', 'Z_r': 'float64',
                                                   'dip': 'float64', 'azimuth': 'float64', 'polarity': 'float64',
                                                   'surface': 'category', 'series': 'category',
                                                   'id': 'int64', 'order_series': 'int64'})
    geo_model.orientations.df['surface'].cat.set_categories(cat_surfaces, inplace=True)
    geo_model.orientations.df['series'].cat.set_categories(cat_series, inplace=True)

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

    # # load solutions in npy files
    # geo_model.solutions.lith_block = np.load(f'{path}/{name}_lith_block.npy')
    # geo_model.solutions.scalar_field_lith = np.load(f"{path}/{name}_scalar_field_lith.npy")
    # geo_model.solutions.fault_blocks = np.load(f'{path}/{name}_fault_blocks.npy')
    # geo_model.solutions.scalar_field_faults = np.load(f'{path}/{name}_scalar_field_faults.npy')
    # geo_model.solutions.gradient = np.load(f'{path}/{name}_gradient.npy')
    # geo_model.solutions.values_block = np.load(f'{path}/{name}_values_block.npy')
    #
    # geo_model.solutions.additional_data.kriging_data.df = geo_model.additional_data.kriging_data.df
    # geo_model.solutions.additional_data.options.df = geo_model.additional_data.options.df
    # geo_model.solutions.additional_data.rescaling_data.df = geo_model.additional_data.rescaling_data.df

    geo_model.update_from_series()
    geo_model.update_from_surfaces()
    geo_model.update_structure()

    if recompile is True:
        set_interpolation_data(geo_model, verbose=[0])

    return geo_model
# endregion


# region Series functionality

def set_series(geo_model: Model, mapping_object: Union[dict, pn.Categorical] = None,
               set_series=True, sort_data: bool = True):
    warnings.warn("set_series will get deprecated in the next version of gempy. It still exist only to keep"
                  "the behaviour equal to older version. Use map_series_to_surfaces isnead.", FutureWarning)

    map_series_to_surfaces(geo_model, mapping_object, set_series, sort_data)


def map_series_to_surfaces(geo_model: Model, mapping_object: Union[dict, pn.Categorical] = None,
                             set_series=True, sort_geometric_data: bool = True, remove_unused_series=True):
    """
    Map the series (column) of the Surface object accordingly to the mapping_object
    Args:
        geo_model:
        mapping_object:
        set_series:
        sort_data:
        remove_unused_series:
        quiet:

    Returns:

    """
    geo_model.map_series_to_surfaces(mapping_object, set_series, sort_geometric_data, remove_unused_series)
    return geo_model.surfaces
# endregion


# region Point-Orientation functionality
@_setdoc([Model.read_data.__doc__])
def read_csv(geo_model: Model, path_i=None, path_o=None, **kwargs):
    if path_i is not None or path_o is not None:
        geo_model.read_data(path_i, path_o, **kwargs)
    return True


def set_orientation_from_surface_points(geo_model, indices_array):
    """
    Create and set orientations from at least 3 points of the :attr:`gempy.data_management.InputData.surface_points`
     Dataframe
    Args:
        geo_model (:class:`gempy.data_management.InputData`)
        indices_array (array-like): 1D or 2D array with the pandas indices of the
          :attr:`gempy.data_management.InputData.surface_points`. If 2D every row of the 2D matrix will be used to create an
          orientation


    Returns:
        :attr:`gempy.data_management.InputData.orientations`: Already updated inplace
    """

    if np.ndim(indices_array) is 1:
        indices = indices_array
        form = geo_model.surface_points['surface'].loc[indices].unique()
        assert form.shape[0] is 1, 'The interface points must belong to the same surface'
        form = form[0]
        print()
        ori_parameters = geo_model.create_orientation_from_surface_points(indices)
        geo_model.add_orientation(X=ori_parameters[0], Y=ori_parameters[1], Z=ori_parameters[2],
                                  dip=ori_parameters[3], azimuth=ori_parameters[4], polarity=ori_parameters[5],
                                  G_x=ori_parameters[6], G_y=ori_parameters[7], G_z=ori_parameters[8],
                                  surface=form)
    elif np.ndim(indices_array) is 2:
        for indices in indices_array:
            form = geo_model.surface_points['surface'].loc[indices].unique()
            assert form.shape[0] is 1, 'The interface points must belong to the same surface'
            form = form[0]
            ori_parameters = geo_model.create_orientation_from_surface_points(indices)
            geo_model.add_orientation(X=ori_parameters[0], Y=ori_parameters[1], Z=ori_parameters[2],
                                      dip=ori_parameters[3], azimuth=ori_parameters[4], polarity=ori_parameters[5],
                                      G_x=ori_parameters[6], G_y=ori_parameters[7], G_z=ori_parameters[8],
                                      surface=form)

    geo_model.update_df()
    return geo_model.orientations


def rescale_data(geo_model: Model, rescaling_factor=None, centers=None):
    """

    object between 0 and 1 due to stability problem of the float32.

    Args:
        geo_model (:class:`gempy.core.model.Model`)
        rescaling_factor(float): factor of the rescaling. Default to maximum distance in one the axis

    Returns:
        True
    """

    geo_model.rescaling.rescale_data(rescaling_factor, centers)
    return geo_model.additional_data.rescaling_data
# endregion


# region Interpolator functionality
@_setdoc([InterpolatorModel.__doc__,
         InterpolatorModel.set_all_shared_parameters.__doc__])
def set_interpolation_data(geo_model: Model, inplace=True, compile_theano: bool=True, output=None,
                           theano_optimizer=None, verbose:list = None):
    """

    Args:
        geo_model:
        inplace:
        compile_theano

    Returns:

    """

    if output is not None:
        geo_model.additional_data.options.df.at['values', 'output'] = output
    if theano_optimizer is not None:
        geo_model.additional_data.options.df.at['values', 'theano_optimizer'] = theano_optimizer
    if verbose is not None:
        geo_model.additional_data.options.df.at['values', 'verbosity'] = verbose

    # TODO add kwargs
    geo_model.rescaling.rescale_data()
    update_additional_data(geo_model)
    geo_model.surface_points.sort_table()
    geo_model.orientations.sort_table()
    geo_model.interpolator.create_theano_graph(geo_model.additional_data, inplace=True)
    geo_model.interpolator.set_all_shared_parameters(reset=True)

    if compile_theano is True:
        geo_model.interpolator.compile_th_fn(inplace=inplace)

    return geo_model.interpolator


def get_interpolator(model: Model):
    return model.interpolator


def get_th_fn(model: Model):
    """
    Get the compiled theano function

    Args:
        model (:class:`gempy.core.model.Model`)

    Returns:
        :class:`theano.compile.function_module.Function`: Compiled function if C or CUDA which computes the interpolation given the input data
            (XYZ of dips, dip, azimuth, polarity, XYZ ref surface_points, XYZ rest surface_points)
    """
    assert getattr(model.interpolator, 'theano_function', False) is not None, 'Theano has not been compiled yet'

    return model.interpolator.theano_function
# endregion


# region Additional data functionality
def update_additional_data(model: Model, update_structure=True, update_kriging=True):
    if update_structure is True:
        model.additional_data.update_structure()
    # if update_rescaling is True:
    #     model.additional_data.update_rescaling_data()
    if update_kriging is True:
        model.additional_data.update_default_kriging()

    return model.additional_data


def get_additional_data(model: Model):
    return model.get_additional_data()
# endregion


# region Computing the model
def compute_model(model: Model, output='geology', compute_mesh=True, reset_weights=False, reset_scalar=False,
                  reset_block=False, sort_surfaces=True, debug=False, set_solutions=True) -> Solution:
    """
    Computes the geological model and any extra output given in the additional data option.

    Args:
        model (:obj:`gempy.core.data.Model`)
        compute_mesh (bool): If true compute polydata

    Returns:
        gempy.core.data.Solution

    """

    # TODO: Assert frame by frame that all data is like is supposed. Otherwise,

    assert model.additional_data.structure_data.df.loc['values', 'len surfaces surface_points'].min() > 1, \
        'To compute the model is necessary at least 2 interface points per layer'

    if output == 'geology':
        assert model.interpolator.theano_function is not None, 'You need to compile the theano function first'
        i = model.interpolator.get_python_input_block(append_control=True, fault_drift=None)
        model.interpolator.reset_flow_control_initial_results(reset_weights, reset_scalar, reset_block)

        sol = model.interpolator.theano_function(*i)
    elif output == 'gravity':
        assert isinstance(model.interpolator_gravity, InterpolatorGravity), 'You need to set the gravity interpolator' \
                                                                            'first. See `Model.set_gravity_interpolator'
        i = model.interpolator_gravity.get_python_input_block(append_control=True, fault_drift=None)

        # TODO So far I reset all shared parameters to be sure. In the future this should be optimize as interpolator
        model.interpolator_gravity.set_theano_shared_tz_kernel()
        model.interpolator_gravity.set_all_shared_parameters(reset=True)
        sol = model.interpolator_gravity.theano_function(*i)

        set_solutions = False
    else:
        raise NotImplementedError('Only geology and gravity are implemented so far')

    if debug is True or set_solutions is False:
        return sol
    elif set_solutions is True:
        if model.grid.active_grids[0] is np.True_:
            model.solutions.set_solution_to_regular_grid(sol, compute_mesh=compute_mesh)
        # TODO @elisa elaborate this
        if model.grid.active_grids[2] is np.True_:
            l0, l1 = model.grid.get_grid_args('topography')
            model.solutions.geological_map = sol[0][:, l0: l1]
        if sort_surfaces:
            model.set_surface_order_from_solution()
        return model.solutions


def compute_model_at(new_grid: Union[ndarray], model: Model, **kwargs):
    """
    This function does the same as :func:`gempy.core.gempy_front.compute_model` plus the addion functionallity of
     passing a given array of points where evaluate the model instead of using the :class:`gempy.core.data.GridClass`.

    Args:
        model:
        new_grid (:class:`_np.array`): 2D array with XYZ (columns) coorinates
        kwargs: `compute_model` arguments

    Returns:
        gempy.core.data.Solution
    """
    # #TODO create backup of the mesh and a method to go back to it
    #     set_grid(model, Grid('custom_grid', custom_grid=new_grid))

    model.grid.deactivate_all_grids()
    model.set_custom_grid(new_grid)

    # Now we are good to compute the model again only in the new point
    sol = compute_model(model, set_solutions=False, **kwargs)
    return sol
# endregion


# region Solution
def get_meshes(model: Model):
    """
    gey vertices and simplices of the surface_points for its vtk visualization and further
    analysis

    Args:
       model (:class:`gempy.core.model.Model`)


    Returns:
        vertices, simpleces
    """
    return model.solutions.vertices, model.solutions.edges
# endregion


# region Model level functions
def get_data(model: Model, itype='data', numeric=False):
    """
    Method to return the data stored in :class:`DataFrame` within a :class:`gempy.interpolator.InterpolatorData`
    object.

    Args:
        model (:class:`gempy.core.model.Model`)
        itype(str {'all', 'surface_points', 'orientations', 'surfaces', 'series', 'faults', 'faults_relations',
        additional data}): input
            data type to be retrieved.
        numeric (bool): if True it only returns numberical properties. This may be useful due to memory issues
        verbosity (int): Number of properties shown

    Returns:
        pandas.core.frame.DataFrame

    """
    return model.get_data(itype=itype, numeric=numeric)


def get_surfaces(model_solution: Union[Model, Solution]):
    if isinstance(model_solution, Model):
        return model_solution.solutions.vertices, model_solution.solutions.edges
    elif isinstance(model_solution, Solution):
        return model_solution.vertices, model_solution.edges
    else:
        raise AttributeError


def create_data(extent: Union[list, ndarray], resolution: Union[list, ndarray] = (50, 50, 50),
                project_name: str = 'default_project', **kwargs) -> Model:
    """
    Create a :class:`gempy.core.model.Model` object and initialize some of the main functions such as:

    - Grid :class:`gempy.core.data.GridClass`: To regular grid.
    - read_csv: SurfacePoints and orientations: From csv files
    - set_values to default


    Args:
        extent (list or array):  [x_min, x_max, y_min, y_max, z_min, z_max]. Extent for the visualization of data
         and default of for the grid class.
        resolution (list or array): [nx, ny, nz]. Resolution for the visualization of data
         and default of for the grid class.
        project_name (str)

    Keyword Args:
        path_i: Path to the data bases of surface_points. Default os.getcwd(),
        path_o: Path to the data bases of orientations. Default os.getcwd()

    Returns:
        :class:`gempy.data_management.InputData`

    """
    warnings.warn("create_data will get deprecated in the next version of gempy. It still exist only to keep"
                  "the behaviour equal to older version. Use init_data.", FutureWarning)

    geo_model = create_model(project_name)
    return init_data(geo_model, extent=extent, resolution=resolution, project_name=project_name, **kwargs)


def init_data(geo_model: Model, extent: Union[list, ndarray] = None,
              resolution: Union[list, ndarray] = None,
              **kwargs) -> Model:
    """
    Create a :class:`gempy.core.model.Model` object and initialize some of the main functions such as:

    - Grid :class:`gempy.core.data.GridClass`: To regular grid.
    - read_csv: SurfacePoints and orientations: From csv files
    - set_values to default


    Args:
        extent (list or array):  [x_min, x_max, y_min, y_max, z_min, z_max]. Extent for the visualization of data
         and default of for the grid class.
        resolution (list or array): [nx, ny, nz]. Resolution for the visualization of data
         and default of for the grid class.
        project_name (str)

    Keyword Args:

        path_i: Path to the data bases of surface_points. Default os.getcwd(),
        path_o: Path to the data bases of orientations. Default os.getcwd()

    Returns:
        :class:`gempy.data_management.InputData`

    """

    if extent is None or resolution is None:
        warnings.warn('Regular grid won\'t be initialize, you will have to create a gridafterwards. See gempy.set_grid')
    else:
        geo_model.set_regular_grid(extent, resolution)

    read_csv(geo_model, **kwargs)

    return geo_model
# endregion


def activate_interactive_df(geo_model: Model, plot_object=None):
    """
    TODO evaluate the use of this functionality

    Notes: Since this feature is for advance levels we will keep only object oriented functionality. Should we
    add in the future,

    TODO: copy docstrings to QgridModelIntegration
    Args:
        geo_model:
        vtk_object:

    Returns:

    """
    try:
        from gempy.core.qgrid_integration import QgridModelIntegration
    except ImportError:
        raise ImportError('qgrid package is not installed. No interactive dataframes available.')
    geo_model.qi = QgridModelIntegration(geo_model, plot_object)
    return geo_model.qi
