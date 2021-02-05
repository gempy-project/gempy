"""
This file is part of gempy.

gempy is free software. For license details, see the LICENSE.md file supplied
 with gempy

Created on 10/10 /2018

@author: Miguel de la Varga
"""

import numpy as np
import pandas as pn
from numpy import ndarray
from typing import Union
import warnings
from gempy.core.model import Project
from gempy.core.solution import Solution
from gempy.utils.meta import _setdoc, _setdoc_pro


# region get
def get():
    pass


# endregion

# region edit
def edit(model: Project,  method: str, data_object: str = None, **kwargs):
    """Function to edit any of the data_objects of gempy. Check
        https://www.gempy.org/documentation data for documentation.

    Args:
        model (Project):
        data_object (str): Object that you want to edit
        method (str): Method you want to use
        **kwargs:
    """
    if data_object is not None:
        data_object = getattr(model, '_' + data_object)
    else:
        data_object = model
    m = getattr(data_object, method)
    return m(**kwargs)


# endregion


# region mapping
@_setdoc_pro(Project.__doc__)
def map_series_to_surfaces(geo_model: Project, mapping_object: Union[dict, pn.Categorical] = None,
                           set_series=True, sort_geometric_data: bool = True, remove_unused_series=True):
    """Mapping which surfaces belongs to which geological feature

    Args:
        geo_model (:class:`gempy.core.model.Project`): [s0]
        mapping_object: [s_mapping_object]
        set_series (bool): If True set missing series from the mapping object
        sort_geometric_data (bool): If True sort the geometric_data according to the new
         order of the stack
        remove_unused_series(bool):

    Returns
        :class:`gempy.core.data.Surfaces`
    """
    warnings.warn('Series is going to get renamed to Stack. Please use'
                  '`map_stack_to_surfaces` instead.', DeprecationWarning)

    geo_model.map_stack_to_surfaces(mapping_object, set_series, sort_geometric_data, remove_unused_series)
    return geo_model._surfaces


@_setdoc_pro(Project.__doc__)
def map_stack_to_surfaces(geo_model: Project, mapping_object: Union[dict, pn.Categorical] = None,
                          set_features=True, sort_geometric_data: bool = True, remove_unused_series=True):
    """Mapping which surfaces belongs to which geological feature

    Args:
        geo_model (:class:`gempy.core.model.Project`): [s0]
        mapping_object: [s_mapping_object]
        set_features (bool): If True set missing features from the mapping object
        sort_geometric_data (bool): If True sort the geometric_data according to the new
         order of the stack
        remove_unused_series(bool):

    Returns
        :class:`gempy.core.data.Surfaces`
    """

    geo_model.map_stack_to_surfaces(mapping_object, set_features, sort_geometric_data, remove_unused_series)
    return geo_model._surfaces

# endregion


# region create
def create_model(project_name='default_project') -> Project:
    """Create a Project object.

    Args:
        project_name (str): Name of the project

    Returns:
        :class:`gempy.core.model.Project`

    See Also:
        :class:`gempy.core.model.Project`

    Notes:
        TODO: Adding saving address
    """
    return Project(project_name)


def create_data(project_name: str = 'default_project',
                extent: Union[list, ndarray] = None,
                resolution: Union[list, ndarray] = None,
                **kwargs) -> Project:
    """Create a :class:`gempy.core.model.Project` object and initialize some of
    the main functions such as:

    - Grid :class:`gempy.core.data.GridClass`: To regular grid.
    - read_csv: SurfacePoints and orientations: From csv files
    - set_values to default



    Args:
        project_name (str):
        extent (list or array): [x_min, x_max, y_min, y_max, z_min, z_max].
            Extent for the visualization of data and default of for the grid
            class.
        resolution (list or array): [nx, ny, nz]. Resolution for the
            visualization of data and default of for the grid class.
        **kwargs:

    Keyword:
        path_i: Path to the data bases of surface_points. Default os.getcwd(),
        path_o: Path to the data bases of orientations. Default os.getcwd()

    Returns:
        :class:`Project`
    """

    geo_model = create_model(project_name)
    return init_data(geo_model, extent=extent, resolution=resolution,
                     project_name=project_name, **kwargs)


# TODO We need to decide how to initialize a model. Having create_data and init
#  data  does not seem too robust
@_setdoc_pro([Project.__doc__])
def init_data(geo_model: Project, extent: Union[list, ndarray] = None,
              resolution: Union[list, ndarray] = None,
              **kwargs) -> Project:
    """Initialize some of the main functions such as:

     - Regular grid (:class:`gempy.core.data.Grid`).
     - read_csv: :class:`gempy.core.data_modules.geometric_data.SurfacePoints`
       and :class:`gempy.core.data_modules.geometric_data.Orientations` From csv files
     - set_values to default

    Args:
        geo_model (Project): [s0]
        extent: [s_extent]
        resolution: [s_resolution]

    Keyword Args:
        path_i: [s_path_i]
        path_o: [s_path_o]
        surface_points_df: [s_surface_points_df]
        orientations_df: [s_orientations_df]
    Returns:
        :class:`gempy.core.model.Project`
    """

    if extent is None or resolution is None:
        warnings.warn('Regular grid won\'t be initialize, you will have to create a gridafterwards. See gempy.set_grid')
    else:
        geo_model.set_regular_grid(extent, resolution)

    if 'path_i' in kwargs or 'path_o' in kwargs:
        read_csv(geo_model, **kwargs)

    if 'surface_points_df' in kwargs:
        geo_model.set_surface_points(kwargs['surface_points_df'], **kwargs)
        # if we set the surfaces names with surfaces they cannot be set again on orientations or pandas will complain.
        kwargs['update_surfaces'] = False
    if 'orientations_df' in kwargs:
        geo_model.set_orientations(kwargs['orientations_df'], **kwargs)

    return geo_model


# endregion


# If everything works out properly update. Should happen automatically. Therefore
# we will keep it just as a method
# region update
def update_additional_data(model: Project, update_structure=True, update_kriging=True):
    """
    Args:
        model (Project):
        update_structure:
        update_kriging:
    """
    warnings.warn('This function is going to be deprecated. Use Project.update_additional_data instead',
                  DeprecationWarning)
    return model.update_additional_data(update_structure, update_kriging)


# endregion


# region io
# @_setdoc([Project.read_data.__doc__])
def read_csv(geo_model: Project, path_i=None, path_o=None, **kwargs):
    """
    Args:
        geo_model (Project):
        path_i:
        path_o:
        **kwargs:
    """
    if path_i is not None or path_o is not None:
        try:
            geo_model.read_data(path_i, path_o, **kwargs)
        except KeyError as e:
            raise KeyError('Loading of CSV file failed. Check if you use commas '
                           'to separate your data.' + str(e))
    return True


# endregion


# region Computing the model
@_setdoc_pro([Project.__doc__, Solution.compute_marching_cubes_regular_grid.__doc__,
              Project.set_surface_order_from_solution.__doc__])
def compute_model(model: Project, output=None, at: np.ndarray = None, compute_mesh=True,
                  reset_weights=False, reset_scalar=False,
                  reset_block=False, sort_surfaces=True,
                  debug=False, set_solutions=True,
                  **kwargs) -> Solution:
    """Computes the geological model and any extra output given in the
    additional data option.

    Args:
        model (Project): [s0]
        output (str {'geology', 'gravity'}): Compute the lithologies or gravity
        at (np.ndarray):
        compute_mesh (bool): if True compute marching cubes: [s1]
        reset_weights (bool): Not Implemented
        reset_scalar (bool): Not Implemented
        reset_block (bool): Not Implemented
        sort_surfaces (bool): if True call
            Project.set_surface_order_from_solution: [s2]
        debug (bool): if True, the computed interpolation are not stored in any
            object but instead returned
        set_solutions (bool): Default True. If True set the results into the
            :class:`Solutions` linked object.
        **kwargs:

    Keyword Args:
        compute_mesh_options (dict): options for the marching cube function. 1)
            rescale: True

    Returns:
        :class:`Solutions`
    """

    # Check config
    # ------------
    _check_valid_model_input(model)

    if output is not None:
        warnings.warn('Argument output has no effect anymore and will be deprecated in GemPy 2.2.'
                      'Set the output only in gempy.set_interpolator.', DeprecationWarning, )
    if at is not None:
        model._grid.deactivate_all_grids()
        model.set_custom_grid(at)

    # ------------

    i = model._interpolator.get_python_input_block(append_control=True, fault_drift=None)
    model._interpolator.reset_flow_control_initial_results(reset_weights, reset_scalar, reset_block)

    sol = model._interpolator.theano_function(*i)

    if debug is True or set_solutions is False:
        return sol

    elif set_solutions is True:

        model.solutions.set_solutions(
            sol,
            compute_mesh,
            sort_surfaces,
            **kwargs)

        if sort_surfaces:
            model.set_surface_order_from_solution()
        return model.solutions


def _check_valid_model_input(model):
    if model._interpolator.theano_function is None:
        raise ValueError('You need to compile graph before. '
                         'See `gempy.set_interpolator`.')
    if model._additional_data.structure_data.df.loc[
        'values', 'len surfaces surface_points'].min() < 1:
        raise ValueError('To compute the model is necessary at least 2 interface '
                         'points per layer')
    if len(model._interpolator.len_series_i) != len(
        model._interpolator.len_series_o):
        raise ValueError('Every Series/Fault need at least 1 orientation and 2 '
                         'surfaces points.')
    is_basement_in_sp = model._surfaces.basement.isin(
        model._surface_points.df['surface']).any()
    is_basement_in_ori = model._surfaces.basement.isin(
        model._orientations.df['surface']).any()

    if is_basement_in_ori or is_basement_in_sp:
        raise ValueError('There are surface points or orientations assigned to the '
                         'Surface defined as basement (bottom of the stack). The '
                         'basement surface only refers to the volume below the last '
                         'surface and is not supposed to be interpolated. '
                         'Add a "basement" surface (`model.add_surface("basement")`)'
                         ' or delete the discordant surface points or orientations')

    last_feature_is_fault = model._stack.df['BottomRelation'].last == 'fault'
    if last_feature_is_fault:
        raise ValueError('Last feature of the stack should not be a fault. '
                         'Reorder the stack using geo_model.reorder_features(List)')


def compute_model_at(new_grid: Union[ndarray], model: Project, **kwargs):
    """This function creates a new custom grid and deactivate all the other
    grids and compute the model there:

    This function does the same as  plus the addition functionality of
        :func:`compute_model`
        passing a given array of points where evaluate the model instead of
        using the :class:`gempy.core.data.GridClass`.

    Args:
        new_grid:
        model (Project):
        kwargs: :func:`compute_model` arguments

    Returns:
        :class:`Solution`
    """
    # #TODO create backup of the mesh and a method to go back to it
    #     set_grid(model, Grid('custom_grid', custom_grid=new_grid))
    warnings.warn('compute_model_at will be deprecated.'
                  'Use argument `at` in compute_model instead', DeprecationWarning)
    model._grid.deactivate_all_grids()
    model.set_custom_grid(new_grid)

    # Now we are good to compute the model again only in the new point
    sol = compute_model(model, set_solutions=False, **kwargs)
    return sol

# endregion


# region activate
@_setdoc_pro([Project.__doc__], )
def activate_interactive_df(geo_model: Project, plot_object=None):
    """Experimental: Activate the use of the QgridProjectIntegration: TODO
    evaluate the use of this functionality

    Notes: Since this feature is for advance levels we will keep only object
    oriented functionality. Should we add in the future,

    TODO: copy docstrings to QgridModelIntegration :param geo_model: [s0]
    :param plot_object: GemPy plot object (so far only vtk is available)

    Args:
        geo_model (Project):
        plot_object:

    Returns:
        :class:`QgridModelIntegration`
    """
    try:
        from gempy.core.qgrid_integration import QgridModelIntegration
    except ImportError:
        raise ImportError('qgrid package is not installed. No interactive dataframes available.')
    geo_model.qi = QgridModelIntegration(geo_model, plot_object)
    return geo_model.qi
# endregion

