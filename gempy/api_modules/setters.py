""" setters API

"""
from gempy import get_data
from gempy.utils.meta import _setdoc, _setdoc_pro
from gempy.utils import docstring as ds
from gempy.core.model import Model, InterpolatorModel
from typing import Union
import warnings
import numpy as np
# This warning comes from numpy complaining about a theano optimization
warnings.filterwarnings("ignore",
                        message='.* a non-tuple sequence for multidimensional '
                                'indexing is deprecated; use*.',
                        append=True)


@_setdoc([InterpolatorModel.__doc__])
@_setdoc_pro([Model.__doc__, ds.compile_theano, ds.theano_optimizer])
def set_interpolator(geo_model: Model, output: list = None, compile_theano: bool = True,
                     theano_optimizer=None, verbose: list = None, grid=None, type_=None,
                     update_structure=True, update_kriging=True,
                     **kwargs):
    """
    Method to create a graph and compile the theano code to compute the interpolation.

    Args:
        geo_model (:class:`gempy.core.model.Project`): [s0]
        output (list[str:{geo, grav}]): type of interpolation.
        compile_theano (bool): [s1]
        theano_optimizer (str {'fast_run', 'fast_compile'}): [s2]
        verbose:
        update_kriging (bool): reset kriging values to its default.
        update_structure (bool): sync Structure instance before setting theano graph.

    Keyword Args:
        -  pos_density (Optional[int]): Only necessary when type='grav'. Location on the Surfaces().df
         where density is located (starting on id being 0).
        - Vs
        - pos_magnetics

    Returns:

    """
    # output = list(output)
    if output is None:
        output = ['geology']

    if type(output) is not list:
        raise TypeError('Output must be a list.')

    # TODO Geology is necessary for everthing?
    if 'gravity' in output and 'geology' not in output:
        output.append('geology')

    if 'magnetics' in output and 'geology' not in output:
        output.append('geology')

    if type_ is not None:
        warnings.warn('type warn is going to be deprecated. Use output insted', FutureWarning)
        output = type_

    if theano_optimizer is not None:
        geo_model._additional_data.options.df.at['values', 'theano_optimizer'] = theano_optimizer
    if verbose is not None:
        geo_model._additional_data.options.df.at['values', 'verbosity'] = verbose
    if 'dtype' in kwargs:
        geo_model._additional_data.options.df.at['values', 'dtype'] = kwargs['dtype']
    if 'device' in kwargs:
        geo_model._additional_data.options.df.at['values', 'device'] = kwargs['device']

    # TODO add kwargs
    geo_model._rescaling.rescale_data()
    geo_model.update_additional_data(update_structure=update_structure, update_kriging=update_kriging)
    geo_model.update_to_interpolator()
    geo_model._surface_points.sort_table()
    geo_model._orientations.sort_table()

    geo_model._interpolator.create_theano_graph(geo_model._additional_data, inplace=True,
                                                output=output, **kwargs)

    if 'gravity' in output:
        pos_density = kwargs.get('pos_density', 1)
        tz = kwargs.get('tz', 'auto')
        if geo_model._grid.centered_grid is not None:
            geo_model._interpolator.set_theano_shared_gravity(tz, pos_density)

    if 'magnetics' in output:
        pos_magnetics = kwargs.get('pos_magnetics', 1)
        Vs = kwargs.get('Vs', 'auto')
        incl = kwargs.get('incl')
        decl = kwargs.get('decl')
        B_ext = kwargs.get('B_ext', 52819.8506939139e-9)
        if geo_model._grid.centered_grid is not None:
            geo_model._interpolator.set_theano_shared_magnetics(Vs, pos_magnetics, incl, decl, B_ext)

    if 'topology' in output:
        # This id is necessary for topology
        id_list = geo_model._surfaces.df.groupby('isFault').cumcount() + 1
        geo_model.add_surface_values(id_list, 'topology_id')
        geo_model._interpolator.set_theano_shared_topology()

        # TODO it is missing to pass to theano the position of topology_id

    if compile_theano is True:
        geo_model._interpolator.set_all_shared_parameters(reset_ctrl=True)

        geo_model._interpolator.compile_th_fn_geo(inplace=True, grid=grid)
    else:
        if grid == 'shared':
            geo_model._interpolator.set_theano_shared_grid(grid)

    print('Kriging values: \n', geo_model._additional_data.kriging_data)
    return geo_model._interpolator


@_setdoc_pro([Model.__doc__])
def set_geometric_data(geo_model: Model, surface_points_df=None,
                       orientations_df=None, **kwargs):
    """ Function to set directly pandas.Dataframes to the gempy geometric data objects

    Args:
        geo_model: [s0]
        surface_points_df:  A pn.Dataframe object with X, Y, Z, and surface columns
        orientations_df: A pn.Dataframe object with X, Y, Z, surface columns and pole or orientation columns
        **kwargs:

    Returns:
        Modified df
    """

    r_ = None

    if surface_points_df is not None:
        geo_model.set_surface_points(surface_points_df, **kwargs)
        r_ = 'surface_points'

    elif orientations_df is not None:
        geo_model.set_orientations(orientations_df, **kwargs)
        r_ = 'data' if r_ == 'surface_points' else 'orientations'

    else:
        raise AttributeError('You need to pass at least one dataframe')

    return get_data(geo_model, itype=r_)


def set_orientation_from_surface_points(geo_model, indices_array):
    """
    Create and set orientations from at least 3 points of the :attr:`gempy.data_management.InputData.surface_points`
     Dataframe

    Args:
        geo_model (:class:`Model`):
        indices_array (array-like): 1D or 2D array with the pandas indices of the
          :attr:`surface_points`. If 2D every row of the 2D matrix will be used to create an
          orientation


    Returns:
        :attr:`orientations`: Already updated inplace
    """

    if np.ndim(indices_array) is 1:
        indices = indices_array
        form = geo_model._surface_points.df['surface'].loc[indices].unique()
        assert form.shape[0] is 1, 'The interface points must belong to the same surface'
        form = form[0]

        ori_parameters = geo_model._orientations.create_orientation_from_surface_points(
            geo_model._surface_points, indices)
        geo_model.add_orientations(X=ori_parameters[0], Y=ori_parameters[1], Z=ori_parameters[2],
                                   orientation=ori_parameters[3:6], pole_vector=ori_parameters[6:9],
                                   surface=form)
    elif np.ndim(indices_array) is 2:
        for indices in indices_array:
            form = geo_model._surface_points.df['surface'].loc[indices].unique()
            assert form.shape[0] is 1, 'The interface points must belong to the same surface'
            form = form[0]
            ori_parameters = geo_model._orientations.create_orientation_from_surface_points(
                geo_model._surface_points, indices)
            geo_model.add_orientations(X=ori_parameters[0], Y=ori_parameters[1], Z=ori_parameters[2],
                                       orientation=ori_parameters[3:6], pole_vector=ori_parameters[6:9],
                                       surface=form)

    return geo_model._orientations

def select_nearest_surfaces_points(geo_model, surface_points, searchcrit):
    """
    Find the neighbour points of the same surface
    by given radius (radius-search) or fix number (knn).
    Parameters
    ----------
    geo_model : geo_model
        GemPy-model.
    surface_points: Pandas-dataframe
        Contains the dataframe of the (point-)data from the GemPy-model.
    searchcrit : int or float
        if is int: uses knn-search.
        if is float: uses radius-search.
    """

    from sklearn.neighbors import NearestNeighbors

    # extract surface names
    surfaces = np.unique(surface_points['surface'])
    neighbours = []
    # for each surface
    if isinstance(searchcrit, int):  # in case knn-search
        searchcrit = searchcrit + 1  # because the point itself is also found
        for s in range(surfaces.size):
            # extract point-ids
            i_surfaces = surface_points['surface'] == surfaces[s]
            # extract point coordinates
            p_surfaces = surface_points[i_surfaces][['X', 'Y', 'Z']]
            # create search-tree
            Tree = NearestNeighbors(n_neighbors=searchcrit)
            # add data to tree
            Tree.fit(p_surfaces)
            # find neighbours
            neighbours_surfaces = Tree.kneighbors(p_surfaces, n_neighbors=searchcrit,
                                            return_distance=False)
            # add neighbours with initial index to total list
            for n in neighbours_surfaces:
                neighbours.append(p_surfaces.index[n])
    else:  # in case radius-search
        for s in range(surfaces.size):
            # extract point-ids
            i_surfaces = surface_points['surface'] == surfaces[s]
            # extract point coordinates
            p_surfaces = surface_points[i_surfaces][['X', 'Y', 'Z']]
            # create search-tree
            Tree = NearestNeighbors(radius=searchcrit)
            # add data to tree
            Tree.fit(p_surfaces)
            # find neighbours (attention: relativ index!)
            neighbours_surfaces = Tree.radius_neighbors(p_surfaces,
                                                  radius=searchcrit,
                                                  return_distance=False)
            # add neighbours with initial index to total list
            for n in neighbours_surfaces:
                neighbours.append(p_surfaces.index[n])
    return neighbours


def set_orientation_from_neighbours(geo_model, neighbours):

    """
    Calculates the orientation of one point with its neighbour points
    of the same surface.
    Parameters
    ----------
    geo_model : geo_model
        GemPy-model.
    neighbours : Int64Index
        point-neighbours-id, first id is the point itself.
    """

    from sklearn.preprocessing import normalize

    # compute normal vector for the point
    if neighbours.size > 2:
        # extract point coordinates
        coo = geo_model._surface_points.df.loc[neighbours][['X', 'Y', 'Z']]
        # calculates covariance matrix
        cov = np.cov(coo.T)
        # calculate normalized normal vector
        normvec = normalize(np.cross(cov[0].T, cov[1].T).reshape(1, -1))[0]
        # check orientation of normal vector (has to be oriented to sky)
        if normvec[2] < 0:
            normvec = normvec*(-1)
        # append to the GemPy-model
        geo_model.add_orientations(geo_model._surface_points.df['X'][neighbours[0]],
                                   geo_model._surface_points.df['Y'][neighbours[0]],
                                   geo_model._surface_points.df['Z'][neighbours[0]],
                                   geo_model._surface_points.df['surface'][neighbours[0]],
                                   normvec.tolist())
    # if computation is impossible set normal vector to default orientation
    else:
        print("orientation calculation of point" + str(neighbours[0]) + "is impossible")
        print("-> default vector is set [0,0,1]")
        geo_model.add_orientations(geo_model._surface_points.df['X'][neighbours[0]],
                                   geo_model._surface_points.df['Y'][neighbours[0]],
                                   geo_model._surface_points.df['Z'][neighbours[0]],
                                   geo_model._surface_points.df['surface'][neighbours[0]],
                                   orientation=[0, 0, 1])
    return geo_model._orientations


def set_orientation_from_neighbours_all(geo_model, neighbours):

    """
    Calculates the orientations for all points with given neighbours.
    Parameters
    ----------
    geo_model : geo_model
        GemPy-model.
    neighbours : list of Int64Index
        point-neighbours-IDs, the first id is the id of the point
        for which the orientation is calculated.
    """

    # compute normal vector for the points
    for n in neighbours:
        set_orientation_from_neighbours(geo_model, n)

    return geo_model._orientations
