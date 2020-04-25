""" Getters API

"""
from gempy import Project, Solution
from typing import Union
import warnings
from gempy.utils.meta import _setdoc_pro


@_setdoc_pro()
def get_data(model: Project, itype='data', numeric=False, verbosity=0):
    """Method to return the data stored in :class:`pandas.DataFrame` within a
    :class:`gempy.core.model.Project` data object.

    Args:
        model: [s_geo_model]
        itype: [s_itype]
        numeric (bool): if True it only returns numerical properties. This may be useful due to memory issues
        verbosity (int): Number of properties shown

    Returns:
        pandas.DataFrame: Data Object df.

    """
    return model.get_data(itype=itype, numeric=numeric, verbosity=verbosity)


def get_surfaces(model_solution: Union[Project, Solution]):
    """Get vertices and simplices of the surface_points for its vtk visualization and further analysis.

    Args:
       model_solution (:class:`Project` or :class:`Solution`)

    Returns:
        list[numpy.array]: vertices, simpleces
    """
    if isinstance(model_solution, Project):
        return model_solution.solutions.vertices, model_solution.solutions.edges
    elif isinstance(model_solution, Solution):
        return model_solution.vertices, model_solution.edges
    else:
        raise AttributeError


def get_interpolator(model: Project):
    return model._interpolator


def get_th_fn(model: Project):
    """Get the compiled theano function

    Args:
        model (:class:`model.Project`)

    Returns:
        :class:`theano.compile.function_module.Function`: Compiled function if C or CUDA which computes the interpolation given the input data
            (XYZ of dips, dip, azimuth, polarity, XYZ ref surface_points, XYZ rest surface_points)
    """
    assert getattr(model._interpolator, 'theano_function', False) is not None, 'Theano has not been compiled yet'

    return model._interpolator.theano_function


def get_additional_data(model: Project):
    warnings.warn('get_additional_data will be deprecrated in GemPy 2.3 Use '
                  'get(\'additional_data\') instead.', DeprecationWarning)
    return model.get_additional_data()
