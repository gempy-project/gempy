""" Getters API

"""
from gempy.core.model import Model, Solution
from typing import Union
import warnings


def get_data(model: Model, itype='data', numeric=False):
    """
    Method to return the data stored in :class:`DataFrame` within a :class:`gempy.interpolator.InterpolatorData`
    object.

    Args:
        model (:class:`gempy.core.model.Model`)
        itype(str {'all', 'surface_points', 'orientations', 'surfaces', 'series', 'faults', 'faults_relations',
        additional data}): input data type to be retrieved.
        numeric (bool): if True it only returns numerical properties. This may be useful due to memory issues
        verbosity (int): Number of properties shown

    Returns:
        pandas.core.frame.DataFrame

    """
    return model.get_data(itype=itype, numeric=numeric)


def get_surfaces(model_solution: Union[Model, Solution]):
    """
    Get vertices and simplices of the surface_points for its vtk visualization and further
    analysis

    Args:
       model_solution (:class:`Model` or :class:`Solution)

    Returns:
        list[np.array]: vertices, simpleces
    """
    if isinstance(model_solution, Model):
        return model_solution.solutions.vertices, model_solution.solutions.edges
    elif isinstance(model_solution, Solution):
        return model_solution.vertices, model_solution.edges
    else:
        raise AttributeError


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


def get_additional_data(model: Model):
    warnings.warn('get_additional_data will be deprecrated in GemPy 2.3 Use '
                  'get(\'additional_data\') instead.', DeprecationWarning)
    return model.get_additional_data()
