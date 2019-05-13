from os import path
import sys

# This is for sphenix to find the packages
sys.path.append(path.dirname( path.dirname( path.abspath(__file__) ) ) )

import numpy as _np
from numpy import ndarray
from pandas import DataFrame
from gempy.core.model import *
from typing import Union
from gempy.utils.meta import _setdoc

# This warning comes from numpy complaining about a theano optimization
warnings.filterwarnings("ignore",
                        message='.* a non-tuple sequence for multidimensional indexing is deprecated; use*.',
                        append=True)


@_setdoc(Series.__doc__)
def create_series(series_distribution=None, order=None):
    return Series(series_distribution=series_distribution, order=order)

@_setdoc(Surfaces.__doc__)
def create_formations(values_array=None, values_names=np.empty(0), formation_names=np.empty(0)):
    f = Surfaces(values_array=values_array, properties_names=values_names, formation_names=formation_names)
    return f

@_setdoc(Faults.__doc__)
def create_faults(series: Series, series_fault=None, rel_matrix=None):
    return Faults(series=series, series_fault=series_fault, rel_matrix=rel_matrix)

@_setdoc(Grid.__doc__)
def create_grid(grid_type: str, **kwargs):
    return Grid(grid_type=grid_type, **kwargs)