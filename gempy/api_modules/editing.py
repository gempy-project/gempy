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


def add_surface_points(data: Union[Model, SurfacePoints]):
    pass


def add_orienations(data: Union[Model, Orientations]):
    pass


def add_series(data: Union[Model, Series]):
    pass


def add_formations(data: Union[Model, Surfaces]):
    pass

def del_surface_points(data: Union[Model, SurfacePoints]):
    pass


def del_orientations(data: Union[Model, Orientations]):
    pass


def del_series(data: Union[Model, Series]):
    pass


def del_formations(data: Union[Model, Surfaces]):
    pass


def modify_surface_points(data: Union[Model, SurfacePoints]):
    pass

def modify_orientations(data: Union[Model, Orientations]):
    pass


def modify_series(data: Union[Model, Series]):
    pass

def modify_formations(data: Union[Model, Surfaces]):
    pass

def modify_faults(data: Union[Model, Faults]):
    pass

def set_is_fault(data: Union[Model, Faults], idx):
    if isinstance(data, Model):
        data.set_is_fault(idx)
    elif isinstance(data, Faults):
        data.set_is_fault()

def modify_faults_network(data: Union[Model, Faults]):
    pass

def modify_options(data: Union[Model, Options]):
    pass

def modify_kriging_parameters(data: Union[Model, KrigingParameters]):
    pass

def modify_rescaling_parametesr(data: Union[Model, RescaledData]):
    pass
