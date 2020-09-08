from os import path
import sys

# This is for sphenix to find the packages
sys.path.append(path.dirname( path.dirname( path.abspath(__file__) ) ) )

from gempy.core.model import *
from typing import Union


def add_surface_points(data: Union[Project, SurfacePoints]):
    pass


def add_orienations(data: Union[Project, Orientations]):
    pass


def add_series(data: Union[Project, Series]):
    pass


def add_formations(data: Union[Project, Surfaces]):
    pass

def del_surface_points(data: Union[Project, SurfacePoints]):
    pass


def del_orientations(data: Union[Project, Orientations]):
    pass


def del_series(data: Union[Project, Series]):
    pass


def del_formations(data: Union[Project, Surfaces]):
    pass


def modify_surface_points(data: Union[Project, SurfacePoints]):
    pass

def modify_orientations(data: Union[Project, Orientations]):
    pass


def modify_series(data: Union[Project, Series]):
    pass

def modify_formations(data: Union[Project, Surfaces]):
    pass

def modify_faults(data: Union[Project, Faults]):
    pass

def set_is_fault(data: Union[Project, Faults], idx):
    if isinstance(data, Project):
        data.set_is_fault(idx)
    elif isinstance(data, Faults):
        data.set_is_fault()

def modify_faults_network(data: Union[Project, Faults]):
    pass

def modify_options(data: Union[Project, Options]):
    pass

def modify_kriging_parameters(data: Union[Project, KrigingParameters]):
    pass

def modify_rescaling_parametesr(data: Union[Project, ScalingSystem]):
    pass
