"""
Module initialisation for GemPy
Created on 06.19.2023

@author: Miguel de la Varga
"""
import sys

import warnings


try:
    import faulthandler
    faulthandler.enable()
except Exception as e:  # pragma: no cover
    warnings.warn('Unable to enable faulthandler:\n%s' % str(e))

# ? This is not clear for what was used so for now I will comment it
# PACKAGE_PARENT = '..'
# SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
# sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# =================== Core ===================
# from .core.data.geo_model import GeoModel
# from .core.data.structural_frame import StructuralFrame
# from .core.data.structural_group import StructuralGroup
# from .core.data import AdditionalData, Options, KrigingParameters
# from .core.data_modules.stack import Faults, Series
# from .core.structure import Structure
# 
# from gempy.core.data.grid import Grid
# from .core.data.importer_helper import ImporterHelper
# from .core.surfaces import Surfaces
# from .core.data_modules.scaling_system import ScalingSystem
# from .core.data_modules.orientations import Orientations
# from .core.data_modules.surface_points import SurfacePoints
# from .core.solution import Solution

from .core import data
# =================== API ===================

from .API import *
from .API import implicit_functions
#from .gempy_api import *
# from .api_modules.getters import *
# from .api_modules.setters import *
# from .api_modules.io import *

# =================== Addons ===================
# from .addons.gempy_to_rexfile import geomodel_to_rex

# =================== Plotting ===================
# import gempy.plot.plot_api as plot
# from .plot.plot_api import plot_2d, plot_3d
# from .plot import _plot as _plot

# =================== Engine ===================
# * (NOTE: miguel (July 2023) For now I am not going to import here any of the engine modules)
# from gempy_engine.core.data.options import InterpolationOptions
# from gempy_engine.core.data.stack_relation_type import StackRelationType

# __all__ = ['GeoModel', 'Grid', 'StackRelationType', 'ImporterHelper', 'StructuralFrame', 'StructuralGroup']

# Assert at least pyton 3.10
assert sys.version_info[0] >= 3 and sys.version_info[1] >= 10, "GemPy requires Python 3.10 or higher"

__version__ = '2023.2.0'

if __name__ == '__main__':
    pass
