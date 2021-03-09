"""
Module initialisation for GemPy
Created on 21/10/2016

@author: Miguel de la Varga
"""
import sys
import os

import warnings

try:
    import faulthandler
    faulthandler.enable()
except Exception as e:  # pragma: no cover
    warnings.warn('Unable to enable faulthandler:\n%s' % str(e))


PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


from gempy.gempy_api import *
from gempy.api_modules.getters import *
from gempy.api_modules.setters import *
from gempy.api_modules.io import *
from gempy.core.model import Project, ImplicitCoKriging, AdditionalData, Faults, Grid, \
    Orientations, ScalingSystem, Series, SurfacePoints, \
    Surfaces, Options, Structure, KrigingParameters

from gempy.core.solution import Solution
from gempy.addons.gempy_to_rexfile import geomodel_to_rex
import gempy.plot.plot_api as plot
from gempy.plot.plot_api import plot_2d, plot_3d

from gempy.plot import _plot as _plot

assert sys.version_info[0] >= 3, "GemPy requires Python 3.X"  # sys.version_info[1] for minor e.g. 6
__version__ = '2.2.8'

if __name__ == '__main__':
    pass
