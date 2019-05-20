"""
Module initialisation for GemPy
Created on 21/10/2016

@author: Miguel de la Varga
"""


import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from .core.gempy_api import *
import gempy.plot.plot as plot

assert sys.version_info[0] >= 3, "GemPy requires Python 3.X"  # sys.version_info[1] for minor e.g. 6
__version__ = '2.0b'

if __name__ == '__main__':
    pass
