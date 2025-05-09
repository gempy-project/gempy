"""
Module initialisation for GemPy
Created on 06.19.2023

@author: Miguel de la Varga
"""
import sys

# =================== CORE ===================
from .core import data
from .core.color_generator import ColorsGenerator

# =================== API ===================

from .API import *
from .API import implicit_functions

# * Assert at least pyton 3.10
assert sys.version_info[0] >= 3 and sys.version_info[1] >= 10, "GemPy requires Python 3.10 or higher"

from ._version import __version__

if __name__ == '__main__':
    pass
