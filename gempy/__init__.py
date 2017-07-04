'''Module initialisation for GeMpy
Created on 21/10/2016

@author: Miguel de la Varga
'''
#from . import qgrid
#import DataManagement
#import theanograf

import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from .GemPy_f import *

if __name__ == '__main__':
    pass
