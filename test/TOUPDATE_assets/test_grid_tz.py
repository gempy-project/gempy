# These two lines are necessary only if GemPy is not installed
import sys, os
sys.path.append("../..")

# Importing GemPy
import gempy as gp
from gempy.assets.geophysics import GeophysicsPreprocessing
from gempy.core.grid_modules.grid_types import CenteredGrid

# Importing auxiliary libraries
import numpy as np
import pandas as pn
import matplotlib.pyplot as plt
import pytest


def test_irregular_grid():
    g = CenteredGrid()
    g.set_centered_grid([5, 5, 5], [100,100,100])
