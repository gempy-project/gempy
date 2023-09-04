# These two lines are necessary only if GemPy is not installed
import sys

sys.path.append("../../..")

# Importing GemPy
from gempy.core.data.grid_modules import CenteredGrid

# Importing auxiliary libraries
import numpy as np


def test_irregular_grid():
    g = CenteredGrid()
    g.set_centered_grid(np.array([0, 0, 0]), resolution=[5, 5, 5], radius = [100,100,100])
