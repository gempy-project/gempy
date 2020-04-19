# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: sphinx
#       format_version: '1.1'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import copy
import matplotlib.pyplot as plt
from matplotlib import cm

import matplotlib.gridspec as gridspect

# Create cmap
from matplotlib.colors import ListedColormap
import matplotlib.colors as colors
import matplotlib.cm as cmx

import numpy as np
import pandas as pn
import scipy.stats as stats
import seaborn as sns

""

 # These two lines are necessary only if GemPy is not installed
import sys, os
sys.path.append("../..")

# Importing GemPy
import gempy as gp

""
from gempy.assets import decision_making as dm
from importlib import reload
reload(dm)

""
g = dm.plot_multiple_loss()
a = plt.gcf()

""
a

""

a = plt.gcf()

""
a.savefig('loss.pdf')

""
a

""
b = plt.legend()

""
b.set_frame_on
