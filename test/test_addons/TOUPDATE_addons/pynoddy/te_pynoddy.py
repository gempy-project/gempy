import sys, os
# Path to development gempy
sys.path.append('../../..')

# Path to development pynoddy
sys.path.append('../../../../pynoddy')

import matplotlib.pyplot as plt
import numpy as np
# adjust some settings for matplotlib
from matplotlib import rcParams
# print rcParams
rcParams['font.size'] = 15
# determine path of repository to set paths corretly below
repo_path = os.path.realpath('../../..')
import gempy as gp
import pynoddy

import pynoddy.history

his = pynoddy.history.NoddyHistory(url = \
            "http://tectonique.net/asg/ch2/ch2_2/ch2_2_1/his/normal.his")

history_name = "fold_thrust.his"
his.write_history(history_name)

import pynoddy.experiment

ex = pynoddy.experiment.Experiment(history_name)


x = [250,750,500]
y = [0,100,200]
layers = [1]

for l in layers:
    i_df = ex.export_interfaces_gempy(x, y, layer=l, group_id="l"+str(l)+"_a")
   # gp.set_surface_points(geo_model, i_df, append=True)