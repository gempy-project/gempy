
import sys
from os import path
import theano.tensor as T
import theano
import numpy as np
import matplotlib.pyplot as plt
theano.config.compute_test_value = "ignore"
# This is for sphenix to find the packages
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )


sys.path.append("../gempy")
sys.path.append("../")
# Importing GeMpy modules
import gempy as gp
import pandas as pn


geo_data = gp.read_pickle('../Prototype Notebook/sandstone.pickle')

interp_data = gp.InterpolatorInput(geo_data, compile_theano=False)

import sys
import gempy as gp
from gempy.GeoPhysics import GeoPhysicsPreprocessing


inter_data = gp.InterpolatorInput(geo_data, compile_theano=False)
gpp = GeoPhysicsPreprocessing(inter_data,580,  [7.050000e+05,747000,6863000,6925000,-20000, 200],
                              res_grav = [70, 40],
                              n_cells = 100, mode='n_closest')
                              #res_grav = [125, 85],
                              #n_cells =1000)
print(gpp)
a = gpp.looping_z_decomp(70*40)
#b = gpp.z_decomposition()