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

"""
# Chapter 1: GemPy Basic

In this first example, we will show how to construct a first basic model and the main objects and functions. First we import gempy:
"""

# These two lines are necessary only if gempy is not installed
import sys, os
sys.path.append("../..")

# Importing gempy
import gempy as gp

# Embedding matplotlib figures into the notebooks
# %matplotlib inline

# Aux imports
import numpy as np

""
gp.create_data([0, 3000, 0, 20, 0, 2000], resolution=[3, 3, 3])

###############################################################################
# All data get stored in a python object InputData.  This object can be easily stored in a Python pickle. However, these files have the limitation that all dependecies must have the same versions as those when the pickle were created. For these reason to have more stable tutorials we will generate the InputData from raw data---i.e. csv files exported from Geomodeller.
#
# These csv files can be found in the input_data folder in the root folder of GemPy. These tables contains uniquely the XYZ (and poles, azimuth and polarity in the foliation case) as well as their respective formation name (but not necessary the formation order).
#

# Importing the data from csv files and settign extent and resolution
geo_data = gp.create_data([0,2000,0,2000,-2000,0],[ 10,10,10],
                         path_f = os.pardir+"/input_data/FabLessPoints_Foliations.csv",
                         path_i = os.pardir+"/input_data/FabLessPoints_Points.csv")

""
# Assigning series to formations as well as their order (timewise)
gp.set_series(geo_data, {"fault":'MainFault', 
                      "Rest": ('SecondaryReservoir','Seal', 'Reservoir', 'Overlying'), 
                               },
                       order_series = ["fault", 'Rest'],
                       order_formations=['MainFault', 
                                         'SecondaryReservoir', 'Seal','Reservoir', 'Overlying',
                                         ]) 
#geo_data = gp.select_series(geo_data, ['Rest'])


""
# %debug

""
import theano.tensor as T
import theano
interp_data = gp.InterpolatorData(geo_data, u_grade=[1],
                                  output='geology', dtype='float64', compile_theano=False)
print(interp_data)
the = interp_data.interpolator.tg

""
input_data_T = the.input_parameters_list()
input_data_T

""
geo_data.interfaces.head()

""
geo_data.formations

""
th_fn = theano.function(input_data_T,
                         the.compute_grad3(0),
#                         mode=theano.compile.MonitorMode(
#                         pre_func=inspect_inputs,
#                         post_func=inspect_outputs),
                                    on_unused_input='ignore',
                                    allow_input_downcast=False,
                                    profile=False)

""
i = interp_data.get_input_data()

""
th_fn(*i)


""
tyito

""
# import pandas as pn
# gp.set_orientations(geo_data,pn.DataFrame(geo_data.orientations.iloc[0,:]).T, append=True)
# geo_data.orientations.set_value(2, 'formation', 'Overlying')

""
# # Assigning series to formations as well as their order (timewise)
# gp.set_series(geo_data, {"fault":'MainFault', 
#                       "Rest": ('SecondaryReservoir','Seal', 'Reservoir'), 
#                                "Rist": ('Overlying')},
#                        order_series = ["fault", 'Rest', 'Rist'],
#                        order_formations=['MainFault', 
#                                          'SecondaryReservoir', 'Seal','Reservoir', 'Overlying',
#                                          ]) 

# geo_data =gp.select_series(geo_data,['Rest', 'Rist'])

""
geo_data.orientations

""
gp.get_sequential_pile(geo_data)

###############################################################################
# ## The ins and outs of Input data objects
#
# As we have seen objects DataManagement.InputData (usually called geo_data in the tutorials) aim to have all the original geological properties, measurements and geological relations stored. 
#
# Once we have the data ready to generate a model, we will need to create the next object type towards the final geological model:

geo_data.interfaces.drop(39, inplace=True)

""
import theano.tensor as T
import theano
interp_data = gp.InterpolatorData(geo_data, u_grade=[1, 1],
                                  output='geology', dtype='float64',
                                   verbose=['scalar_field_iter', 'block_series', 'yet_simulated'],
                                  compile_theano=True)


""
interp_data.interpolator.tg.len_series_i.get_value()

""


""
interp_data.interpolator.tg.n_formations_per_serie.get_value()

""
interp_data.interpolator.tg.n_formations_per_serie.set_value(np.array([0, 3, 4], dtype='int32'))

""
interp_data.interpolator.tg.npf.get_value()

""


""
geo_data.interfaces.shape

""
interp_data.interpolator.tg.npf.get_value()

""
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(-10,50)
sigm = (1. / (1 + np.exp(-1 * (x - 0)))) *5 - (-1. / (1 + np.exp(1 * (x - 30))) +1) *0
plt.plot(x, sigm)

""
interp_data.update_interpolator(geo_data)

""
interp_data.interpolator.tg.n_formation_float.get_value()

""
interp_data.interpolator.tg.n_formation_float.set_value(np.array([ 1.,  2.,  3.,  4.,  5., 6.], dtype='float32'))

""
sol = gp.compute_model(interp_data)

""
# %matplotlib notebook
gp.plot_section(geo_data,sol[0][0].astype(float), 30, plot_data = True, direction='y')

""
interp_data.interpolator.tg.n_formation_float.set_value(np.array([ 1.,  2.,  3.,  4.,  5., 6.], dtype='float32'))

""
interp_data.interpolator.tg.n_formations_per_serie.get_value()

""
interp_data.interpolator.tg.len_series_i.set_value(np.array([ 0,  4, 35], dtype='int32'))

""
interp_data.interpolator.tg.len_series_i.get_value()

""
interp_data.interpolator.tg.npf.get_value()

""
asa = interp_data.get_input_data()

""
interp_data.th_fn(*asa)

""
interp_data.interpolator.pandas_ref_layer_points

""

interp_data.geo_data_res.interfaces.drop(39, inplace=True)

""

