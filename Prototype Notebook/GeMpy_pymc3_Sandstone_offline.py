
# coding: utf-8

# # Example 2b: Implementing GeMpy into PyMC3
# 
# **Notebook an offline version of Example 2**
# 
# 
# 

# ### Generating data

# In[1]:

# Importing and data
import theano.tensor as T
import theano
import sys, os
sys.path.append("../GeMpy")

# Importing GeMpy modules
import GeMpy

# Reloading (only for development purposes)
import importlib
importlib.reload(GeMpy)

# Usuful packages
import numpy as np
import pandas as pn

import matplotlib.pyplot as plt

# This was to choose the gpu
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Default options of printin
np.set_printoptions(precision = 6, linewidth= 130, suppress =  True)

#%matplotlib inline
# %matplotlib inline



# Setting the extent
geo_data = GeMpy.import_data([0,10,0,10,0,10], [50,50,50])


# =========================
# DATA GENERATION IN PYTHON
# =========================
# Layers coordinates
layer_1 = np.array([[0.5,4,7], [2,4,6.5], [4,4,7], [5,4,6]])#-np.array([5,5,4]))/8+0.5
layer_2 = np.array([[3,4,5], [6,4,4],[8,4,4], [7,4,3], [1,4,6]])
layers = np.asarray([layer_1,layer_2])

# Foliations coordinates
dip_pos_1 = np.array([7,4,7])#- np.array([5,5,4]))/8+0.5
dip_pos_2 = np.array([2.,4,4])

# Dips
dip_angle_1 = float(15)
dip_angle_2 = float(340)
dips_angles = np.asarray([dip_angle_1, dip_angle_2], dtype="float64")

# Azimuths
azimuths = np.asarray([90,90], dtype="float64")

# Polarity
polarity = np.asarray([1,1], dtype="float64")

# Setting foliations and interfaces values
GeMpy.set_interfaces(geo_data, pn.DataFrame(
    data = {"X" :np.append(layer_1[:, 0],layer_2[:,0]),
            "Y" :np.append(layer_1[:, 1],layer_2[:,1]),
            "Z" :np.append(layer_1[:, 2],layer_2[:,2]),
            "formation" : np.append(
               np.tile("Layer 1", len(layer_1)), 
               np.tile("Layer 2", len(layer_2))),
            "labels" : [r'${\bf{x}}_{\alpha \, 0}^1$',
               r'${\bf{x}}_{\alpha \, 1}^1$',
               r'${\bf{x}}_{\alpha \, 2}^1$',
               r'${\bf{x}}_{\alpha \, 3}^1$',
               r'${\bf{x}}_{\alpha \, 0}^2$',
               r'${\bf{x}}_{\alpha \, 1}^2$',
               r'${\bf{x}}_{\alpha \, 2}^2$',
               r'${\bf{x}}_{\alpha \, 3}^2$',
               r'${\bf{x}}_{\alpha \, 4}^2$'] }))

GeMpy.set_foliations(geo_data,  pn.DataFrame(
    data = {"X" :np.append(dip_pos_1[0],dip_pos_2[0]),
            "Y" :np.append(dip_pos_1[ 1],dip_pos_2[1]),
            "Z" :np.append(dip_pos_1[ 2],dip_pos_2[2]),
            "azimuth" : azimuths,
            "dip" : dips_angles,
            "polarity" : polarity,
            "formation" : ["Layer 1", "Layer 2"],
            "labels" : [r'${\bf{x}}_{\beta \,{0}}$',
              r'${\bf{x}}_{\beta \,{1}}$'] })) 



layer_3 = np.array([[2,4,3], [8,4,2], [9,4,3]])
dip_pos_3 = np.array([1,4,1])
dip_angle_3 = float(80)
azimuth_3 = 90
polarity_3 = 1



GeMpy.set_interfaces(geo_data, pn.DataFrame(
    data = {"X" :layer_3[:, 0],
            "Y" :layer_3[:, 1],
            "Z" :layer_3[:, 2],
            "formation" : np.tile("Layer 3", len(layer_3)), 
            "labels" : [  r'${\bf{x}}_{\alpha \, 0}^3$',
                           r'${\bf{x}}_{\alpha \, 1}^3$',
                           r'${\bf{x}}_{\alpha \, 2}^3$'] }), append = True)
GeMpy.get_raw_data(geo_data,"interfaces")


GeMpy.set_foliations(geo_data, pn.DataFrame(data = {
                     "X" : dip_pos_3[0],
                     "Y" : dip_pos_3[1],
                     "Z" : dip_pos_3[2],
            
                     "azimuth" : azimuth_3,
                     "dip" : dip_angle_3,
                     "polarity" : polarity_3,
                     "formation" : [ 'Layer 3'],
                     "labels" : r'${\bf{x}}_{\beta \,{2}}$'}), append = True)


GeMpy.set_data_series(geo_data, {'younger': ('Layer 1', 'Layer 2'),
                      'older': 'Layer 3'}, order_series = ['younger', 'older'])




# In[2]:

# Select series to interpolate (if you do not want to interpolate all)
new_series = GeMpy.select_series(geo_data, ['younger'])
data_interp = GeMpy.set_interpolator(new_series, u_grade = 0)


# In[3]:

geo_data


# In[4]:

# This are the shared parameters and the compilation of the function. This will be hidden as well at some point
input_data_T = data_interp.interpolator.tg.input_parameters_list()
debugging = theano.function(input_data_T, data_interp.interpolator.tg.potential_field_at_all(), on_unused_input='ignore',
                            allow_input_downcast=True, profile=True)


# In[5]:

# This prepares the user data to the theano function
input_data_P = data_interp.interpolator.data_prep() 

# Solution of theano
sol = debugging(input_data_P[0], input_data_P[1], input_data_P[2], input_data_P[3],input_data_P[4], input_data_P[5])


# In[6]:

sol.shape


# In[7]:

# GeMpy.plot_potential_field(new_series, sol[:-14].reshape(50,50,50),13, plot_data = True)


# In[8]:

# If you change the values here. Here changes the plot as well
geo_data.foliations.set_value(0, 'dip', 40)


# In[9]:

# You need to set the interpolator again
new_series = GeMpy.select_series(geo_data, ['younger'])
data_interp = GeMpy.set_interpolator(new_series, u_grade = 0, verbose= ['cov_function'])


# In[10]:

# If you change it here is not necesary. Maybe some function in GeMpy with an attribute to choose would be good
data_interp.interpolator._data_scaled.foliations.set_value(0, 'dip', 40)
# In any case, data prep has to be called to convert the data to pure arrays. This function should be hidden I guess
input_data_P = data_interp.interpolator.data_prep()


# In[11]:

sol = debugging(input_data_P[0], input_data_P[1], input_data_P[2], input_data_P[3],input_data_P[4], input_data_P[5])


# In[12]:

# GeMpy.plot_potential_field(new_series, sol[:-14].reshape(50,50,50),13, plot_data = True)


# ## PyMC3

# In[13]:

data_interp = GeMpy.set_interpolator(geo_data, u_grade = 0)

# This are the shared parameters and the compilation of the function. This will be hidden as well at some point
input_data_T = data_interp.interpolator.tg.input_parameters_list()
# This prepares the user data to the theano function
input_data_P = data_interp.interpolator.data_prep() 


# In[14]:

# We create the op. Because is an op we cannot call it with python variables anymore. Thats why we have to make them shared
# Before
op2 = theano.OpFromGraph(input_data_T, [data_interp.interpolator.tg.whole_block_model()], on_unused_input='ignore')


# In[15]:

import pymc3 as pm
theano.config.compute_test_value = 'ignore'
model = pm.Model()
with model:
    # Stochastic value
    foliation = pm.Normal('foliation', 40, sd=10)
    
    # We convert a python variable to theano.shared
    dips = theano.shared(input_data_P[1])
    
    # We add the stochastic value to the correspondant array
    dips = T.set_subtensor(dips[0], foliation)

    geo_model = pm.Deterministic('GeMpy', op2(theano.shared(input_data_P[0]), dips, 
                                     theano.shared(input_data_P[2]), theano.shared(input_data_P[3]),
                                     theano.shared(input_data_P[4]), theano.shared(input_data_P[5])))

    # Set here the number of samples
    trace = pm.sample(100)


# In[16]:

trace.varnames, trace.get_values("GeMpy")


# In[17]:

# for i in trace.get_values('GeMpy')[:6]:
#     GeMpy.plot_section(new_series, 13, block = i, plot_data = False)
#     plt.show()


# In[18]:

## Added by FW:
## Posterior analysis
trace.get_values('GeMpy').shape


# In[19]:

# estimate unit probabilities

block = trace.get_values('GeMpy')[0]


# In[20]:

unit_ids = np.unique(block)


# In[21]:

block_probs = {}
for unit_id in unit_ids:
    block_probs[unit_id] = np.zeros_like(block)
    for block in trace.get_values('GeMpy'):
        tmp = np.ones_like(block)
        block_probs[unit_id] += tmp * (block==unit_id)
    
    # normalise
    print(np.min(block_probs[unit_id]), np.max(block_probs[unit_id]))
    block_probs[unit_id] = block_probs[unit_id]/ np.max(block_probs[unit_id])


# In[22]:

id0_grid = block_probs[1].reshape([50,50,50])


# In[23]:

#plt.imshow(id0_grid[:,25,:].transpose(), origin='bottom')
# plt.plot(id0)
# plt.colorbar()


# In[24]:

# using masked arrays
import numpy.ma as ma


# In[25]:

# Entropy calculation
h = np.zeros_like(block, dtype='float64')
for unit_id in unit_ids:
    block_masked = ma.masked_equal(block_probs[unit_id], 0)
    h -= ma.log2(block_masked) * block_masked


# In[26]:

# h = h.reshape([50,50,50])
# plt.imshow(h[:,0,:].transpose(), origin='bottom', cmap='viridis')
# plt.plot(id0)
# plt.colorbar()


# In[27]:

import pickle


# In[28]:

# save generated objects for further use
with open('data_offline.pkl', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(block_probs, f)


# In[41]:

# first, get values and convert to smaller type to save memory
# conversion to int8 reduces memory/ file size by almost a factor of 10!
values = trace.get_values('GeMpy') #[:1000]
values = values.astype('int8')

with open('trace_values_offline.pkl', 'wb') as f:
    pickle.dump(values, f)


# In[ ]:

np.save("h_offline.npy", h)


# In[47]:

# h = h.reshape([50,50,50])
# plt.imshow(h[5,:,:].transpose(), origin='bottom', cmap='viridis')
# plt.plot(id0)
# plt.colorbar()


# In[50]:

# for i in trace.get_values('GeMpy')[:6]:
#     GeMpy.plot_section(new_series, 5, direction='x', block = i, plot_data = False)
#     plt.show()


# In[51]:

# import ipyvolume.pylab as p3
# import ipyvolume.serialize
# ipyvolume.serialize.performance = 1 # 1 for binary, 0 for JSON
#p3 = ipyvolume.pylab.figure(width=200,height=600)


# In[ ]:




# In[ ]:



