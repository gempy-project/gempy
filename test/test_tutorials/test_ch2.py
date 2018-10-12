
# coding: utf-8

# # Chapter 2: A real example. Importing data and setting series
# 
# ## Data Management
# 
# 
# In this example we will show how we can import data from a csv and generate a model with several depositional series.

# In[1]:


# These two lines are necessary only if gempy is not installed
import sys, os
sys.path.append("../..")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Importing gempy
import gempy as gp

#from ..conftest import theano_f
input_path = os.path.dirname(__file__)+'/../../notebooks'

# Aux imports
import numpy as np
import pytest


def test_ch2(interpolator_islith_nofault):
    # Importing the data from csv files and settign extent and resolution
    geo_data = gp.create_data([696000,747000,6863000,6930000,-20000, 200], [50, 50, 50],
                             path_o=input_path+"/input_data/tut_SandStone/SandStone_Foliations.csv",
                             path_i=input_path+"/input_data/tut_SandStone/SandStone_Points.csv")


    gp.plotting.plot_data(geo_data, direction='z')

    # Assigning series to formations as well as their order (timewise)
    gp.set_series(geo_data, {"EarlyGranite_Series": 'EarlyGranite',
                             "BIF_Series":('SimpleMafic2', 'SimpleBIF'),
                                  "SimpleMafic_Series":'SimpleMafic1'},
                          order_series = ["EarlyGranite_Series",
                                          "BIF_Series",
                                          "SimpleMafic_Series"],
                          order_formations= ['EarlyGranite', 'SimpleMafic2', 'SimpleBIF', 'SimpleMafic1'],
                  verbose=1)

    geo_data.set_theano_function(interpolator_islith_nofault)
    sol = gp.compute_model(geo_data)

    import matplotlib.pyplot as plt

    gp.plotting.plot_section(geo_data, -2, plot_data=True, direction='z')
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)

    gp.plotting.plot_section(geo_data,25, plot_data=True, direction='x')
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)

    gp.plotting.plot_scalar_field(geo_data, 11, cmap='viridis', N=100)
    import matplotlib.pyplot as plt
    plt.colorbar(orientation='horizontal')

    vertices, simplices = gp.get_surfaces(geo_data)
    pyevtk = pytest.importorskip("pyevtk")
    gp.export_to_vtk(geo_data, path=os.path.dirname(__file__)+'/vtk_files/test2')

    # gp.plot_surfaces_3D_real_time(interp_data, vertices, simplices, alpha=1)

