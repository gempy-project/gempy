
# coding: utf-8

# # Chapter 5: Computing forward gravity. (Under development)
# 
# GemPy also brings a module to compute the forward gravity response. The idea is to be able to use gravity as a likelihood to validate the geological models within the Bayesian inference. In this chapter we will see how we can compute the gravity response of the sandstone model of chapter 2.

# In[1]:


# These two lines are necessary only if gempy is not installed
import sys, os
sys.path.append("../..")

# Importing gempy
import gempy as gp


# Aux imports
import numpy as np
#from ..conftest import theano_f_grav, theano_f
input_path = os.path.dirname(__file__)+'/../../notebooks'

def test_ch5(theano_f_grav, theano_f):
    # Importing the data from csv files and settign extent and resolution
    geo_data = gp.create_data([696000,747000,6863000,6950000,-20000, 200],[50, 50, 50],
                             path_o = input_path+"/input_data/tut_SandStone/SandStone_Foliations.csv",
                             path_i = input_path+"/input_data/tut_SandStone/SandStone_Points.csv")


    # Assigning series to formations as well as their order (timewise)
    gp.set_series(geo_data, {"EarlyGranite_Series": 'EarlyGranite',
                                  "BIF_Series":('SimpleMafic2', 'SimpleBIF'),
                                  "SimpleMafic_Series":'SimpleMafic1'},
                          order_series = ["EarlyGranite_Series",
                                          "BIF_Series",
                                          "SimpleMafic_Series"],
                          order_formations= ['EarlyGranite', 'SimpleMafic2',
                                             'SimpleBIF', 'SimpleMafic1'],
                  verbose=1)



    gp.plot_data(geo_data)


    #interp_data = gp.InterpolatorData(geo_data, compile_theano=True)
    interp_data = theano_f
    interp_data.update_interpolator(geo_data)

    lith_block, fault_block = gp.compute_model(interp_data)

    import matplotlib.pyplot as plt
    gp.plot_section(geo_data, lith_block[0], 10, plot_data=True, direction='y')
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)

    from matplotlib.patches import Rectangle

    currentAxis = plt.gca()

    currentAxis.add_patch(Rectangle((7.050000e+05, 6863000),
                                    747000 - 7.050000e+05,
                                    6925000 - 6863000,
                          alpha=0.3, fill='none', color ='green' ))

    ver_s, sim_s = gp.get_surfaces(interp_data, lith_block[1],
                                   None,
                                   original_scale=True)

   # gp.plot_surfaces_3D_real_time(interp_data, ver_s, sim_s)

    # Importing the data from csv files and settign extent and resolution
    geo_data_extended = gp.create_data([696000-10000,
                                        747000 + 20600,
                                        6863000 - 20600,6950000 + 20600,
                                        -20000, 600],
                                       [50, 50, 50],
                                   path_o=input_path + "/input_data/tut_SandStone/SandStone_Foliations.csv",
                                   path_i=input_path + "/input_data/tut_SandStone/SandStone_Points.csv")


    # Assigning series to formations as well as their order (timewise)
    gp.set_series(geo_data_extended, {"EarlyGranite_Series": 'EarlyGranite',
                                  "BIF_Series":('SimpleMafic2', 'SimpleBIF'),
                                  "SimpleMafic_Series":'SimpleMafic1'},
                          order_series = ["EarlyGranite_Series",
                                          "BIF_Series",
                                          "SimpleMafic_Series"],
                          order_formations= ['EarlyGranite', 'SimpleMafic2',
                                             'SimpleBIF', 'SimpleMafic1'],
                  verbose=1)

   # interp_data_extended = gp.InterpolatorData(geo_data_extended, output='geology',
    #                                           compile_theano=True)
    interp_data_extended = interp_data
    interp_data_extended.update_interpolator(geo_data_extended)

    geo_data_extended.set_formations(formation_values=[2.61,2.92,3.1,2.92,2.61],
                            formation_order=['EarlyGranite', 'SimpleMafic2',
                                             'SimpleBIF', 'SimpleMafic1',
                                             'basement'])

    lith_ext, fautl = gp.compute_model(interp_data_extended)

    import matplotlib.pyplot as plt

    gp.plot_section(geo_data_extended, lith_ext[0], -1, plot_data=True, direction='z')
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)

    from matplotlib.patches import Rectangle

    currentAxis = plt.gca()

    currentAxis.add_patch(Rectangle((7.050000e+05, 6863000),  747000 - 7.050000e+05,
                                     6925000 - 6863000,
                          alpha=0.3, fill='none', color ='green' ))



    interp_data_grav = theano_f_grav
    interp_data_grav.update_interpolator(geo_data_extended)

    gp.set_geophysics_obj(interp_data_grav,  [7.050000e+05,747000,6863000,6925000,-20000, 200],
                                                 [10, 10],)

    gp.precomputations_gravity(interp_data_grav, 10)

    lith, fault, grav = gp.compute_model(interp_data_grav, 'gravity')

    import matplotlib.pyplot as plt

    plt.imshow(grav.reshape(10, 10), cmap='viridis', origin='lower',
               extent=[7.050000e+05,747000,6863000,6950000] )
    plt.colorbar()
