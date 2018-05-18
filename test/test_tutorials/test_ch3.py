
# coding: utf-8

# # Chapter 3: Stochastic Simulations in pymc2
# 
# This tutorial will show you how to use GemPy for stochastic simulation of geological models. We will address two approaches for this: (i) Monte Carlo forward simulation, treating input data as uncertain parameter distributions; (ii) Bayesian inference, where we extent the approach with the use of likelihood functions to constrain the stochastic modeling results with additional data.

# 
# ## Preparation
# 
# Import GemPy, matplotlib for plotting, numpy and pandas for data handling.

# In[2]:


import sys, os
sys.path.append("../..")

# import gempy
import gempy as gp

# inline figures in jupyter notebooks
import matplotlib.pyplot as plt

import numpy as np
import pandas as pn
import theano

#from ..conftest import theano_f
input_path = os.path.dirname(__file__)+'/../../notebooks'
import pytest
pymc = pytest.importorskip("pymc")

def test_ch3_a(theano_f):

    # set cube size and model extent
    cs = 50
    extent = (3000, 200, 2000)  # (x, y, z)
    res = (120, 4, 80)


    # initialize geo_data object
    geo_data = gp.create_data([0, extent[0],
                               0, extent[1],
                               0, extent[2]],
                              resolution=[res[0],  # number of voxels
                                          res[1],
                                          res[2]])

    geo_data.set_interfaces(pn.read_csv(input_path+"/input_data/tut_chapter3/tutorial_ch3_interfaces",
                                        index_col="Unnamed: 0"), append=True)
    geo_data.set_orientations(pn.read_csv(input_path+"/input_data/tut_chapter3/tutorial_ch3_foliations",
                                        index_col="Unnamed: 0"))

    # let's have a look at the upper five interface data entries in the dataframe
    gp.get_data(geo_data, 'interfaces', verbosity=1).head()

    # Original pile
    gp.get_sequential_pile(geo_data)

    # Ordered pile
    gp.set_order_formations(geo_data, ['Layer 2', 'Layer 3', 'Layer 4','Layer 5'])
    gp.get_sequential_pile(geo_data)

    # and at all of the foliation data
    gp.get_data(geo_data, 'orientations', verbosity=0)

    gp.plot_data(geo_data, direction="y")
    plt.xlim(0,3000)
    plt.ylim(0,2000);

    gp.data_to_pickle(geo_data, os.path.dirname(__file__)+"/ch3-pymc2_tutorial_geo_data")

    #interp_data = gp.InterpolatorData(geo_data, u_grade=[1], compile_theano=True)
    interp_data = theano_f
    interp_data.update_interpolator(geo_data)

    # Afterwards we can compute the geological model
    lith_block, fault_block = gp.compute_model(interp_data)


    # And plot a section:
    gp.plot_section(geo_data, lith_block[0], 2, plot_data = True)

    import pymc

    # Checkpoint in case you did not execute the cells above
    geo_data = gp.read_pickle(os.path.dirname(__file__)+"/ch3-pymc2_tutorial_geo_data.pickle")

    gp.get_data(geo_data, 'orientations', verbosity=1).head()

    # So let's assume the vertical location of our layer interfaces is uncertain, and we want to represent this
    #  uncertainty by using a normal distribution. To define a normal distribution, we need a mean and a measure
    #  of deviation (e.g. standard deviation). For convenience the input data is already grouped by a "group_id" value,
    # which allows us to collectively modify data that belongs together. In this example we want to treat the vertical
    # position of each layer interface, on each side of the anticline, as uncertain. Therefore, we want to perturbate
    # the respective three points on each side of the anticline collectively.

    # These are our unique group id's, the number representing the layer, and a/b the side of the anticline.

    group_ids = geo_data.interfaces["group_id"].dropna().unique()
    print(group_ids)


    # As a reminder, GemPy stores data in two main objects, an InputData object (called geo_data in the tutorials) and
    # a InpterpolatorInput object (interp_data) in tutorials. geo_data contains the original data while interp_data the
    # data prepared (and compiled) to compute the 3D model.
    #
    # Since we do not want to compile our code at every new stochastic realization, from here on we will need to work
    # with thte interp_data. And remember that to improve float32 to stability we need to work with rescaled data
    # (between 0 and 1). Therefore all the stochastic data needs to be rescaled accordingly. The object interp_data
    #  contains a property with the rescale factor (see below. As default depends on the model extent), or it is
    # possible to add the stochastic data to the pandas dataframe of the geo_data---when the InterpolatorInput object
    # is created the rescaling happens under the hood.

    interface_Z_modifier = []

    # We rescale the standard deviation
    std = 20./interp_data.rescaling_factor

    # loop over the unique group id's and create a pymc.Normal distribution for each
    for gID in group_ids:
        stoch = pymc.Normal(gID+'_stoch', 0, 1./std**2)
        interface_Z_modifier.append(stoch)

    # Let's have a look at one:


    # sample from a distribtion
    samples = [interface_Z_modifier[3].rand() for i in range(10000)]
    # plot histogram
    plt.hist(samples, bins=24, normed=True);
    plt.xlabel("Z modifier")
    plt.vlines(0, 0, 0.01)
    plt.ylabel("n");


    #  Now we need to somehow sample from these distribution and put them into GemPy

    # ## Input data handling
    #
    # First we need to write a function which modifies the input data for each iteration of the stochastic simulation.
    #  As this process is highly dependant on the simulation (e.g. what input parameters you want modified in which way),
    #  this process generally can't be automated.
    #
    # The idea is to change the column Z (in this case) of the rescaled dataframes in our interp_data object (which can
    #  be found in interp_data.geo_data_res). First we simply create the pandas Dataframes we are interested on:


    import copy
    # First we extract from our original intep_data object the numerical data that is necessary for the interpolation.
    # geo_data_stoch is a pandas Dataframe

    # This is the inital model so it has to be outside the stochastic frame
    geo_data_stoch_init = copy.deepcopy(interp_data.geo_data_res)

    gp.get_data(geo_data_stoch_init, numeric=True).head()

    @pymc.deterministic(trace=True)
    def input_data(value = 0,
                   interface_Z_modifier = interface_Z_modifier,
                   geo_data_stoch_init = geo_data_stoch_init,
                   verbose=0):
        # First we extract from our original intep_data object the numerical data that is necessary for the interpolation.
        # geo_data_stoch is a pandas Dataframe

        geo_data_stoch = gp.get_data(geo_data_stoch_init, numeric=True)
        # Now we loop each id which share the same uncertainty variable. In this case, each layer.
        for e, gID in enumerate(group_ids):
            # First we obtain a boolean array with trues where the id coincide
            sel = gp.get_data(interp_data.geo_data_res, verbosity=2)['group_id'] == gID

            # We add to the original Z value (its mean) the stochastic bit in the correspondant groups id
            geo_data_stoch.loc[sel, 'Z']  += np.array(interface_Z_modifier[e])

        if verbose > 0:
            print(geo_data_stoch)

        # then return the input data to be input into the modeling function. Due to the way pymc2 stores the traces
        # We need to save the data as numpy arrays
        return [geo_data_stoch.xs('interfaces')[["X", "Y", "Z"]].values, geo_data_stoch.xs('orientations').values]


    # ## Modeling function

    @pymc.deterministic(trace=False)
    def gempy_model(value=0,
                    input_data=input_data, verbose=True):

        # modify input data values accordingly
        interp_data.geo_data_res.interfaces[["X", "Y", "Z"]] = input_data[0]

        # Gx, Gy, Gz are just used for visualization. The theano function gets azimuth dip and polarity!!!
        interp_data.geo_data_res.orientations[["G_x", "G_y", "G_z", "X", "Y", "Z",  'dip', 'azimuth', 'polarity']] = input_data[1]

        try:
            # try to compute model
            lb, fb = gp.compute_model(interp_data)
            if True:
                gp.plot_section(interp_data.geo_data_res, lb[0], 0, plot_data=True)

            return lb, fb

        except np.linalg.linalg.LinAlgError as err:
            # if it fails (e.g. some input data combinations could lead to
            # a singular matrix and thus break the chain) return an empty model
            # with same dimensions (just zeros)
            if verbose:
                print("Exception occured.")
            return np.zeros_like(lith_block), np.zeros_like(fault_block)

    # We then create a pymc model with the two deterministic functions (*input_data* and *gempy_model*), as well as all
    #  the prior parameter distributions stored in the list *interface_Z_modifier*:

    params = [input_data, gempy_model, *interface_Z_modifier]
    model = pymc.Model(params)

    # Then we set the number of iterations:

    # Then we create an MCMC chain (in pymc an MCMC chain without a likelihood function is essentially a Monte Carlo
    # forward simulation) and specify an hdf5 database to store the results in

    RUN = pymc.MCMC(model, db="hdf5", dbname=os.path.dirname(__file__)+"/ch3-pymc2.hdf5")

    # and we are finally able to run the simulation:

    RUN.sample(iter=100, verbose=0)


def test_ch3_b(theano_f):

    geo_data = gp.read_pickle(os.path.dirname(__file__)+"/ch3-pymc2_tutorial_geo_data.pickle")

    # Check the stratigraphic pile for correctness:


    gp.get_sequential_pile(geo_data)


    # Then we can then compile the GemPy modeling function:


    #interp_data = gp.InterpolatorData(geo_data, u_grade=[1])
    interp_data = theano_f
    interp_data.update_interpolator(geo_data)

    # Now we can reproduce the original model:



    lith_block, fault_block = gp.compute_model(interp_data)
    gp.plot_section(geo_data, lith_block[0], 0)


    # But of course we want to look at the perturbation results. We have a class for that:

    import gempy.posterior_analysis

    dbname = os.path.dirname(__file__)+"/ch3-pymc2.hdf5"
    post = gempy.posterior_analysis.Posterior(dbname)



    post.change_input_data(interp_data, 80)


    lith_block, fault_block = gp.compute_model(interp_data)
    gp.plot_section(interp_data.geo_data_res, lith_block[0], 2, plot_data=True)



    post.change_input_data(interp_data, 15)
    lith_block, fault_block = gp.compute_model(interp_data)
    gp.plot_section(interp_data.geo_data_res, lith_block[0], 2, plot_data=True)

    post.change_input_data(interp_data, 95)
    lith_block, fault_block = gp.compute_model(interp_data)
    gp.plot_section(geo_data, lith_block[0], 2)

    ver, sim = gp.get_surfaces(interp_data, lith_block[1], None, original_scale= True)
    #gp.plotting.plot_surfaces_3D_real_time(geo_data, interp_data,
     #                                      ver, sim, posterior=post, alpha=1)



