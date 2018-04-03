
Chapter 3: Stochastic Simulations in pymc2
==========================================

This tutorial will show you how to use GemPy for stochastic simulation
of geological models. We will address two approaches for this: (i) Monte
Carlo forward simulation, treating input data as uncertain parameter
distributions; (ii) Bayesian inference, where we extent the approach
with the use of likelihood functions to constrain the stochastic
modeling results with additional data.

Preparation
-----------

Import GemPy, matplotlib for plotting, numpy and pandas for data
handling.

.. code:: ipython3

    import sys, os
    sys.path.append("../..")
    
    # import gempy
    import gempy as gp
    
    # inline figures in jupyter notebooks
    import matplotlib.pyplot as plt
    %matplotlib inline
    
    import numpy as np
    import pandas as pn
    import theano


.. parsed-literal::

    WARNING (theano.tensor.blas): Using NumPy C-API based implementation for BLAS functions.


Initialize an example model
---------------------------

First we define the cube size and model extent of our model and
initialize the GemPy data object:

.. code:: ipython3

    # set cube size and model extent
    cs = 50
    extent = (3000, 200, 2000)  # (x, y, z)
    res = (120, 4, 80)

.. code:: ipython3

    # initialize geo_data object
    geo_data = gp.create_data([0, extent[0],
                               0, extent[1], 
                               0, extent[2]],
                              resolution=[res[0],  # number of voxels
                                          res[1], 
                                          res[2]])

Then we use pandas to load the example data stored as csv files:

.. code:: ipython3

    geo_data.set_interfaces(pn.read_csv("../input_data/tutorial_ch3_interfaces",
                                        index_col="Unnamed: 0"))
    geo_data.set_orientations(pn.read_csv("../input_data/tutorial_ch3_foliations",
                                        index_col="Unnamed: 0"))

.. code:: ipython3

    # let's have a look at the upper five interface data entries in the dataframe
    gp.get_data(geo_data, 'interfaces', verbosity=1).head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>X</th>
          <th>X_std</th>
          <th>Y</th>
          <th>Y_std</th>
          <th>Z</th>
          <th>Z_std</th>
          <th>annotations</th>
          <th>formation</th>
          <th>formation_number</th>
          <th>group_id</th>
          <th>isFault</th>
          <th>order_series</th>
          <th>series</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>250</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>996</td>
          <td>0</td>
          <td>${\bf{x}}_{\alpha \,{\bf{1}},0}$</td>
          <td>Layer 2</td>
          <td>1</td>
          <td>l2_a</td>
          <td>False</td>
          <td>1</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2500</td>
          <td>0</td>
          <td>200</td>
          <td>0</td>
          <td>1149</td>
          <td>0</td>
          <td>${\bf{x}}_{\alpha \,{\bf{1}},1}$</td>
          <td>Layer 2</td>
          <td>1</td>
          <td>l2_b</td>
          <td>False</td>
          <td>1</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>2</th>
          <td>2250</td>
          <td>0</td>
          <td>100</td>
          <td>0</td>
          <td>1298</td>
          <td>0</td>
          <td>${\bf{x}}_{\alpha \,{\bf{1}},2}$</td>
          <td>Layer 2</td>
          <td>1</td>
          <td>l2_b</td>
          <td>False</td>
          <td>1</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>3</th>
          <td>2750</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>995</td>
          <td>0</td>
          <td>${\bf{x}}_{\alpha \,{\bf{1}},3}$</td>
          <td>Layer 2</td>
          <td>1</td>
          <td>l2_b</td>
          <td>False</td>
          <td>1</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>4</th>
          <td>500</td>
          <td>0</td>
          <td>200</td>
          <td>0</td>
          <td>1149</td>
          <td>0</td>
          <td>${\bf{x}}_{\alpha \,{\bf{1}},4}$</td>
          <td>Layer 2</td>
          <td>1</td>
          <td>l2_a</td>
          <td>False</td>
          <td>1</td>
          <td>Default serie</td>
        </tr>
      </tbody>
    </table>
    </div>



We can visualize (and modify making use of an interactive backend) the
stratigraphic pile. Since the formations are arbitary we can set them
by:

.. code:: ipython3

    # Original pile
    gp.get_sequential_pile(geo_data)




.. parsed-literal::

    <gempy.sequential_pile.StratigraphicPile at 0x7fda8cfcb160>




.. image:: ch3_files/ch3_10_1.png


.. code:: ipython3

    # Ordered pile
    gp.set_order_formations(geo_data, ['Layer 2', 'Layer 3', 'Layer 4','Layer 5'])
    gp.get_sequential_pile(geo_data)




.. parsed-literal::

    <gempy.sequential_pile.StratigraphicPile at 0x7fd9fe0bdac8>




.. image:: ch3_files/ch3_11_1.png


.. code:: ipython3

    # and at all of the foliation data
    gp.get_data(geo_data, 'orientations', verbosity=0)




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>X</th>
          <th>Y</th>
          <th>Z</th>
          <th>G_x</th>
          <th>G_y</th>
          <th>G_z</th>
          <th>dip</th>
          <th>azimuth</th>
          <th>polarity</th>
          <th>formation</th>
          <th>series</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>500</td>
          <td>100</td>
          <td>1148</td>
          <td>-0.516992</td>
          <td>-0.00855937</td>
          <td>0.855947</td>
          <td>31.1355</td>
          <td>269.051</td>
          <td>1</td>
          <td>Layer 2</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2500</td>
          <td>100</td>
          <td>1147.33</td>
          <td>0.516122</td>
          <td>-0.0142732</td>
          <td>0.856397</td>
          <td>31.0857</td>
          <td>91.5841</td>
          <td>1</td>
          <td>Layer 2</td>
          <td>Default serie</td>
        </tr>
      </tbody>
    </table>
    </div>



Visualize the input data
------------------------

Now let's have a look at the data in the xz-plane:

.. code:: ipython3

    gp.plot_data(geo_data, direction="y")
    plt.xlim(0,3000)
    plt.ylim(0,2000);



.. image:: ch3_files/ch3_14_0.png


At this point we should store the input data object as a pickle, for
future reference:

.. code:: ipython3

    gp.data_to_pickle(geo_data, "./pickles/ch3-pymc2_tutorial_geo_data")

Compile the interpolator function
---------------------------------

Now that we have some input data, the next step is to compile the
interpolator function of GemPy with the imported model setup and data:

.. code:: ipython3

    interp_data = gp.InterpolatorData(geo_data, u_grade=[1], compile_theano=True)


.. parsed-literal::

    Compiling theano function...
    Compilation Done!
    Level of Optimization:  fast_compile
    Device:  cpu
    Precision:  float32
    Number of faults:  0


Afterwards we can compute the geological model:

.. code:: ipython3

    lith_block, fault_block = gp.compute_model(interp_data)

And plot a section:

.. code:: ipython3

    gp.plot_section(geo_data, lith_block[0], 2, plot_data = True)



.. image:: ch3_files/ch3_22_0.png


Setting up the pymc-Functions
-----------------------------

pymc has two distinct types of objects: **deterministic** and
**stochastic** objects. As the `pymc
documentation <https://pymc-devs.github.io/pymc/modelbuilding.html>`__
puts it: "A *Stochastic* object represents a variable whose value is not
completely determined by its parents, and a *Deterministic* object
represents a variable that is entirely determined by its parents."
Stochastic objects can essentially be seen as *parameter distributions*
or *likelihood functions*, while Deterministic objects can be seen as
function that take a specific input and return a specific (determined)
output for this input. An example for the latter would be the modeling
function of GemPy, which takes a specific set of input parameters and
always creates the same model from those parameters.

.. code:: ipython3

    import pymc

Setting up the parameter distributions
--------------------------------------

For conducting a stochastic simulation of the geological model, we need
to consider our input data (dips and layer interfaces) as uncertain -
i.e. as distributions.

.. code:: ipython3

    # Checkpoint in case you did not execute the cells above
    geo_data = gp.read_pickle("./pickles/ch3-pymc2_tutorial_geo_data.pickle")

.. code:: ipython3

    gp.get_data(geo_data, 'orientations', verbosity=1).head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>G_x</th>
          <th>G_y</th>
          <th>G_z</th>
          <th>X</th>
          <th>X_std</th>
          <th>Y</th>
          <th>Y_std</th>
          <th>Z</th>
          <th>Z_std</th>
          <th>annotations</th>
          <th>...</th>
          <th>azimuth_std</th>
          <th>dip</th>
          <th>dip_std</th>
          <th>formation</th>
          <th>formation_number</th>
          <th>group_id</th>
          <th>isFault</th>
          <th>order_series</th>
          <th>polarity</th>
          <th>series</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>-0.516992</td>
          <td>-0.00855937</td>
          <td>0.855947</td>
          <td>500</td>
          <td>NaN</td>
          <td>100</td>
          <td>NaN</td>
          <td>1148</td>
          <td>NaN</td>
          <td>${\bf{x}}_{\beta \,{\bf{1}},0}$</td>
          <td>...</td>
          <td>NaN</td>
          <td>31.1355</td>
          <td>NaN</td>
          <td>Layer 2</td>
          <td>1</td>
          <td>l2_a</td>
          <td>False</td>
          <td>1</td>
          <td>1</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.516122</td>
          <td>-0.0142732</td>
          <td>0.856397</td>
          <td>2500</td>
          <td>NaN</td>
          <td>100</td>
          <td>NaN</td>
          <td>1147.33</td>
          <td>NaN</td>
          <td>${\bf{x}}_{\beta \,{\bf{1}},1}$</td>
          <td>...</td>
          <td>NaN</td>
          <td>31.0857</td>
          <td>NaN</td>
          <td>Layer 2</td>
          <td>1</td>
          <td>l2_b</td>
          <td>False</td>
          <td>1</td>
          <td>1</td>
          <td>Default serie</td>
        </tr>
      </tbody>
    </table>
    <p>2 rows × 21 columns</p>
    </div>



So let's assume the vertical location of our layer interfaces is
uncertain, and we want to represent this uncertainty by using a normal
distribution. To define a normal distribution, we need a mean and a
measure of deviation (e.g. standard deviation). For convenience the
input data is already grouped by a "group\_id" value, which allows us to
collectively modify data that belongs together. In this example we want
to treat the vertical position of each layer interface, on each side of
the anticline, as uncertain. Therefore, we want to perturbate the
respective three points on each side of the anticline collectively.

These are our unique group id's, the number representing the layer, and
a/b the side of the anticline.

.. code:: ipython3

    group_ids = np.unique(geo_data.interfaces["group_id"])
    print(group_ids)


.. parsed-literal::

    ['l2_a' 'l2_b' 'l3_a' 'l3_b' 'l4_a' 'l4_b' 'l5_a' 'l5_b']


As a reminder, GemPy stores data in two main objects, an InputData
object (called geo\_data in the tutorials) and a InpterpolatorInput
object (interp\_data) in tutorials. geo\_data contains the original data
while interp\_data the data prepared (and compiled) to compute the 3D
model.

Since we do not want to compile our code at every new stochastic
realization, from here on we will need to work with thte interp\_data.
And remember that to improve float32 to stability we need to work with
rescaled data (between 0 and 1). Therefore all the stochastic data needs
to be rescaled accordingly. The object interp\_data contains a property
with the rescale factor (see below. As default depends on the model
extent), or it is possible to add the stochastic data to the pandas
dataframe of the geo\_data---when the InterpolatorInput object is
created the rescaling happens under the hood.

.. code:: ipython3

    interface_Z_modifier = []
    
    # We rescale the standard deviation
    std = 20./interp_data.rescaling_factor
    
    # loop over the unique group id's and create a pymc.Normal distribution for each
    for gID in group_ids:
        stoch = pymc.Normal(gID+'_stoch', 0, 1./std**2)
        interface_Z_modifier.append(stoch)

our list of parameter distribution:

.. code:: ipython3

    interface_Z_modifier




.. parsed-literal::

    [<pymc.distributions.new_dist_class.<locals>.new_class 'l2_a_stoch' at 0x7fd9e04e31d0>,
     <pymc.distributions.new_dist_class.<locals>.new_class 'l2_b_stoch' at 0x7fd9fd2e9240>,
     <pymc.distributions.new_dist_class.<locals>.new_class 'l3_a_stoch' at 0x7fd9fdfdff28>,
     <pymc.distributions.new_dist_class.<locals>.new_class 'l3_b_stoch' at 0x7fd9e04e3208>,
     <pymc.distributions.new_dist_class.<locals>.new_class 'l4_a_stoch' at 0x7fd9e04e36a0>,
     <pymc.distributions.new_dist_class.<locals>.new_class 'l4_b_stoch' at 0x7fd9e04e3978>,
     <pymc.distributions.new_dist_class.<locals>.new_class 'l5_a_stoch' at 0x7fd9e04e3748>,
     <pymc.distributions.new_dist_class.<locals>.new_class 'l5_b_stoch' at 0x7fd9e04e37b8>]



Let's have a look at one:

.. code:: ipython3

    # sample from a distribtion
    samples = [interface_Z_modifier[3].rand() for i in range(10000)]
    # plot histogram
    plt.hist(samples, bins=24, normed=True);
    plt.xlabel("Z modifier")
    plt.vlines(0, 0, 0.01)
    plt.ylabel("n");



.. image:: ch3_files/ch3_36_0.png


Now we need to somehow sample from these distribution and put them into
GemPy

Input data handling
-------------------

First we need to write a function which modifies the input data for each
iteration of the stochastic simulation. As this process is highly
dependant on the simulation (e.g. what input parameters you want
modified in which way), this process generally can't be automated.

The idea is to change the column Z (in this case) of the rescaled
dataframes in our interp\_data object (which can be found in
interp\_data.geo\_data\_res). First we simply create the pandas
Dataframes we are interested on:

.. code:: ipython3

    import copy
    # First we extract from our original intep_data object the numerical data that is necessary for the interpolation.
    # geo_data_stoch is a pandas Dataframe
    
    # This is the inital model so it has to be outside the stochastic frame
    geo_data_stoch_init = copy.deepcopy(interp_data.geo_data_res)

.. code:: ipython3

    gp.get_data(geo_data_stoch_init, numeric=True).head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th></th>
          <th>X</th>
          <th>Y</th>
          <th>Z</th>
          <th>G_x</th>
          <th>G_y</th>
          <th>G_z</th>
          <th>dip</th>
          <th>azimuth</th>
          <th>polarity</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th rowspan="5" valign="top">interfaces</th>
          <th>0</th>
          <td>0.2501</td>
          <td>0.4801</td>
          <td>0.5299</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.7001</td>
          <td>0.5201</td>
          <td>0.5605</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.6501</td>
          <td>0.5001</td>
          <td>0.5903</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.7501</td>
          <td>0.4801</td>
          <td>0.5297</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>4</th>
          <td>0.3001</td>
          <td>0.5201</td>
          <td>0.5605</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    @pymc.deterministic(trace=True)
    def input_data(value = 0, 
                   interface_Z_modifier = interface_Z_modifier,
                   geo_data_stoch_init = geo_data_stoch_init,
                   verbose=0):
        # First we extract from our original intep_data object the numerical data that is necessary for the interpolation.
        # geo_data_stoch is a pandas Dataframe
     #   geo_data_stoch = gp.get_data(interp_data_original.geo_data_res, numeric=True)
    
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

Modeling function
-----------------

Second, we need a function that takes the modified input data output by
the above function, and created our geological model from it. Although,
we could store the model itself it tends to be a too large file once we
make thousands of iterations. For this reason is preferible to keep the
input data and the geological model split and only to store the input
data ( which we will use to reconstruct each geological model since that
operation is deterministic).

.. code:: ipython3

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
               # gp.plot_data(interp_data.geo_data_res, direction='y')
    
            return lb, fb
        
        except np.linalg.linalg.LinAlgError as err:
            # if it fails (e.g. some input data combinations could lead to 
            # a singular matrix and thus break the chain) return an empty model
            # with same dimensions (just zeros)
            if verbose:
                print("Exception occured.")
            return np.zeros_like(lith_block), np.zeros_like(fault_block)



.. image:: ch3_files/ch3_43_0.png


.. code:: ipython3

    interp_data.geo_data_res.orientations




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>G_x</th>
          <th>G_y</th>
          <th>G_z</th>
          <th>X</th>
          <th>X_std</th>
          <th>Y</th>
          <th>Y_std</th>
          <th>Z</th>
          <th>Z_std</th>
          <th>annotations</th>
          <th>...</th>
          <th>azimuth_std</th>
          <th>dip</th>
          <th>dip_std</th>
          <th>formation</th>
          <th>formation_number</th>
          <th>group_id</th>
          <th>isFault</th>
          <th>order_series</th>
          <th>polarity</th>
          <th>series</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0.3001</td>
          <td>0.5001</td>
          <td>0.555743</td>
          <td>-0.516992</td>
          <td>NaN</td>
          <td>-0.008559</td>
          <td>NaN</td>
          <td>0.855947</td>
          <td>NaN</td>
          <td>${\bf{x}}_{\beta \,{\bf{1}},0}$</td>
          <td>...</td>
          <td>NaN</td>
          <td>31.135451</td>
          <td>NaN</td>
          <td>Layer 2</td>
          <td>1</td>
          <td>l2_a</td>
          <td>False</td>
          <td>1</td>
          <td>1.0</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.7001</td>
          <td>0.5001</td>
          <td>0.559481</td>
          <td>0.516122</td>
          <td>NaN</td>
          <td>-0.014273</td>
          <td>NaN</td>
          <td>0.856397</td>
          <td>NaN</td>
          <td>${\bf{x}}_{\beta \,{\bf{1}},1}$</td>
          <td>...</td>
          <td>NaN</td>
          <td>31.085652</td>
          <td>NaN</td>
          <td>Layer 2</td>
          <td>1</td>
          <td>l2_b</td>
          <td>False</td>
          <td>1</td>
          <td>1.0</td>
          <td>Default serie</td>
        </tr>
      </tbody>
    </table>
    <p>2 rows × 21 columns</p>
    </div>



We then create a pymc model with the two deterministic functions
(*input\_data* and *gempy\_model*), as well as all the prior parameter
distributions stored in the list *interface\_Z\_modifier*:

.. code:: ipython3

    params = [input_data, gempy_model, *interface_Z_modifier] 
    model = pymc.Model(params)

Then we set the number of iterations:

.. code:: ipython3

    iterations = 100

Then we create an MCMC chain (in pymc an MCMC chain without a likelihood
function is essentially a Monte Carlo forward simulation) and specify an
hdf5 database to store the results in:

.. code:: ipython3

    RUN = pymc.MCMC(model, db="hdf5", dbname="./pymc-db/ch3-pymc2_tutorial-db")

and we are finally able to run the simulation:

.. code:: ipython3

    RUN.sample(iter=100, verbose=0)


.. parsed-literal::

     [-----------------100%-----------------] 100 of 100 complete in 6.3 sec

Analyzing the results
---------------------

When we want to analyze the results, we first have to load the stored
geo\_data object:

(this part is only necessary if the notebook above is not executed)

.. code:: ipython3

    geo_data = gp.read_pickle("./pymc-db/ch3-pymc2_tutorial_geo_data.pickle")

Check the stratigraphic pile for correctness:

.. code:: ipython3

    gp.get_sequential_pile(geo_data)




.. parsed-literal::

    <gempy.sequential_pile.StratigraphicPile at 0x7fd9e04e3a90>




.. image:: ch3_files/ch3_56_1.png


Then we can then compile the GemPy modeling function:

.. code:: ipython3

    interp_data = gp.InterpolatorData(geo_data, u_grade=[1])

Now we can reproduce the original model:

.. code:: ipython3

    lith_block, fault_block = gp.compute_model(interp_data)
    gp.plot_section(geo_data, lith_block[0], 0)


.. parsed-literal::

    Compiling theano function...
    Compilation Done!
    Level of Optimization:  fast_compile
    Device:  cpu
    Precision:  float32
    Number of faults:  0



.. image:: ch3_files/ch3_60_1.png


But of course we want to look at the perturbation results. We have a
class for that:

(in the mid term the most important methods of this class will be moved
to the gempy main framework---i.e. gp)

.. code:: ipython3

    import gempy.posterior_analysis
    import importlib
    importlib.reload(gempy.posterior_analysis)




.. parsed-literal::

    <module 'gempy.posterior_analysis' from '../../gempy/posterior_analysis.py'>



Which allows us to load the stored pymc2 database

.. code:: ipython3

    dbname = "ch3-pymc2_tutorial-db"
    post = gempy.posterior_analysis.Posterior(dbname)

Alright, it tells us that we did not tally any GemPy models (we set the
trace flag for the gempy\_model function to False!). But we can just
replace the input data with the ones stored at each iteration. So let's
plot the model result of the 85th iteration:

.. code:: ipython3

    post.change_input_data(interp_data, 80)




.. parsed-literal::

    <gempy.interpolator.InterpolatorData at 0x7fd9dff67c50>



Then we compute the model and plot it:

.. code:: ipython3

    lith_block, fault_block = gp.compute_model(interp_data)
    gp.plot_section(interp_data.geo_data_res, lith_block[0], 2, plot_data=True)



.. image:: ch3_files/ch3_68_0.png


or the 34th:

.. code:: ipython3

    post.change_input_data(interp_data, 15)
    lith_block, fault_block = gp.compute_model(interp_data)
    gp.plot_section(interp_data.geo_data_res, lith_block[0], 2, plot_data=True)



.. image:: ch3_files/ch3_70_0.png


or the 95th:

.. code:: ipython3

    post.change_input_data(interp_data, 95)
    lith_block, fault_block = gp.compute_model(interp_data)
    gp.plot_section(geo_data, lith_block[0], 2)



.. image:: ch3_files/ch3_72_0.png


As you can see, we have successfully perturbated the vertical layer
interface positions - although only like a 100 times. While some models
represent reasonable, geologically meaningful systems, some may not.
This is due to the stochastic selection of input parameters and an
inherent problem of the approach of Monte Carlo forward simulation -
results do not get validated in any sense. This where Bayesian inference
comes into play, as a method to constrain modeling outcomes by the use
of geological likelihood functions. We will introduce you to the
approach and how to use it with GemPy and pymc in the following chapter.

.. code:: ipython3

    ver, sim = gp.get_surfaces(interp_data,lith_block[1], None, original_scale= False)
    gp.plot_surfaces_3D_real_time(interp_data, ver, sim, posterior=post, alpha=1)
