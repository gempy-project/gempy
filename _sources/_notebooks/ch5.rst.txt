
Chapter 5: Computing forward gravity. (Under development)
=========================================================

GemPy also brings a module to compute the forward gravity response. The
idea is to be able to use gravity as a likelihood to validate the
geological models within the Bayesian inference. In this chapter we will
see how we can compute the gravity response of the sandstone model of
chapter 2.

.. code:: ipython3

    # These two lines are necessary only if gempy is not installed
    import sys, os
    sys.path.append("../..")
    
    # Importing gempy
    import gempy as gp
    
    # Embedding matplotlib figures into the notebooks
    %matplotlib inline
    
    # Aux imports
    import numpy as np


.. parsed-literal::

    WARNING (theano.tensor.blas): Using NumPy C-API based implementation for BLAS functions.


First we just recreate the model as usual.

.. code:: ipython3

    # Importing the data from csv files and settign extent and resolution
    geo_data = gp.create_data([696000,747000,6863000,6950000,-20000, 200],[50, 50, 50],
                             path_o = os.pardir+"/input_data/a_Foliations.csv",
                             path_i = os.pardir+"/input_data/a_Points.csv")

Setting the series and the formations order:

.. code:: ipython3

    # Assigning series to formations as well as their order (timewise)
    gp.set_series(geo_data, {"EarlyGranite_Series": 'EarlyGranite', 
                                  "BIF_Series":('SimpleMafic2', 'SimpleBIF'),
                                  "SimpleMafic_Series":'SimpleMafic1'}, 
                          order_series = ["EarlyGranite_Series",
                                          "BIF_Series",
                                          "SimpleMafic_Series"],
                          order_formations= ['EarlyGranite', 'SimpleMafic2', 'SimpleBIF', 'SimpleMafic1'],
                  verbose=1)




.. parsed-literal::

    <gempy.sequential_pile.StratigraphicPile at 0x7fdda82276d8>




.. image:: ch5_files/ch5_5_1.png


Projection in 2D:

.. code:: ipython3

    gp.plot_data(geo_data)



.. image:: ch5_files/ch5_7_0.png


Computing the model
-------------------

Now as in the previous chapter we just need to create the interpolator
object and compute the model.

.. code:: ipython3

    interp_data = gp.InterpolatorData(geo_data, compile_theano=True)


.. parsed-literal::

    Compiling theano function...
    Compilation Done!
    Level of Optimization:  fast_compile
    Device:  cpu
    Precision:  float32
    Number of faults:  0


.. code:: ipython3

    lith_block, fault_block = gp.compute_model(interp_data)

The green rectangle represent the area where we want to compute the
forward gravity (in this case is due to this is the area where we have
measured data). As we can see the original extent of the geological
model is not going to be enough (remember that gravity is affected by a
cone, not only the mass right below). An advantage of the method is that
we can extrapolate as much as needed keeping in mind that the error will
increase accordingly.

.. code:: ipython3

    import matplotlib.pyplot as plt
    
    gp.plot_section(geo_data, lith_block[0], -1, plot_data=True, direction='z')
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    
    from matplotlib.patches import Rectangle
    
    currentAxis = plt.gca()
    
    currentAxis.add_patch(Rectangle((7.050000e+05, 6863000),  747000 - 7.050000e+05, 6925000 - 6863000,
                          alpha=0.3, fill='none', color ='green' ))




.. parsed-literal::

    <matplotlib.patches.Rectangle at 0x7fdcf527ccf8>




.. image:: ch5_files/ch5_13_1.png


.. code:: ipython3

    ver_s, sim_s = gp.get_surfaces(interp_data,lith_block[1],
                                   None,
                                   original_scale=False)

.. code:: ipython3

    gp.plot_surfaces_3D_real_time(interp_data, ver_s, sim_s)

So we recalculate all just adding some padding around the measured data
(the green rectangle)

.. code:: ipython3

    # Importing the data from csv files and settign extent and resolution
    geo_data_extended = gp.create_data([696000-10000,747000 + 20600,6863000 - 20600,6950000 + 20600,-20000, 600],[50, 50, 50],
                             path_o = os.pardir+"/input_data/a_Foliations.csv",
                             path_i = os.pardir+"/input_data/a_Points.csv")
    
    
    # Assigning series to formations as well as their order (timewise)
    gp.set_series(geo_data_extended, {"EarlyGranite_Series": 'EarlyGranite', 
                                  "BIF_Series":('SimpleMafic2', 'SimpleBIF'),
                                  "SimpleMafic_Series":'SimpleMafic1'}, 
                          order_series = ["EarlyGranite_Series",
                                          "BIF_Series",
                                          "SimpleMafic_Series"],
                          order_formations= ['EarlyGranite', 'SimpleMafic2', 'SimpleBIF', 'SimpleMafic1'],
                  verbose=1)
    
    interp_data_extended = gp.InterpolatorData(geo_data_extended, output='geology', compile_theano=True)


.. parsed-literal::

    ../../gempy/data_management.py:61: UserWarning: path_f is deprecated use instead path_o
      warnings.warn('path_f is deprecated use instead path_o')


.. parsed-literal::

    Compiling theano function...
    Compilation Done!
    Level of Optimization:  fast_compile
    Device:  cpu
    Precision:  float32
    Number of faults:  0



.. image:: ch5_files/ch5_17_2.png


.. code:: ipython3

    lith_ext, fautl = gp.compute_model(interp_data_extended)

.. code:: ipython3

    import matplotlib.pyplot as plt
    
    gp.plot_section(geo_data_extended, lith_ext[0], -1, plot_data=True, direction='z')
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    
    from matplotlib.patches import Rectangle
    
    currentAxis = plt.gca()
    
    currentAxis.add_patch(Rectangle((7.050000e+05, 6863000),  747000 - 7.050000e+05, 6925000 - 6863000,
                          alpha=0.3, fill='none', color ='green' ))




.. parsed-literal::

    <matplotlib.patches.Rectangle at 0x7fdcf5231550>




.. image:: ch5_files/ch5_19_1.png


.. code:: ipython3

    interp_data_grav = gp.InterpolatorData(geo_data_extended, output='gravity', compile_theano=True)


.. parsed-literal::

    Compiling theano function...
    Compilation Done!
    Level of Optimization:  fast_compile
    Device:  cpu
    Precision:  float32
    Number of faults:  0


.. code:: ipython3

    gp.set_geophysics_obj(interp_data_grav,  [7.050000e+05,747000,6863000,6925000,-20000, 200],
                                                 [50, 50],)





.. parsed-literal::

    <gempy.geophysics.GravityPreprocessing at 0x7fdcb4cfdeb8>



.. code:: ipython3

    gp.precomputations_gravity(interp_data_grav, 25, 
                               [2.92, 3.1, 2.92, 2.61, 2.61])




.. parsed-literal::

    (array([[  2.42752881e-05,   1.44979296e-05,   4.65235891e-06, ...,
               1.44979296e-05,   4.65235891e-06,  -5.21516146e-06],
            [  2.42752881e-05,   1.44979296e-05,   4.65235891e-06, ...,
               1.44979296e-05,   4.65235891e-06,  -5.21516146e-06],
            [  2.42752881e-05,   1.44979296e-05,   4.65235891e-06, ...,
               1.44979296e-05,   4.65235891e-06,  -5.21516146e-06],
            ..., 
            [  2.42752881e-05,   1.44979296e-05,   4.65235891e-06, ...,
               1.44979296e-05,   4.65235891e-06,  -5.21516146e-06],
            [  2.42752881e-05,   1.44979296e-05,   4.65235891e-06, ...,
               1.44979296e-05,   4.65235891e-06,  -5.21516146e-06],
            [  2.42752881e-05,   1.44979296e-05,   4.65235891e-06, ...,
               1.44979296e-05,   4.65235891e-06,  -5.21516146e-06]]),
     array([False, False, False, ..., False, False, False], dtype=bool))



.. code:: ipython3

    lith, fault, grav = gp.compute_model(interp_data_grav, 'gravity')

.. code:: ipython3

    import matplotlib.pyplot as plt
    
    plt.imshow(grav.reshape(50,50), cmap='viridis', origin='lower', extent=[7.050000e+05,747000,6863000,6950000] )
    plt.colorbar()




.. parsed-literal::

    <matplotlib.colorbar.Colorbar at 0x7fdcb58bfba8>




.. image:: ch5_files/ch5_24_1.png

