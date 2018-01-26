
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
    sys.path.append("../")
    
    # Importing gempy
    import gempy as gp
    
    # Embedding matplotlib figures into the notebooks
    %matplotlib inline
    
    # Aux imports
    import numpy as np

First we just recreate the model as usual.

.. code:: ipython3

    # Importing the data from csv files and settign extent and resolution
    geo_data = gp.create_data([696000,747000,6863000,6950000,-20000, 200],[50, 50, 50],
                             path_f = os.pardir+"/input_data/a_Foliations.csv",
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

    <gempy.strat_pile.StratigraphicPile at 0x7fa630047f98>




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

    interp_data = gp.InterpolatorInput(geo_data)


.. parsed-literal::

    Level of Optimization:  fast_compile
    Device:  cpu
    Precision:  float32


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

    <matplotlib.patches.Rectangle at 0x7fa60a8e4860>




.. image:: ch5_files/ch5_13_1.png


So we recalculate all just adding some padding around the measured data
(the green rectangle)

.. code:: ipython3

    # Importing the data from csv files and settign extent and resolution
    geo_data_extended = gp.create_data([696000-10000,747000 + 20600,6863000 - 20600,6950000 + 20600,-20000, 600],[50, 50, 50],
                             path_f = os.pardir+"/input_data/a_Foliations.csv",
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
    
    interp_data_extended = gp.InterpolatorInput(geo_data_extended, output='geology', compile_theano=True)


.. parsed-literal::

    Level of Optimization:  fast_compile
    Device:  cpu
    Precision:  float32



.. image:: ch5_files/ch5_15_1.png


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

    <matplotlib.patches.Rectangle at 0x7fa61f180128>




.. image:: ch5_files/ch5_17_1.png


.. code:: ipython3

    interp_data_grav = gp.InterpolatorInput(geo_data_extended, output='gravity', compile_theano=True)


.. parsed-literal::

    Level of Optimization:  fast_compile
    Device:  cpu
    Precision:  float32


.. code:: ipython3

    gp.set_geophysics_obj(interp_data_grav,  [7.050000e+05,747000,6863000,6925000,-20000, 200],
                                                 [50,50], )




.. parsed-literal::

    <gempy.GeoPhysics.GeoPhysicsPreprocessing_pro at 0x7fa5f0125358>



.. code:: ipython3

    gp.precomputations_gravity(interp_data_grav, 25, [2.92, 3.1, 2.92, 2.61, 2.61])




.. parsed-literal::

    (array([[  2.32206772e-05,   1.38317570e-05,   4.37779836e-06, ...,
               1.38316011e-05,   4.37774898e-06,  -5.09674338e-06],
            [  2.32206772e-05,   1.38317570e-05,   4.37779837e-06, ...,
               1.38316011e-05,   4.37774898e-06,  -5.09674338e-06],
            [  2.32206772e-05,   1.38317570e-05,   4.37779837e-06, ...,
               1.38316011e-05,   4.37774898e-06,  -5.09674338e-06],
            ..., 
            [  2.32204160e-05,   1.38316011e-05,   4.37774898e-06, ...,
               1.38316011e-05,   4.37774898e-06,  -5.09674338e-06],
            [  2.32204160e-05,   1.38316011e-05,   4.37774898e-06, ...,
               1.38316011e-05,   4.37774898e-06,  -5.09674338e-06],
            [  2.32204160e-05,   1.38316011e-05,   4.37774898e-06, ...,
               1.38316011e-05,   4.37774898e-06,  -5.09674338e-06]]),
     array([False, False, False, ..., False, False, False], dtype=bool))



.. code:: ipython3

    grav = gp.compute_model(interp_data_grav, 'gravity')

.. code:: ipython3

    plt.imshow(grav.reshape(50,50), cmap='viridis', origin='lower', extent=[7.050000e+05,747000,6863000,6950000] )
    plt.colorbar()




.. parsed-literal::

    <matplotlib.colorbar.Colorbar at 0x7fa5ebdc1a58>




.. image:: ch5_files/ch5_22_1.png

