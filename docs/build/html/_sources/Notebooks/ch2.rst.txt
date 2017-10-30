
Chapter 2: A real example. Importing data and setting series
============================================================

Data Management
---------------

In this example we will show how we can import data from a csv and
generate a model with several depositional series.

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

In this case instead loading a geo\_data object directly, we will create
one. The main atributes we need to pass are: - Extent: X min, X max, Y
min, Y max, Z min, Z max - Resolution: X,Y,Z

Additionaly we can pass the address to csv files (GeoModeller3D format)
with the data.

.. code:: ipython3

    # Importing the data from csv files and settign extent and resolution
    geo_data = gp.create_data([696000,747000,6863000,6950000,-20000, 200],[50, 50, 50],
                             path_f = os.pardir+"/input_data/a_Foliations.csv",
                             path_i = os.pardir+"/input_data/a_Points.csv")

You can visualize the points in 3D (work in progress)

.. code:: ipython3

    gp.plot_data_3D(geo_data)

Or a projection in 2D:

.. code:: ipython3

    gp.plot_data(geo_data, direction='y')



.. image:: ch2_files/ch2_7_0.png


This model consist in 3 different depositional series. This mean that
only data in the same depositional series affect the interpolation. To
select with formations belong to witch series we will use the
``set_data_series`` function which takes a python dictionary as input.

We can see the unique formations with:

.. code:: ipython3

    gp.get_series(geo_data)




.. raw:: html

    <div>
    <style>
        .dataframe thead tr:only-child th {
            text-align: right;
        }
    
        .dataframe thead th {
            text-align: left;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Default serie</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>SimpleMafic2</td>
        </tr>
        <tr>
          <th>1</th>
          <td>SimpleBIF</td>
        </tr>
        <tr>
          <th>2</th>
          <td>SimpleMafic1</td>
        </tr>
        <tr>
          <th>3</th>
          <td>EarlyGranite</td>
        </tr>
      </tbody>
    </table>
    </div>



Setting the series we also give the specific order of the series. In
python 3.6 and above the dictionaries conserve the key order so it is
not necessary to give explicitly the order of the series.

Notice as well that the order of the formations within each series is
not relevant for the result but in case of being wrong can lead to
confusing color coding (work in progress).

In the representation given by ``get_series`` the elements get repeated
but is only how Pandas print tables.

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

    <gempy.strat_pile.StratigraphicPile at 0x7fa8f69f0588>




.. image:: ch2_files/ch2_11_1.png


.. code:: ipython3

    gp.plot_data(geo_data)



.. image:: ch2_files/ch2_12_0.png


Computing the model
-------------------

Now as in the previous chapter we just need to create the interpolator
object and compute the model.

.. code:: ipython3

    interp_data = gp.InterpolatorInput(geo_data)


.. parsed-literal::

    Level of Optimization:  fast_run
    Device:  cpu
    Precision:  float32


.. code:: ipython3

    lith_block, fault_block = gp.compute_model(interp_data)

Now if we analyse the results we have a 3D array where the axis 0
represent the superposition of the series (potential fields). The color
coding is working process yet.

.. code:: ipython3

    import matplotlib.pyplot as plt
    
    gp.plot_section(geo_data, lith_block[0], -1, plot_data=True, direction='z')
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)



.. image:: ch2_files/ch2_18_0.png


.. code:: ipython3

    import matplotlib.pyplot as plt
    
    gp.plot_section(geo_data, lith_block[0], -1, plot_data=True, direction='z')
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)



.. image:: ch2_files/ch2_19_0.png


.. code:: ipython3

    %matplotlib inline
    gp.plot_section(geo_data, lith_block[0],25, plot_data=True, direction='x')
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)



.. image:: ch2_files/ch2_20_0.png


The second row keeps the potential field:

.. code:: ipython3

    gp.plot_potential_field(geo_data, lith_block[1], 11, cmap='inferno_r')
    import matplotlib.pyplot as plt
    plt.colorbar()




.. parsed-literal::

    <matplotlib.colorbar.Colorbar at 0x7fa8e40e4a90>




.. image:: ch2_files/ch2_22_1.png


And the axis 2 keeps the faults network that in this model since there
is not faults does not represent anything.

Additionally with can export the blocks to vtk in order to visualize
them in Paraview. We are working in visualization in place as well.

.. code:: ipython3

    gp.export_vtk_rectilinear(geo_data, lith_block[0], path=None)

.. code:: ipython3

    ver, sim = gp.get_surfaces(interp_data, lith_block[1], None, original_scale=True)

.. code:: ipython3

    gp.plot_surfaces_3D(geo_data, ver, sim, alpha=1)
