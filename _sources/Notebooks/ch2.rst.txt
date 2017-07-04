
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
    
    gp.get_raw_data(geo_data, 'interfaces').head()




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
          <th>X</th>
          <th>Y</th>
          <th>Z</th>
          <th>formation</th>
          <th>series</th>
          <th>order_series</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>735484.817806</td>
          <td>6.891936e+06</td>
          <td>-1819.319309</td>
          <td>SimpleMafic2</td>
          <td>Default serie</td>
          <td>1</td>
        </tr>
        <tr>
          <th>1</th>
          <td>722693.188000</td>
          <td>6.907492e+06</td>
          <td>555.452867</td>
          <td>SimpleMafic1</td>
          <td>Default serie</td>
          <td>1</td>
        </tr>
        <tr>
          <th>2</th>
          <td>725092.188000</td>
          <td>6.913005e+06</td>
          <td>514.864987</td>
          <td>SimpleMafic1</td>
          <td>Default serie</td>
          <td>1</td>
        </tr>
        <tr>
          <th>3</th>
          <td>744692.688000</td>
          <td>6.890291e+06</td>
          <td>496.019711</td>
          <td>SimpleMafic1</td>
          <td>Default serie</td>
          <td>1</td>
        </tr>
        <tr>
          <th>4</th>
          <td>738924.813000</td>
          <td>6.900194e+06</td>
          <td>551.633725</td>
          <td>SimpleMafic1</td>
          <td>Default serie</td>
          <td>1</td>
        </tr>
      </tbody>
    </table>
    </div>



You can visualize the points in 3D (work in progress)

.. code:: ipython3

    gp.visualize(geo_data)

Or a projection in 2D:

.. code:: ipython3

    gp.plot_data(geo_data, direction='z')




.. parsed-literal::

    <gempy.Visualization.PlotData at 0x7f8d04466ba8>




.. image:: ch2_files/ch2_7_1.png


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
    gp.set_data_series(geo_data, {"EarlyGranite_Series": 'EarlyGranite', 
                                  "BIF_Series":('SimpleMafic2', 'SimpleBIF'),
                                  "SimpleMafic_Series":'SimpleMafic1'}, 
                          order_series = ["EarlyGranite_Series",
                                          "BIF_Series",
                                          "SimpleMafic_Series"], verbose=1)




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
          <th>EarlyGranite_Series</th>
          <th>BIF_Series</th>
          <th>SimpleMafic_Series</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>EarlyGranite</td>
          <td>SimpleMafic2</td>
          <td>SimpleMafic1</td>
        </tr>
        <tr>
          <th>1</th>
          <td>EarlyGranite</td>
          <td>SimpleBIF</td>
          <td>SimpleMafic1</td>
        </tr>
      </tbody>
    </table>
    </div>



Computing the model
-------------------

Now as in the previous chapter we just need to create the interpolator
object and compute the model.

.. code:: ipython3

    interp_data = gp.InterpolatorInput(geo_data)


.. parsed-literal::

    I am in the setting
    float32
    I am here
    [2, 2]


.. code:: ipython3

    sol = gp.compute_model(interp_data)


.. parsed-literal::

    [9 9 9]


.. parsed-literal::

    /home/miguel/anaconda3/lib/python3.6/site-packages/scipy/linalg/basic.py:223: RuntimeWarning: scipy.linalg.solve
    Ill-conditioned matrix detected. Result is not guaranteed to be accurate.
    Reciprocal condition number: 2.4606194415355276e-08
      ' condition number: {}'.format(rcond), RuntimeWarning)


Now if we analyse the results we have a 3D array where the axis 0
represent the superposition of the series (potential fields). The color
coding is working process yet.

.. code:: ipython3

    import matplotlib.pyplot as plt
    gp.plot_section(geo_data, sol[0,0,:], 11)
    plt.show()
    gp.plot_section(geo_data, sol[1,0,:], 11)
    plt.show()
    gp.plot_section(geo_data, sol[2,0,:], 11)
    plt.show()



.. image:: ch2_files/ch2_17_0.png



.. image:: ch2_files/ch2_17_1.png



.. image:: ch2_files/ch2_17_2.png


The axis 1 keeps the potential field:

.. code:: ipython3

    gp.plot_potential_field(geo_data, sol[0,1,:], 11, cmap='inferno_r')
    plt.show()
    gp.plot_potential_field(geo_data, sol[1,1,:], 11, cmap='inferno_r')
    plt.show()
    gp.plot_potential_field(geo_data, sol[2,1,:], 11, cmap='inferno_r')
    plt.show()



.. image:: ch2_files/ch2_19_0.png



.. image:: ch2_files/ch2_19_1.png



.. image:: ch2_files/ch2_19_2.png


And the axis 2 keeps the faults network that in this model since there
is not faults does not represent anything.

Additionally with can export the blocks to vtk in order to visualize
them in Paraview. We are working in visualization in place as well.

.. code:: ipython3

    gp.export_vtk_rectilinear(geo_data, sol[-1, 0, :], path=None)
