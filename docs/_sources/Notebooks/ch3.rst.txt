
Chapter 3: Faults
=================

Here we will see how we can interpolate faults networks and how we can
use them to offset the lithological model. For this example we will use
the example of the first chapter adding a fault in the middle;

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

.. code:: ipython3

    geo_data = gp.read_pickle('BasicFault.pickle')


.. code:: ipython3

    gp.plot_data(geo_data)




.. parsed-literal::

    <gempy.Visualization.PlotData at 0x7f6d340feeb8>




.. image:: ch3_files/ch3_3_1.png


.. code:: ipython3

    gp.set_data_series(geo_data, {"fault":geo_data.formations[4], 
                          "Rest":np.delete(geo_data.formations, 4)},
                           order_series = ["fault",
                                           "Rest",
                                           ], verbose=0)

.. code:: ipython3

    gp.get_raw_data(geo_data).head()




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
          <th></th>
          <th>G_x</th>
          <th>G_y</th>
          <th>G_z</th>
          <th>X</th>
          <th>Y</th>
          <th>Z</th>
          <th>azimuth</th>
          <th>dip</th>
          <th>formation</th>
          <th>isFault</th>
          <th>order_series</th>
          <th>polarity</th>
          <th>series</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th rowspan="5" valign="top">interfaces</th>
          <th>0</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>800.0</td>
          <td>1000.0</td>
          <td>-1600.0</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>MainFault</td>
          <td>True</td>
          <td>1</td>
          <td>NaN</td>
          <td>fault</td>
        </tr>
        <tr>
          <th>1</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>1200.0</td>
          <td>1000.0</td>
          <td>-400.0</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>MainFault</td>
          <td>True</td>
          <td>1</td>
          <td>NaN</td>
          <td>fault</td>
        </tr>
        <tr>
          <th>2</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>1100.0</td>
          <td>1000.0</td>
          <td>-700.0</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>MainFault</td>
          <td>True</td>
          <td>1</td>
          <td>NaN</td>
          <td>fault</td>
        </tr>
        <tr>
          <th>3</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>900.0</td>
          <td>1000.0</td>
          <td>-1300.0</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>MainFault</td>
          <td>True</td>
          <td>1</td>
          <td>NaN</td>
          <td>fault</td>
        </tr>
        <tr>
          <th>4</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>1000.0</td>
          <td>1000.0</td>
          <td>-1000.0</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>MainFault</td>
          <td>True</td>
          <td>1</td>
          <td>NaN</td>
          <td>fault</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    interp_data = gp.InterpolatorInput(geo_data, u_grade=[3,3])


.. parsed-literal::

    I am in the setting
    float32
    I am here
    [2, 2]


.. code:: ipython3

    sol = gp.compute_model(interp_data)


.. parsed-literal::

    [3, 3]


.. code:: ipython3

    gp.plot_section(geo_data, sol[0,:], 25, plot_data = True)




.. parsed-literal::

    <gempy.Visualization.PlotData at 0x7f6d213210f0>




.. image:: ch3_files/ch3_8_1.png


.. code:: ipython3

    gp.plot_section(geo_data, sol[2,:], 25)




.. parsed-literal::

    <gempy.Visualization.PlotData at 0x7f6d2164ab00>




.. image:: ch3_files/ch3_9_1.png

