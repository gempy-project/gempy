
Chapter 1: GemPy Basic
======================

In this first example, we will show how to construct a first basic model
and the main objects and functions. First we import gempy:

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

All data get stored in a python object InputData. Therefore we can use
python serialization to save the input of the models. In the next
chapter we will see different ways to create data but for this example
we will use a stored one

.. code:: ipython3

    geo_data = gp.read_pickle('NoFault.pickle')
    geo_data.n_faults = 0
    print(geo_data)


::


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-2-10570d7499cd> in <module>()
    ----> 1 geo_data = gp.read_pickle('NoFault.pickle')
          2 geo_data.n_faults = 0
          3 print(geo_data)


    ~/PycharmProjects/gempy/gempy/GemPy_f.py in read_pickle(path)
         42         # The protocol version used is detected automatically, so we do not
         43         # have to specify it.
    ---> 44         data = pickle.load(f)
         45         return data
         46 


    AttributeError: Can't get attribute 'InputData.GridClass' on <module 'gempy.DataManagement' from '../gempy/DataManagement.py'>


This geo\_data object contains essential information that we can access
through the correspondent getters. Such a the coordinates of the grid.

.. code:: ipython3

    print(gp.get_grid(geo_data))


.. parsed-literal::

    [[    0.             0.         -2000.        ]
     [    0.             0.         -1959.18371582]
     [    0.             0.         -1918.36730957]
     ..., 
     [ 2000.          2000.           -81.63265228]
     [ 2000.          2000.           -40.81632614]
     [ 2000.          2000.             0.        ]]


The main input the potential field method is the coordinates of
interfaces points as well as the orientations. These pandas dataframes
can we access by the following methods:

Interfaces Dataframe
^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

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
          <th>isFault</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>5</th>
          <td>300.0</td>
          <td>1000.0</td>
          <td>-950.0</td>
          <td>Reservoir</td>
          <td>Rest</td>
          <td>2</td>
          <td>False</td>
        </tr>
        <tr>
          <th>6</th>
          <td>2000.0</td>
          <td>1000.0</td>
          <td>-1275.0</td>
          <td>Reservoir</td>
          <td>Rest</td>
          <td>2</td>
          <td>False</td>
        </tr>
        <tr>
          <th>7</th>
          <td>1900.0</td>
          <td>1000.0</td>
          <td>-1300.0</td>
          <td>Reservoir</td>
          <td>Rest</td>
          <td>2</td>
          <td>False</td>
        </tr>
        <tr>
          <th>8</th>
          <td>1300.0</td>
          <td>1000.0</td>
          <td>-1100.0</td>
          <td>Reservoir</td>
          <td>Rest</td>
          <td>2</td>
          <td>False</td>
        </tr>
        <tr>
          <th>9</th>
          <td>600.0</td>
          <td>1000.0</td>
          <td>-1050.0</td>
          <td>Reservoir</td>
          <td>Rest</td>
          <td>2</td>
          <td>False</td>
        </tr>
      </tbody>
    </table>
    </div>



Foliations Dataframe
^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    gp.get_raw_data(geo_data, 'foliations').head()




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
          <th>azimuth</th>
          <th>dip</th>
          <th>polarity</th>
          <th>formation</th>
          <th>series</th>
          <th>order_series</th>
          <th>G_x</th>
          <th>G_y</th>
          <th>G_z</th>
          <th>isFault</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>1</th>
          <td>1450.0</td>
          <td>1000.0</td>
          <td>-1150.0</td>
          <td>90.0</td>
          <td>18.435</td>
          <td>1</td>
          <td>Reservoir</td>
          <td>Rest</td>
          <td>2</td>
          <td>0.316229</td>
          <td>1.936342e-17</td>
          <td>0.948683</td>
          <td>False</td>
        </tr>
      </tbody>
    </table>
    </div>



It is important to notice the columns of each data frame. These not only
contains the geometrical properties of the data but also the
**formation** and **series** at which they belong. This division is
fundamental in order to preserve the depositional ages of the setting to
model.

A projection of the aforementioned data can be visualized in to 2D by
the following function. It is possible to choose the direction of
visualization as well as the series:

.. code:: ipython3

    geo_data.geo_data_type= 3
    gp.plot_data(geo_data, direction='y')




.. parsed-literal::

    <gempy.Visualization.PlotData at 0x7fd8e729c828>




.. image:: ch1_files/ch1_11_1.png


GemPy supports visualization in 3D as well trough vtk.

.. code:: ipython3

    gp.visualize(geo_data)

The ins and outs of Input data objects
--------------------------------------

As we have seen objects DataManagement.InputData (usually called
geo\_data in the tutorials) aim to have all the original geological
properties, measurements and geological relations stored.

Once we have the data ready to generate a model, we will need to create
the next object type towards the final geological model:

.. code:: ipython3

    interp_data = gp.InterpolatorInput(geo_data, u_grade = [3])
    print(interp_data)


.. parsed-literal::

    I am here
    I am in the setting
    float32
    I am here
    [2, 2]


By default (there is a flag in case you do not need) when we create a
interp\_data object we also compile the theano function that compute the
model. That is the reason why takes long.

gempy.DataManagement.InterpolatorInput (usually called interp\_data in
the tutorials) prepares the original data to the interpolation algorithm
by scaling the coordinates for better and adding all the mathematical
parametrization needed.

.. code:: ipython3

    gp.get_kriging_parameters(interp_data)

These later parameters have a default value computed from the original
data or can be changed by the user (be careful of changing any of these
if you do not fully understand their meaning).

At this point, we have all what we need to compute our model. By default
everytime we compute a model we obtain 3 results:

-  Lithology block model
-  The potential field
-  Faults network block model

.. code:: ipython3

    sol = gp.compute_model(interp_data)


.. parsed-literal::

    [3]


This solution can be plot with the correspondent plotting function.
Blocks:

.. code:: ipython3

    gp.plot_section(geo_data, sol[0], 25)




.. parsed-literal::

    <gempy.Visualization.PlotData at 0x7f629ef6b9e8>




.. image:: ch1_files/ch1_21_1.png


Potential field:

.. code:: ipython3

    gp.plot_potential_field(geo_data, sol[1], 25)



.. image:: ch1_files/ch1_23_0.png

