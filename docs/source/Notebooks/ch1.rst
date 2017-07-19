
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

    geo_data = gp.read_pickle('../input_data/NoFault.pickle')
    geo_data.n_faults = 0
    print(geo_data)


.. parsed-literal::

    <gempy.DataManagement.InputData object at 0x7f165846c208>


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



.. code:: ipython3

    geo_data.interfaces




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
        <tr>
          <th>10</th>
          <td>1600.0</td>
          <td>1000.0</td>
          <td>-1200.0</td>
          <td>Reservoir</td>
          <td>Rest</td>
          <td>2</td>
          <td>False</td>
        </tr>
        <tr>
          <th>11</th>
          <td>1750.0</td>
          <td>1000.0</td>
          <td>-1250.0</td>
          <td>Reservoir</td>
          <td>Rest</td>
          <td>2</td>
          <td>False</td>
        </tr>
        <tr>
          <th>12</th>
          <td>1000.0</td>
          <td>25.0</td>
          <td>-1050.0</td>
          <td>Reservoir</td>
          <td>Rest</td>
          <td>2</td>
          <td>False</td>
        </tr>
        <tr>
          <th>13</th>
          <td>1000.0</td>
          <td>100.0</td>
          <td>-1100.0</td>
          <td>Reservoir</td>
          <td>Rest</td>
          <td>2</td>
          <td>False</td>
        </tr>
        <tr>
          <th>14</th>
          <td>1000.0</td>
          <td>1975.0</td>
          <td>-1050.0</td>
          <td>Reservoir</td>
          <td>Rest</td>
          <td>2</td>
          <td>False</td>
        </tr>
        <tr>
          <th>15</th>
          <td>1000.0</td>
          <td>1900.0</td>
          <td>-1100.0</td>
          <td>Reservoir</td>
          <td>Rest</td>
          <td>2</td>
          <td>False</td>
        </tr>
        <tr>
          <th>16</th>
          <td>900.0</td>
          <td>1000.0</td>
          <td>-650.0</td>
          <td>SecondaryReservoir</td>
          <td>Rest</td>
          <td>2</td>
          <td>False</td>
        </tr>
        <tr>
          <th>17</th>
          <td>600.0</td>
          <td>1000.0</td>
          <td>-550.0</td>
          <td>SecondaryReservoir</td>
          <td>Rest</td>
          <td>2</td>
          <td>False</td>
        </tr>
        <tr>
          <th>18</th>
          <td>1300.0</td>
          <td>1000.0</td>
          <td>-600.0</td>
          <td>SecondaryReservoir</td>
          <td>Rest</td>
          <td>2</td>
          <td>False</td>
        </tr>
        <tr>
          <th>19</th>
          <td>1900.0</td>
          <td>1000.0</td>
          <td>-800.0</td>
          <td>SecondaryReservoir</td>
          <td>Rest</td>
          <td>2</td>
          <td>False</td>
        </tr>
        <tr>
          <th>20</th>
          <td>800.0</td>
          <td>1800.0</td>
          <td>-1400.0</td>
          <td>Reservoir</td>
          <td>Rest</td>
          <td>2</td>
          <td>False</td>
        </tr>
        <tr>
          <th>21</th>
          <td>800.0</td>
          <td>200.0</td>
          <td>-1400.0</td>
          <td>Reservoir</td>
          <td>Rest</td>
          <td>2</td>
          <td>False</td>
        </tr>
        <tr>
          <th>22</th>
          <td>1150.0</td>
          <td>1000.0</td>
          <td>-1050.0</td>
          <td>Reservoir</td>
          <td>Rest</td>
          <td>2</td>
          <td>False</td>
        </tr>
        <tr>
          <th>23</th>
          <td>600.0</td>
          <td>1000.0</td>
          <td>-650.0</td>
          <td>Seal</td>
          <td>Rest</td>
          <td>2</td>
          <td>False</td>
        </tr>
        <tr>
          <th>24</th>
          <td>1300.0</td>
          <td>1000.0</td>
          <td>-700.0</td>
          <td>Seal</td>
          <td>Rest</td>
          <td>2</td>
          <td>False</td>
        </tr>
        <tr>
          <th>25</th>
          <td>1600.0</td>
          <td>1000.0</td>
          <td>-800.0</td>
          <td>Seal</td>
          <td>Rest</td>
          <td>2</td>
          <td>False</td>
        </tr>
        <tr>
          <th>26</th>
          <td>1900.0</td>
          <td>1000.0</td>
          <td>-900.0</td>
          <td>Seal</td>
          <td>Rest</td>
          <td>2</td>
          <td>False</td>
        </tr>
        <tr>
          <th>27</th>
          <td>2000.0</td>
          <td>1000.0</td>
          <td>-875.0</td>
          <td>Seal</td>
          <td>Rest</td>
          <td>2</td>
          <td>False</td>
        </tr>
        <tr>
          <th>28</th>
          <td>600.0</td>
          <td>1000.0</td>
          <td>-1350.0</td>
          <td>NonReservoirDeep</td>
          <td>Rest</td>
          <td>2</td>
          <td>False</td>
        </tr>
        <tr>
          <th>29</th>
          <td>1600.0</td>
          <td>1000.0</td>
          <td>-700.0</td>
          <td>SecondaryReservoir</td>
          <td>Rest</td>
          <td>2</td>
          <td>False</td>
        </tr>
        <tr>
          <th>30</th>
          <td>300.0</td>
          <td>1000.0</td>
          <td>-1250.0</td>
          <td>NonReservoirDeep</td>
          <td>Rest</td>
          <td>2</td>
          <td>False</td>
        </tr>
        <tr>
          <th>31</th>
          <td>2000.0</td>
          <td>1000.0</td>
          <td>-1575.0</td>
          <td>NonReservoirDeep</td>
          <td>Rest</td>
          <td>2</td>
          <td>False</td>
        </tr>
        <tr>
          <th>32</th>
          <td>1300.0</td>
          <td>1000.0</td>
          <td>-1400.0</td>
          <td>NonReservoirDeep</td>
          <td>Rest</td>
          <td>2</td>
          <td>False</td>
        </tr>
        <tr>
          <th>33</th>
          <td>1600.0</td>
          <td>1000.0</td>
          <td>-1500.0</td>
          <td>NonReservoirDeep</td>
          <td>Rest</td>
          <td>2</td>
          <td>False</td>
        </tr>
        <tr>
          <th>34</th>
          <td>1750.0</td>
          <td>1000.0</td>
          <td>-1550.0</td>
          <td>NonReservoirDeep</td>
          <td>Rest</td>
          <td>2</td>
          <td>False</td>
        </tr>
        <tr>
          <th>35</th>
          <td>1000.0</td>
          <td>1000.0</td>
          <td>-1300.0</td>
          <td>NonReservoirDeep</td>
          <td>Rest</td>
          <td>2</td>
          <td>False</td>
        </tr>
        <tr>
          <th>36</th>
          <td>1900.0</td>
          <td>1000.0</td>
          <td>-1600.0</td>
          <td>NonReservoirDeep</td>
          <td>Rest</td>
          <td>2</td>
          <td>False</td>
        </tr>
        <tr>
          <th>37</th>
          <td>900.0</td>
          <td>1000.0</td>
          <td>-750.0</td>
          <td>Seal</td>
          <td>Rest</td>
          <td>2</td>
          <td>False</td>
        </tr>
        <tr>
          <th>38</th>
          <td>2000.0</td>
          <td>1000.0</td>
          <td>-775.0</td>
          <td>SecondaryReservoir</td>
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

    <gempy.Visualization.PlotData2D at 0x7f1659329cf8>




.. image:: ch1_files/ch1_12_1.png


GemPy supports visualization in 3D as well trough vtk.

.. code:: ipython3

    gp.plot_data_3D(geo_data)


.. parsed-literal::

    1450.0 1000.0 -1150.0
    1450.0 1000.0 -1150.0
    1450.0 1000.0 -1150.0
    1450.0 1000.0 -1150.0


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

    [2, 2]
    <gempy.DataManagement.InterpolatorInput object at 0x7f16592aadd8>


.. code:: ipython3

    interp_data.get_formation_number()




.. parsed-literal::

    {'DefaultBasement': 0,
     'NonReservoirDeep': 4,
     'Reservoir': 1,
     'Seal': 3,
     'SecondaryReservoir': 2}



By default (there is a flag in case you do not need) when we create a
interp\_data object we also compile the theano function that compute the
model. That is the reason why takes long.

gempy.DataManagement.InterpolatorInput (usually called interp\_data in
the tutorials) prepares the original data to the interpolation algorithm
by scaling the coordinates for better and adding all the mathematical
parametrization needed.

.. code:: ipython3

    gp.get_kriging_parameters(interp_data)


.. parsed-literal::

    range 0.8882311582565308 3464.1015172
    Number of drift equations [2 2]
    Covariance at 0 0.01878463476896286
    Foliations nugget effect 0.009999999776482582


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

    gp.plot_section(geo_data, sol[0], 25, plot_data=True)




.. parsed-literal::

    <gempy.Visualization.PlotData2D at 0x7f1651dbe048>




.. image:: ch1_files/ch1_23_1.png


Potential field:

.. code:: ipython3

    gp.plot_potential_field(geo_data, sol[1], 25)



.. image:: ch1_files/ch1_25_0.png


