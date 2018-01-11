
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

All data get stored in a python object InputData. This object can be
easily stored in a Python pickle. However, these files have the
limitation that all dependecies must have the same versions as those
when the pickle were created. For these reason to have more stable
tutorials we will generate the InputData from raw data---i.e. csv files
exported from Geomodeller.

These csv files can be found in the input\_data folder in the root
folder of GemPy. These tables contains uniquely the XYZ (and poles,
azimuth and polarity in the foliation case) as well as their respective
formation name (but not necessary the formation order).

.. code:: ipython3

    # Importing the data from csv files and settign extent and resolution
    geo_data = gp.create_data([0,2000,0,2000,-2000,0],[ 50,50,50],
                             path_f = os.pardir+"/input_data/FabLessPoints_Foliations.csv",
                             path_i = os.pardir+"/input_data/FabLessPoints_Points.csv")

.. code:: ipython3

    import pandas as pn
    gp.set_interfaces(geo_data,
                      pn.DataFrame(data = {"X" : (1200, 1800),
                                           "Y" : (1000,  1000),
                                           "Z" : (- 1600, -1800),
                                           "formation" : np.tile("Layer 3", 2),
                                                          }), append=True)
    
    
    
    gp.set_foliations(geo_data,
                       pn.DataFrame(data = {"X" : (1500),
                          "Y" : (1000 ),
                          "Z" : ( -1400),
                          "azimuth" : ( 90),
                          "dip" : (20),
                          "polarity" : (1),
                          "formation" : ['Layer 3'],
                                           }),append=True)
    
    
    gp.set_interfaces(geo_data,
                      pn.DataFrame(data = {"X" : (500, 600),
                                           "Y" : (1000,  1000),
                                           "Z" : (- 150, -1700),
                                           "formation" : np.tile("fault2", 2),
                                                          }), append=True)
    
    
    
    gp.set_foliations(geo_data,
                       pn.DataFrame(data = {"X" : (400),
                          "Y" : (1000),
                          "Z" : ( -1300),
                          "azimuth" : ( 90),
                          "dip" : (90),
                          "polarity" : (1),
                          "formation" : ['fault2'],
                                           }),append=True)

With the command get data is possible to see all the input data.
However, at the moment the (depositional) order of the formation and the
separation of the series (More explanation about this in the next
notebook) is totally arbitrary.

.. code:: ipython3

    gp.get_data(geo_data)




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
          <th>X</th>
          <th>Y</th>
          <th>Z</th>
          <th>azimuth</th>
          <th>dip</th>
          <th>formation</th>
          <th>polarity</th>
          <th>series</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th rowspan="41" valign="top">interfaces</th>
          <th>0</th>
          <td>800</td>
          <td>1000</td>
          <td>-1600</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>MainFault</td>
          <td>NaN</td>
          <td>fault</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1200</td>
          <td>1000</td>
          <td>-400</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>MainFault</td>
          <td>NaN</td>
          <td>fault</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1100</td>
          <td>1000</td>
          <td>-700</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>MainFault</td>
          <td>NaN</td>
          <td>fault</td>
        </tr>
        <tr>
          <th>3</th>
          <td>900</td>
          <td>1000</td>
          <td>-1300</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>MainFault</td>
          <td>NaN</td>
          <td>fault</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1000</td>
          <td>1000</td>
          <td>-1000</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>MainFault</td>
          <td>NaN</td>
          <td>fault</td>
        </tr>
        <tr>
          <th>5</th>
          <td>800</td>
          <td>200</td>
          <td>-1400</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Reservoir</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>6</th>
          <td>800</td>
          <td>1800</td>
          <td>-1400</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Reservoir</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>7</th>
          <td>600</td>
          <td>1000</td>
          <td>-1050</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Reservoir</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>8</th>
          <td>300</td>
          <td>1000</td>
          <td>-950</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Reservoir</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>9</th>
          <td>2000</td>
          <td>1000</td>
          <td>-1275</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Reservoir</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>10</th>
          <td>1900</td>
          <td>1000</td>
          <td>-1300</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Reservoir</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>11</th>
          <td>1300</td>
          <td>1000</td>
          <td>-1100</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Reservoir</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>12</th>
          <td>1600</td>
          <td>1000</td>
          <td>-1200</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Reservoir</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>13</th>
          <td>1750</td>
          <td>1000</td>
          <td>-1250</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Reservoir</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>14</th>
          <td>1000</td>
          <td>100</td>
          <td>-1100</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Reservoir</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>15</th>
          <td>1000</td>
          <td>1975</td>
          <td>-1050</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Reservoir</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>16</th>
          <td>1000</td>
          <td>1900</td>
          <td>-1100</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Reservoir</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>17</th>
          <td>1150</td>
          <td>1000</td>
          <td>-1050</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Reservoir</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>18</th>
          <td>1000</td>
          <td>25</td>
          <td>-1050</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Reservoir</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>19</th>
          <td>1300</td>
          <td>1000</td>
          <td>-600</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>SecondaryReservoir</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>20</th>
          <td>600</td>
          <td>1000</td>
          <td>-550</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>SecondaryReservoir</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>21</th>
          <td>900</td>
          <td>1000</td>
          <td>-650</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>SecondaryReservoir</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>22</th>
          <td>1600</td>
          <td>1000</td>
          <td>-700</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>SecondaryReservoir</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>23</th>
          <td>1900</td>
          <td>1000</td>
          <td>-800</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>SecondaryReservoir</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>24</th>
          <td>2000</td>
          <td>1000</td>
          <td>-775</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>SecondaryReservoir</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>25</th>
          <td>2000</td>
          <td>1000</td>
          <td>-875</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Seal</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>26</th>
          <td>600</td>
          <td>1000</td>
          <td>-650</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Seal</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>27</th>
          <td>1300</td>
          <td>1000</td>
          <td>-700</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Seal</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>28</th>
          <td>1900</td>
          <td>1000</td>
          <td>-900</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Seal</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>29</th>
          <td>900</td>
          <td>1000</td>
          <td>-750</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Seal</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>30</th>
          <td>1600</td>
          <td>1000</td>
          <td>-800</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Seal</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>31</th>
          <td>600</td>
          <td>1000</td>
          <td>-1350</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Overlying</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>32</th>
          <td>300</td>
          <td>1000</td>
          <td>-1250</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Overlying</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>33</th>
          <td>1000</td>
          <td>1000</td>
          <td>-1300</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Overlying</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>34</th>
          <td>2000</td>
          <td>1000</td>
          <td>-1575</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Overlying</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>35</th>
          <td>1300</td>
          <td>1000</td>
          <td>-1400</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Overlying</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>36</th>
          <td>1600</td>
          <td>1000</td>
          <td>-1500</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Overlying</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>37</th>
          <td>1750</td>
          <td>1000</td>
          <td>-1550</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Overlying</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>38</th>
          <td>1900</td>
          <td>1000</td>
          <td>-1600</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Overlying</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>39</th>
          <td>1800</td>
          <td>1000</td>
          <td>200</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Layer 3</td>
          <td>NaN</td>
          <td>unco</td>
        </tr>
        <tr>
          <th>40</th>
          <td>1200</td>
          <td>1000</td>
          <td>200</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Layer 3</td>
          <td>NaN</td>
          <td>unco</td>
        </tr>
        <tr>
          <th rowspan="3" valign="top">foliations</th>
          <th>0</th>
          <td>917.45</td>
          <td>1000</td>
          <td>-1135.4</td>
          <td>270</td>
          <td>71.565</td>
          <td>MainFault</td>
          <td>1</td>
          <td>fault</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1450</td>
          <td>1000</td>
          <td>-1150</td>
          <td>90</td>
          <td>18.435</td>
          <td>Reservoir</td>
          <td>1</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1500</td>
          <td>1000</td>
          <td>400</td>
          <td>90</td>
          <td>0</td>
          <td>Layer 3</td>
          <td>1</td>
          <td>unco</td>
        </tr>
      </tbody>
    </table>
    </div>



To set the number of (depositional) series and which formation belongs
to which, it is possible to use the function set\_series. Here, there
are two important things to notice:

-  set\_series requires a dictionary. In Python dictionaries are not
   order (keys dictionaries since Python 3.6 are) and therefore there is
   not guarantee of having the right order.

   -  The order of the series are vital for the method (from younger to
      older).
   -  The order of the formations (as long they belong to the correct
      series) are only important for the color code. If the order of the
      pile differs from the final result the color of the interfaces and
      input data will be different

-  Every fault is treated as an independent series and **have to be at
   the top of the pile**. The relative order between the distinct faults
   represent the tectonic relation between them (from younger to older
   as well).

The order of the series (for Python < 3.6, otherwise passing the correct
order of keys in the dictionary is enough) can be given as attribute as
in the cell below. For the order of formations can be passed as
attribute as well or using the specific function set\_order\_formations.

.. code:: ipython3

    # Assigning series to formations as well as their order (timewise)
    gp.set_series(geo_data, {"fault":'MainFault', 
                          "Rest": ('SecondaryReservoir','Seal', 'Reservoir', 'Overlying'),
                            'unco' : 'Layer 3', 'Fault_serie2':'fault2'},
                           order_series = ["fault", 'Fault_serie2', 'Rest', 'unco'],
                           order_formations=['MainFault', 'fault2',
                                             'SecondaryReservoir', 'Seal','Reservoir', 'Overlying', 'Layer 3',
                                             ]) 




.. parsed-literal::

    <gempy.strat_pile.StratigraphicPile at 0x7fdf34fda898>




.. image:: ch1-Copy1_files/ch1-Copy1_8_1.png


As an alternative the stratigraphic pile is interactive given the right
backend (try %matplotlib notebook or %matplotlib qt5). These backends
sometimes give some trouble though. Try to execute the cell twice:

.. code:: ipython3

    %matplotlib notebook
    gp.get_stratigraphic_pile(geo_data)



.. parsed-literal::

    <IPython.core.display.Javascript object>



.. raw:: html

    <img src="data:image/png;base64,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" width="747.9999777078635">




.. parsed-literal::

    <gempy.strat_pile.StratigraphicPile at 0x7f6e21c1d470>



Notice that the colors depends on the order and therefore every time the
cell is executed the colors are always in the same position. Be aware of
the legend to be sure that the pile is as you wish!! (In the future
every color will have the annotation within the rectangles to avoid
confusion)

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

    gp.get_data(geo_data, 'interfaces').head()




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
          <th>annotations</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1000</td>
          <td>1000</td>
          <td>-1000</td>
          <td>MainFault</td>
          <td>fault</td>
          <td>${\bf{x}}_{\alpha \,{\bf{1}},0}$</td>
        </tr>
        <tr>
          <th>1</th>
          <td>800</td>
          <td>1000</td>
          <td>-1600</td>
          <td>MainFault</td>
          <td>fault</td>
          <td>${\bf{x}}_{\alpha \,{\bf{1}},1}$</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1200</td>
          <td>1000</td>
          <td>-400</td>
          <td>MainFault</td>
          <td>fault</td>
          <td>${\bf{x}}_{\alpha \,{\bf{1}},2}$</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1100</td>
          <td>1000</td>
          <td>-700</td>
          <td>MainFault</td>
          <td>fault</td>
          <td>${\bf{x}}_{\alpha \,{\bf{1}},3}$</td>
        </tr>
        <tr>
          <th>4</th>
          <td>900</td>
          <td>1000</td>
          <td>-1300</td>
          <td>MainFault</td>
          <td>fault</td>
          <td>${\bf{x}}_{\alpha \,{\bf{1}},4}$</td>
        </tr>
      </tbody>
    </table>
    </div>



Foliations Dataframe
^^^^^^^^^^^^^^^^^^^^

Now the formations and the series are correctly set.

.. code:: ipython3

    gp.get_data(geo_data, 'foliations').head()




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
          <th>dip</th>
          <th>azimuth</th>
          <th>polarity</th>
          <th>formation</th>
          <th>series</th>
          <th>annotations</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>917.45</td>
          <td>1000.0</td>
          <td>-1135.398</td>
          <td>71.565</td>
          <td>270.0</td>
          <td>1</td>
          <td>MainFault</td>
          <td>fault</td>
          <td>${\bf{x}}_{\beta \,{\bf{1}},0}$</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1450.00</td>
          <td>1000.0</td>
          <td>-1150.000</td>
          <td>18.435</td>
          <td>90.0</td>
          <td>1</td>
          <td>Reservoir</td>
          <td>Rest</td>
          <td>${\bf{x}}_{\beta \,{\bf{4}},0}$</td>
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

    %matplotlib inline
    gp.plot_data(geo_data, direction='y')



.. image:: ch1-Copy1_files/ch1-Copy1_20_0.png


GemPy supports visualization in 3D as well trough vtk. These plots are
interactive. Try to drag and drop a point or interface! In the
perpendicular views only 2D movements are possible to help to place the
data where is required.

.. code:: ipython3

    gp.plot_data_3D(geo_data)

The ins and outs of Input data objects
--------------------------------------

As we have seen objects DataManagement.InputData (usually called
geo\_data in the tutorials) aim to have all the original geological
properties, measurements and geological relations stored.

Once we have the data ready to generate a model, we will need to create
the next object type towards the final geological model:

.. code:: ipython3

    interp_data = gp.InterpolatorInput(geo_data, u_grade=[3,3, 3, 3])#, verbose=['faults_matrix', 'covariance_matrix'])
    print(interp_data)


.. parsed-literal::

    Level of Optimization:  fast_run
    Device:  cpu
    Precision:  float32
    <gempy.DataManagement.InterpolatorInput object at 0x7fdf34ec0320>


.. code:: ipython3

    len(interp_data.interpolator.tg.len_series_f.get_value())-1 




.. parsed-literal::

    3



.. code:: ipython3

    interp_data.get_formation_number()




.. parsed-literal::

    {'DefaultBasement': 0,
     'Layer 3': 6,
     'MainFault': 1,
     'Overlying': 5,
     'Reservoir': 4,
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
everytime we compute a model we obtain:

-  Lithology block model

   -  with the lithology values in 0
   -  with the potential field values in 1

-  Fault block model

   -  with the faults zones values (i.e. every divided region by each
      fault has one number) in 0
   -  with the potential field values in 1

.. code:: ipython3

    interp_data.geo_data_res.interfaces




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
          <th>annotations</th>
          <th>formation</th>
          <th>formation number</th>
          <th>isFault</th>
          <th>order_series</th>
          <th>series</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0.410356</td>
          <td>0.5001</td>
          <td>0.371895</td>
          <td>${\bf{x}}_{\alpha \,{\bf{1}},0}$</td>
          <td>MainFault</td>
          <td>1</td>
          <td>True</td>
          <td>1</td>
          <td>fault</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.512921</td>
          <td>0.5001</td>
          <td>0.679587</td>
          <td>${\bf{x}}_{\alpha \,{\bf{1}},1}$</td>
          <td>MainFault</td>
          <td>1</td>
          <td>True</td>
          <td>1</td>
          <td>fault</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.487279</td>
          <td>0.5001</td>
          <td>0.602664</td>
          <td>${\bf{x}}_{\alpha \,{\bf{1}},2}$</td>
          <td>MainFault</td>
          <td>1</td>
          <td>True</td>
          <td>1</td>
          <td>fault</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.435997</td>
          <td>0.5001</td>
          <td>0.448818</td>
          <td>${\bf{x}}_{\alpha \,{\bf{1}},3}$</td>
          <td>MainFault</td>
          <td>1</td>
          <td>True</td>
          <td>1</td>
          <td>fault</td>
        </tr>
        <tr>
          <th>4</th>
          <td>0.461638</td>
          <td>0.5001</td>
          <td>0.525741</td>
          <td>${\bf{x}}_{\alpha \,{\bf{1}},4}$</td>
          <td>MainFault</td>
          <td>1</td>
          <td>True</td>
          <td>1</td>
          <td>fault</td>
        </tr>
        <tr>
          <th>5</th>
          <td>0.538562</td>
          <td>0.5001</td>
          <td>0.628305</td>
          <td>${\bf{x}}_{\alpha \,{\bf{2}},0}$</td>
          <td>SecondaryReservoir</td>
          <td>2</td>
          <td>False</td>
          <td>2</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>6</th>
          <td>0.359074</td>
          <td>0.5001</td>
          <td>0.641126</td>
          <td>${\bf{x}}_{\alpha \,{\bf{2}},1}$</td>
          <td>SecondaryReservoir</td>
          <td>2</td>
          <td>False</td>
          <td>2</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>7</th>
          <td>0.435997</td>
          <td>0.5001</td>
          <td>0.615485</td>
          <td>${\bf{x}}_{\alpha \,{\bf{2}},2}$</td>
          <td>SecondaryReservoir</td>
          <td>2</td>
          <td>False</td>
          <td>2</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>8</th>
          <td>0.615485</td>
          <td>0.5001</td>
          <td>0.602664</td>
          <td>${\bf{x}}_{\alpha \,{\bf{2}},3}$</td>
          <td>SecondaryReservoir</td>
          <td>2</td>
          <td>False</td>
          <td>2</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>9</th>
          <td>0.692408</td>
          <td>0.5001</td>
          <td>0.577023</td>
          <td>${\bf{x}}_{\alpha \,{\bf{2}},4}$</td>
          <td>SecondaryReservoir</td>
          <td>2</td>
          <td>False</td>
          <td>2</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>10</th>
          <td>0.718049</td>
          <td>0.5001</td>
          <td>0.583433</td>
          <td>${\bf{x}}_{\alpha \,{\bf{2}},5}$</td>
          <td>SecondaryReservoir</td>
          <td>2</td>
          <td>False</td>
          <td>2</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>11</th>
          <td>0.718049</td>
          <td>0.5001</td>
          <td>0.557792</td>
          <td>${\bf{x}}_{\alpha \,{\bf{3}},0}$</td>
          <td>Seal</td>
          <td>3</td>
          <td>False</td>
          <td>2</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>12</th>
          <td>0.359074</td>
          <td>0.5001</td>
          <td>0.615485</td>
          <td>${\bf{x}}_{\alpha \,{\bf{3}},1}$</td>
          <td>Seal</td>
          <td>3</td>
          <td>False</td>
          <td>2</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>13</th>
          <td>0.538562</td>
          <td>0.5001</td>
          <td>0.602664</td>
          <td>${\bf{x}}_{\alpha \,{\bf{3}},2}$</td>
          <td>Seal</td>
          <td>3</td>
          <td>False</td>
          <td>2</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>14</th>
          <td>0.692408</td>
          <td>0.5001</td>
          <td>0.551382</td>
          <td>${\bf{x}}_{\alpha \,{\bf{3}},3}$</td>
          <td>Seal</td>
          <td>3</td>
          <td>False</td>
          <td>2</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>15</th>
          <td>0.435997</td>
          <td>0.5001</td>
          <td>0.589844</td>
          <td>${\bf{x}}_{\alpha \,{\bf{3}},4}$</td>
          <td>Seal</td>
          <td>3</td>
          <td>False</td>
          <td>2</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>16</th>
          <td>0.615485</td>
          <td>0.5001</td>
          <td>0.577023</td>
          <td>${\bf{x}}_{\alpha \,{\bf{3}},5}$</td>
          <td>Seal</td>
          <td>3</td>
          <td>False</td>
          <td>2</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>17</th>
          <td>0.410356</td>
          <td>0.294972</td>
          <td>0.423177</td>
          <td>${\bf{x}}_{\alpha \,{\bf{4}},0}$</td>
          <td>Reservoir</td>
          <td>4</td>
          <td>False</td>
          <td>2</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>18</th>
          <td>0.410356</td>
          <td>0.705228</td>
          <td>0.423177</td>
          <td>${\bf{x}}_{\alpha \,{\bf{4}},1}$</td>
          <td>Reservoir</td>
          <td>4</td>
          <td>False</td>
          <td>2</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>19</th>
          <td>0.359074</td>
          <td>0.5001</td>
          <td>0.512921</td>
          <td>${\bf{x}}_{\alpha \,{\bf{4}},2}$</td>
          <td>Reservoir</td>
          <td>4</td>
          <td>False</td>
          <td>2</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>20</th>
          <td>0.282151</td>
          <td>0.5001</td>
          <td>0.538562</td>
          <td>${\bf{x}}_{\alpha \,{\bf{4}},3}$</td>
          <td>Reservoir</td>
          <td>4</td>
          <td>False</td>
          <td>2</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>21</th>
          <td>0.718049</td>
          <td>0.5001</td>
          <td>0.455228</td>
          <td>${\bf{x}}_{\alpha \,{\bf{4}},4}$</td>
          <td>Reservoir</td>
          <td>4</td>
          <td>False</td>
          <td>2</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>22</th>
          <td>0.692408</td>
          <td>0.5001</td>
          <td>0.448818</td>
          <td>${\bf{x}}_{\alpha \,{\bf{4}},5}$</td>
          <td>Reservoir</td>
          <td>4</td>
          <td>False</td>
          <td>2</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>23</th>
          <td>0.538562</td>
          <td>0.5001</td>
          <td>0.5001</td>
          <td>${\bf{x}}_{\alpha \,{\bf{4}},6}$</td>
          <td>Reservoir</td>
          <td>4</td>
          <td>False</td>
          <td>2</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>24</th>
          <td>0.615485</td>
          <td>0.5001</td>
          <td>0.474459</td>
          <td>${\bf{x}}_{\alpha \,{\bf{4}},7}$</td>
          <td>Reservoir</td>
          <td>4</td>
          <td>False</td>
          <td>2</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>25</th>
          <td>0.653946</td>
          <td>0.5001</td>
          <td>0.461638</td>
          <td>${\bf{x}}_{\alpha \,{\bf{4}},8}$</td>
          <td>Reservoir</td>
          <td>4</td>
          <td>False</td>
          <td>2</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>26</th>
          <td>0.461638</td>
          <td>0.269331</td>
          <td>0.5001</td>
          <td>${\bf{x}}_{\alpha \,{\bf{4}},9}$</td>
          <td>Reservoir</td>
          <td>4</td>
          <td>False</td>
          <td>2</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>27</th>
          <td>0.461638</td>
          <td>0.7501</td>
          <td>0.512921</td>
          <td>${\bf{x}}_{\alpha \,{\bf{4}},10}$</td>
          <td>Reservoir</td>
          <td>4</td>
          <td>False</td>
          <td>2</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>28</th>
          <td>0.461638</td>
          <td>0.730869</td>
          <td>0.5001</td>
          <td>${\bf{x}}_{\alpha \,{\bf{4}},11}$</td>
          <td>Reservoir</td>
          <td>4</td>
          <td>False</td>
          <td>2</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>29</th>
          <td>0.5001</td>
          <td>0.5001</td>
          <td>0.512921</td>
          <td>${\bf{x}}_{\alpha \,{\bf{4}},12}$</td>
          <td>Reservoir</td>
          <td>4</td>
          <td>False</td>
          <td>2</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>30</th>
          <td>0.461638</td>
          <td>0.2501</td>
          <td>0.512921</td>
          <td>${\bf{x}}_{\alpha \,{\bf{4}},13}$</td>
          <td>Reservoir</td>
          <td>4</td>
          <td>False</td>
          <td>2</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>31</th>
          <td>0.359074</td>
          <td>0.5001</td>
          <td>0.435997</td>
          <td>${\bf{x}}_{\alpha \,{\bf{5}},0}$</td>
          <td>Overlying</td>
          <td>5</td>
          <td>False</td>
          <td>2</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>32</th>
          <td>0.282151</td>
          <td>0.5001</td>
          <td>0.461638</td>
          <td>${\bf{x}}_{\alpha \,{\bf{5}},1}$</td>
          <td>Overlying</td>
          <td>5</td>
          <td>False</td>
          <td>2</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>33</th>
          <td>0.461638</td>
          <td>0.5001</td>
          <td>0.448818</td>
          <td>${\bf{x}}_{\alpha \,{\bf{5}},2}$</td>
          <td>Overlying</td>
          <td>5</td>
          <td>False</td>
          <td>2</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>34</th>
          <td>0.718049</td>
          <td>0.5001</td>
          <td>0.378305</td>
          <td>${\bf{x}}_{\alpha \,{\bf{5}},3}$</td>
          <td>Overlying</td>
          <td>5</td>
          <td>False</td>
          <td>2</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>35</th>
          <td>0.538562</td>
          <td>0.5001</td>
          <td>0.423177</td>
          <td>${\bf{x}}_{\alpha \,{\bf{5}},4}$</td>
          <td>Overlying</td>
          <td>5</td>
          <td>False</td>
          <td>2</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>36</th>
          <td>0.615485</td>
          <td>0.5001</td>
          <td>0.397536</td>
          <td>${\bf{x}}_{\alpha \,{\bf{5}},5}$</td>
          <td>Overlying</td>
          <td>5</td>
          <td>False</td>
          <td>2</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>37</th>
          <td>0.653946</td>
          <td>0.5001</td>
          <td>0.384715</td>
          <td>${\bf{x}}_{\alpha \,{\bf{5}},6}$</td>
          <td>Overlying</td>
          <td>5</td>
          <td>False</td>
          <td>2</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>38</th>
          <td>0.692408</td>
          <td>0.5001</td>
          <td>0.371895</td>
          <td>${\bf{x}}_{\alpha \,{\bf{5}},7}$</td>
          <td>Overlying</td>
          <td>5</td>
          <td>False</td>
          <td>2</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>39</th>
          <td>0.666767</td>
          <td>0.5001</td>
          <td>0.320613</td>
          <td>${\bf{x}}_{\alpha \,{\bf{6}},0}$</td>
          <td>Layer 3</td>
          <td>6</td>
          <td>False</td>
          <td>3</td>
          <td>unco</td>
        </tr>
        <tr>
          <th>40</th>
          <td>0.512921</td>
          <td>0.5001</td>
          <td>0.320613</td>
          <td>${\bf{x}}_{\alpha \,{\bf{6}},1}$</td>
          <td>Layer 3</td>
          <td>6</td>
          <td>False</td>
          <td>3</td>
          <td>unco</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    lith_block, fault_block = gp.compute_model(interp_data)

This solution can be plot with the correspondent plotting function.
Blocks:

.. code:: ipython3

    %matplotlib inline
    gp.plot_section(geo_data, lith_block[0], 25, plot_data=True)



.. image:: ch1-Copy1_files/ch1-Copy1_33_0.png


.. code:: ipython3

    %matplotlib inline
    gp.plot_section(geo_data, fault_block[0], 25, plot_data=True)



.. image:: ch1-Copy1_files/ch1-Copy1_34_0.png


Potential field:

.. code:: ipython3

    gp.plot_potential_field(geo_data, lith_block[1], 25, N=500, cmap='viridis')



.. image:: ch1-Copy1_files/ch1-Copy1_36_0.png


From the potential fields (of lithologies and faults) it is possible to
extract vertices and simpleces to create the 3D triangles for a vtk
visualization.

.. code:: ipython3

    ver, sim = gp.get_surfaces(interp_data,lith_block[1], fault_block[1], original_scale=True)


::


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-49-b70d47564eaf> in <module>()
    ----> 1 ver, sim = gp.get_surfaces(interp_data,lith_block[1], fault_block[1], original_scale=True)
    

    ~/PycharmProjects/gempy/gempy/GemPy_f.py in get_surfaces(interp_data, potential_lith, potential_fault, n_formation, step_size, original_scale)
        580             else:
        581                 v, s = get_surface(potential_fault, interp_data, pot_int, n,
    --> 582                                    step_size=step_size, original_scale=original_scale)
        583                 vertices.append(v)
        584                 simplices.append(s)


    ~/PycharmProjects/gempy/gempy/GemPy_f.py in get_surface(potential_block, interp_data, pot_int, n_formation, step_size, original_scale)
        556             spacing=((interp_data.geo_data_res.extent[1] - interp_data.geo_data_res.extent[0]) / interp_data.geo_data_res.resolution[0],
        557                      (interp_data.geo_data_res.extent[3] - interp_data.geo_data_res.extent[2]) / interp_data.geo_data_res.resolution[1],
    --> 558                      (interp_data.geo_data_res.extent[5] - interp_data.geo_data_res.extent[4]) / interp_data.geo_data_res.resolution[2]))
        559 
        560 


    ~/anaconda3/lib/python3.6/site-packages/skimage/measure/_marching_cubes_lewiner.py in marching_cubes_lewiner(volume, level, spacing, gradient_direction, step_size, allow_degenerate, use_classic)
        197         level = float(level)
        198         if level < volume.min() or level > volume.max():
    --> 199             raise ValueError("Surface level must be within volume data range.")
        200     # spacing
        201     if len(spacing) != 3:


    ValueError: Surface level must be within volume data range.


.. code:: ipython3

    %debug


.. parsed-literal::

    > [0;32m/home/miguel/anaconda3/lib/python3.6/site-packages/skimage/measure/_marching_cubes_lewiner.py[0m(199)[0;36mmarching_cubes_lewiner[0;34m()[0m
    [0;32m    197 [0;31m        [0mlevel[0m [0;34m=[0m [0mfloat[0m[0;34m([0m[0mlevel[0m[0;34m)[0m[0;34m[0m[0m
    [0m[0;32m    198 [0;31m        [0;32mif[0m [0mlevel[0m [0;34m<[0m [0mvolume[0m[0;34m.[0m[0mmin[0m[0;34m([0m[0;34m)[0m [0;32mor[0m [0mlevel[0m [0;34m>[0m [0mvolume[0m[0;34m.[0m[0mmax[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0m
    [0m[0;32m--> 199 [0;31m            [0;32mraise[0m [0mValueError[0m[0;34m([0m[0;34m"Surface level must be within volume data range."[0m[0;34m)[0m[0;34m[0m[0m
    [0m[0;32m    200 [0;31m    [0;31m# spacing[0m[0;34m[0m[0;34m[0m[0m
    [0m[0;32m    201 [0;31m    [0;32mif[0m [0mlen[0m[0;34m([0m[0mspacing[0m[0;34m)[0m [0;34m!=[0m [0;36m3[0m[0;34m:[0m[0;34m[0m[0m
    [0m
    ipdb> up
    > [0;32m/home/miguel/PycharmProjects/gempy/gempy/GemPy_f.py[0m(558)[0;36mget_surface[0;34m()[0m
    [0;32m    556 [0;31m            spacing=((interp_data.geo_data_res.extent[1] - interp_data.geo_data_res.extent[0]) / interp_data.geo_data_res.resolution[0],
    [0m[0;32m    557 [0;31m                     [0;34m([0m[0minterp_data[0m[0;34m.[0m[0mgeo_data_res[0m[0;34m.[0m[0mextent[0m[0;34m[[0m[0;36m3[0m[0;34m][0m [0;34m-[0m [0minterp_data[0m[0;34m.[0m[0mgeo_data_res[0m[0;34m.[0m[0mextent[0m[0;34m[[0m[0;36m2[0m[0;34m][0m[0;34m)[0m [0;34m/[0m [0minterp_data[0m[0;34m.[0m[0mgeo_data_res[0m[0;34m.[0m[0mresolution[0m[0;34m[[0m[0;36m1[0m[0;34m][0m[0;34m,[0m[0;34m[0m[0m
    [0m[0;32m--> 558 [0;31m                     (interp_data.geo_data_res.extent[5] - interp_data.geo_data_res.extent[4]) / interp_data.geo_data_res.resolution[2]))
    [0m[0;32m    559 [0;31m[0;34m[0m[0m
    [0m[0;32m    560 [0;31m[0;34m[0m[0m
    [0m
    ipdb> pot_int[n_formation-1]
    90.703789
    ipdb> potential_block.max()
    95.060684
    ipdb> potential_block.min()
    90.841385
    ipdb> exit


.. code:: ipython3

    gp.plot_surfaces_3D(geo_data, ver, sim, alpha=1)

Additionally is possible to update the model and recompute the surfaces
in real time. To do so, we need to pass the data rescaled. To get an
smooth response is important to have the theano optimizer flag in
fast\_run and run theano in the gpu. This can speed up the modeling time
in a factor of 20.

.. code:: ipython3

    gp.get_data(interp_data.geo_data_res, verbosity=1)




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
          <th>annotations</th>
          <th>azimuth</th>
          <th>dip</th>
          <th>formation</th>
          <th>formation number</th>
          <th>isFault</th>
          <th>order_series</th>
          <th>polarity</th>
          <th>series</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th rowspan="43" valign="top">interfaces</th>
          <th>0</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.474459</td>
          <td>0.5001</td>
          <td>0.49369</td>
          <td>${\bf{x}}_{\alpha \,{\bf{1}},0}$</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>MainFault</td>
          <td>1</td>
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
          <td>0.423177</td>
          <td>0.5001</td>
          <td>0.339844</td>
          <td>${\bf{x}}_{\alpha \,{\bf{1}},1}$</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>MainFault</td>
          <td>1</td>
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
          <td>0.525741</td>
          <td>0.5001</td>
          <td>0.647536</td>
          <td>${\bf{x}}_{\alpha \,{\bf{1}},2}$</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>MainFault</td>
          <td>1</td>
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
          <td>0.5001</td>
          <td>0.5001</td>
          <td>0.570613</td>
          <td>${\bf{x}}_{\alpha \,{\bf{1}},3}$</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>MainFault</td>
          <td>1</td>
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
          <td>0.448818</td>
          <td>0.5001</td>
          <td>0.416767</td>
          <td>${\bf{x}}_{\alpha \,{\bf{1}},4}$</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>MainFault</td>
          <td>1</td>
          <td>True</td>
          <td>1</td>
          <td>NaN</td>
          <td>fault</td>
        </tr>
        <tr>
          <th>5</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.679587</td>
          <td>0.5001</td>
          <td>0.314203</td>
          <td>${\bf{x}}_{\alpha \,{\bf{2}},0}$</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>fault2</td>
          <td>2</td>
          <td>True</td>
          <td>2</td>
          <td>NaN</td>
          <td>Fault_serie2</td>
        </tr>
        <tr>
          <th>6</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.269331</td>
          <td>0.5001</td>
          <td>0.711638</td>
          <td>${\bf{x}}_{\alpha \,{\bf{2}},1}$</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>fault2</td>
          <td>2</td>
          <td>True</td>
          <td>2</td>
          <td>NaN</td>
          <td>Fault_serie2</td>
        </tr>
        <tr>
          <th>7</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.705228</td>
          <td>0.5001</td>
          <td>0.544972</td>
          <td>${\bf{x}}_{\alpha \,{\bf{3}},0}$</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>SecondaryReservoir</td>
          <td>3</td>
          <td>False</td>
          <td>3</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>8</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.628305</td>
          <td>0.5001</td>
          <td>0.570613</td>
          <td>${\bf{x}}_{\alpha \,{\bf{3}},1}$</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>SecondaryReservoir</td>
          <td>3</td>
          <td>False</td>
          <td>3</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>9</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.371895</td>
          <td>0.5001</td>
          <td>0.609074</td>
          <td>${\bf{x}}_{\alpha \,{\bf{3}},2}$</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>SecondaryReservoir</td>
          <td>3</td>
          <td>False</td>
          <td>3</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>10</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.551382</td>
          <td>0.5001</td>
          <td>0.596254</td>
          <td>${\bf{x}}_{\alpha \,{\bf{3}},3}$</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>SecondaryReservoir</td>
          <td>3</td>
          <td>False</td>
          <td>3</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>11</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.448818</td>
          <td>0.5001</td>
          <td>0.583433</td>
          <td>${\bf{x}}_{\alpha \,{\bf{3}},4}$</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>SecondaryReservoir</td>
          <td>3</td>
          <td>False</td>
          <td>3</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>12</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.730869</td>
          <td>0.5001</td>
          <td>0.551382</td>
          <td>${\bf{x}}_{\alpha \,{\bf{3}},5}$</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>SecondaryReservoir</td>
          <td>3</td>
          <td>False</td>
          <td>3</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>13</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.448818</td>
          <td>0.5001</td>
          <td>0.557792</td>
          <td>${\bf{x}}_{\alpha \,{\bf{4}},0}$</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Seal</td>
          <td>4</td>
          <td>False</td>
          <td>3</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>14</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.730869</td>
          <td>0.5001</td>
          <td>0.525741</td>
          <td>${\bf{x}}_{\alpha \,{\bf{4}},1}$</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Seal</td>
          <td>4</td>
          <td>False</td>
          <td>3</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>15</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.371895</td>
          <td>0.5001</td>
          <td>0.583433</td>
          <td>${\bf{x}}_{\alpha \,{\bf{4}},2}$</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Seal</td>
          <td>4</td>
          <td>False</td>
          <td>3</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>16</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.551382</td>
          <td>0.5001</td>
          <td>0.570613</td>
          <td>${\bf{x}}_{\alpha \,{\bf{4}},3}$</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Seal</td>
          <td>4</td>
          <td>False</td>
          <td>3</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>17</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.628305</td>
          <td>0.5001</td>
          <td>0.544972</td>
          <td>${\bf{x}}_{\alpha \,{\bf{4}},4}$</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Seal</td>
          <td>4</td>
          <td>False</td>
          <td>3</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>18</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.705228</td>
          <td>0.5001</td>
          <td>0.519331</td>
          <td>${\bf{x}}_{\alpha \,{\bf{4}},5}$</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Seal</td>
          <td>4</td>
          <td>False</td>
          <td>3</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>19</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.423177</td>
          <td>0.294972</td>
          <td>0.391126</td>
          <td>${\bf{x}}_{\alpha \,{\bf{5}},0}$</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Reservoir</td>
          <td>5</td>
          <td>False</td>
          <td>3</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>20</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.423177</td>
          <td>0.705228</td>
          <td>0.391126</td>
          <td>${\bf{x}}_{\alpha \,{\bf{5}},1}$</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Reservoir</td>
          <td>5</td>
          <td>False</td>
          <td>3</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>21</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.364376</td>
          <td>0.504601</td>
          <td>0.481594</td>
          <td>${\bf{x}}_{\alpha \,{\bf{5}},2}$</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Reservoir</td>
          <td>5</td>
          <td>False</td>
          <td>3</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>22</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.294972</td>
          <td>0.5001</td>
          <td>0.50651</td>
          <td>${\bf{x}}_{\alpha \,{\bf{5}},3}$</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Reservoir</td>
          <td>5</td>
          <td>False</td>
          <td>3</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>23</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.730869</td>
          <td>0.5001</td>
          <td>0.423177</td>
          <td>${\bf{x}}_{\alpha \,{\bf{5}},4}$</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Reservoir</td>
          <td>5</td>
          <td>False</td>
          <td>3</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>24</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.705228</td>
          <td>0.5001</td>
          <td>0.416767</td>
          <td>${\bf{x}}_{\alpha \,{\bf{5}},5}$</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Reservoir</td>
          <td>5</td>
          <td>False</td>
          <td>3</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>25</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.551382</td>
          <td>0.5001</td>
          <td>0.468049</td>
          <td>${\bf{x}}_{\alpha \,{\bf{5}},6}$</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Reservoir</td>
          <td>5</td>
          <td>False</td>
          <td>3</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>26</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.628305</td>
          <td>0.5001</td>
          <td>0.442408</td>
          <td>${\bf{x}}_{\alpha \,{\bf{5}},7}$</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Reservoir</td>
          <td>5</td>
          <td>False</td>
          <td>3</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>27</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.666767</td>
          <td>0.5001</td>
          <td>0.429587</td>
          <td>${\bf{x}}_{\alpha \,{\bf{5}},8}$</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Reservoir</td>
          <td>5</td>
          <td>False</td>
          <td>3</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>28</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.474459</td>
          <td>0.730869</td>
          <td>0.468049</td>
          <td>${\bf{x}}_{\alpha \,{\bf{5}},9}$</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Reservoir</td>
          <td>5</td>
          <td>False</td>
          <td>3</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>29</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.512921</td>
          <td>0.5001</td>
          <td>0.480869</td>
          <td>${\bf{x}}_{\alpha \,{\bf{5}},10}$</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Reservoir</td>
          <td>5</td>
          <td>False</td>
          <td>3</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>30</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.474459</td>
          <td>0.269331</td>
          <td>0.468049</td>
          <td>${\bf{x}}_{\alpha \,{\bf{5}},11}$</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Reservoir</td>
          <td>5</td>
          <td>False</td>
          <td>3</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>31</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.474459</td>
          <td>0.7501</td>
          <td>0.480869</td>
          <td>${\bf{x}}_{\alpha \,{\bf{5}},12}$</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Reservoir</td>
          <td>5</td>
          <td>False</td>
          <td>3</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>32</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.474459</td>
          <td>0.2501</td>
          <td>0.480869</td>
          <td>${\bf{x}}_{\alpha \,{\bf{5}},13}$</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Reservoir</td>
          <td>5</td>
          <td>False</td>
          <td>3</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>33</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.474459</td>
          <td>0.5001</td>
          <td>0.416767</td>
          <td>${\bf{x}}_{\alpha \,{\bf{6}},0}$</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Overlying</td>
          <td>6</td>
          <td>False</td>
          <td>3</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>34</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.730869</td>
          <td>0.5001</td>
          <td>0.346254</td>
          <td>${\bf{x}}_{\alpha \,{\bf{6}},1}$</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Overlying</td>
          <td>6</td>
          <td>False</td>
          <td>3</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>35</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.628305</td>
          <td>0.5001</td>
          <td>0.365485</td>
          <td>${\bf{x}}_{\alpha \,{\bf{6}},2}$</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Overlying</td>
          <td>6</td>
          <td>False</td>
          <td>3</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>36</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.666767</td>
          <td>0.5001</td>
          <td>0.352664</td>
          <td>${\bf{x}}_{\alpha \,{\bf{6}},3}$</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Overlying</td>
          <td>6</td>
          <td>False</td>
          <td>3</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>37</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.705228</td>
          <td>0.5001</td>
          <td>0.339844</td>
          <td>${\bf{x}}_{\alpha \,{\bf{6}},4}$</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Overlying</td>
          <td>6</td>
          <td>False</td>
          <td>3</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>38</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.551382</td>
          <td>0.5001</td>
          <td>0.391126</td>
          <td>${\bf{x}}_{\alpha \,{\bf{6}},5}$</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Overlying</td>
          <td>6</td>
          <td>False</td>
          <td>3</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>39</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.371895</td>
          <td>0.5001</td>
          <td>0.403946</td>
          <td>${\bf{x}}_{\alpha \,{\bf{6}},6}$</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Overlying</td>
          <td>6</td>
          <td>False</td>
          <td>3</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>40</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.294972</td>
          <td>0.5001</td>
          <td>0.429587</td>
          <td>${\bf{x}}_{\alpha \,{\bf{6}},7}$</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Overlying</td>
          <td>6</td>
          <td>False</td>
          <td>3</td>
          <td>NaN</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>41</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.679587</td>
          <td>0.5001</td>
          <td>0.288562</td>
          <td>${\bf{x}}_{\alpha \,{\bf{7}},0}$</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Layer 3</td>
          <td>7</td>
          <td>False</td>
          <td>4</td>
          <td>NaN</td>
          <td>unco</td>
        </tr>
        <tr>
          <th>42</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.525741</td>
          <td>0.5001</td>
          <td>0.339844</td>
          <td>${\bf{x}}_{\alpha \,{\bf{7}},1}$</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Layer 3</td>
          <td>7</td>
          <td>False</td>
          <td>4</td>
          <td>NaN</td>
          <td>unco</td>
        </tr>
        <tr>
          <th rowspan="4" valign="top">foliations</th>
          <th>0</th>
          <td>-0.948683</td>
          <td>-1.7427e-16</td>
          <td>0.316229</td>
          <td>0.453292</td>
          <td>0.5001</td>
          <td>0.458972</td>
          <td>${\bf{x}}_{\beta \,{\bf{1}},0}$</td>
          <td>270</td>
          <td>71.565</td>
          <td>MainFault</td>
          <td>1</td>
          <td>True</td>
          <td>1</td>
          <td>1</td>
          <td>fault</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.642788</td>
          <td>3.93594e-17</td>
          <td>0.766044</td>
          <td>0.577023</td>
          <td>0.5001</td>
          <td>0.416767</td>
          <td>${\bf{x}}_{\beta \,{\bf{2}},0}$</td>
          <td>90</td>
          <td>40</td>
          <td>fault2</td>
          <td>2</td>
          <td>True</td>
          <td>2</td>
          <td>1</td>
          <td>Fault_serie2</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.316229</td>
          <td>1.93634e-17</td>
          <td>0.948683</td>
          <td>0.589844</td>
          <td>0.5001</td>
          <td>0.455228</td>
          <td>${\bf{x}}_{\beta \,{\bf{5}},0}$</td>
          <td>90</td>
          <td>18.435</td>
          <td>Reservoir</td>
          <td>5</td>
          <td>False</td>
          <td>3</td>
          <td>1</td>
          <td>Rest</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.34202</td>
          <td>2.09427e-17</td>
          <td>0.939693</td>
          <td>0.602664</td>
          <td>0.5001</td>
          <td>0.391126</td>
          <td>${\bf{x}}_{\beta \,{\bf{7}},0}$</td>
          <td>90</td>
          <td>20</td>
          <td>Layer 3</td>
          <td>7</td>
          <td>False</td>
          <td>4</td>
          <td>1</td>
          <td>unco</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    ver_s, sim_s = gp.get_surfaces(interp_data,lith_block[1],
                                   fault_block[1],
                                   original_scale=False)


::


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-50-88ef9c3214df> in <module>()
          1 ver_s, sim_s = gp.get_surfaces(interp_data,lith_block[1],
          2                                fault_block[1],
    ----> 3                                original_scale=False)
    

    ~/PycharmProjects/gempy/gempy/GemPy_f.py in get_surfaces(interp_data, potential_lith, potential_fault, n_formation, step_size, original_scale)
        580             else:
        581                 v, s = get_surface(potential_fault, interp_data, pot_int, n,
    --> 582                                    step_size=step_size, original_scale=original_scale)
        583                 vertices.append(v)
        584                 simplices.append(s)


    ~/PycharmProjects/gempy/gempy/GemPy_f.py in get_surface(potential_block, interp_data, pot_int, n_formation, step_size, original_scale)
        556             spacing=((interp_data.geo_data_res.extent[1] - interp_data.geo_data_res.extent[0]) / interp_data.geo_data_res.resolution[0],
        557                      (interp_data.geo_data_res.extent[3] - interp_data.geo_data_res.extent[2]) / interp_data.geo_data_res.resolution[1],
    --> 558                      (interp_data.geo_data_res.extent[5] - interp_data.geo_data_res.extent[4]) / interp_data.geo_data_res.resolution[2]))
        559 
        560 


    ~/anaconda3/lib/python3.6/site-packages/skimage/measure/_marching_cubes_lewiner.py in marching_cubes_lewiner(volume, level, spacing, gradient_direction, step_size, allow_degenerate, use_classic)
        197         level = float(level)
        198         if level < volume.min() or level > volume.max():
    --> 199             raise ValueError("Surface level must be within volume data range.")
        200     # spacing
        201     if len(spacing) != 3:


    ValueError: Surface level must be within volume data range.


.. code:: ipython3

    gp.plot_surfaces_3D_real_time(interp_data, ver_s, sim_s)

In the same manner we can visualize the fault block:

.. code:: ipython3

    gp.plot_section(geo_data, fault_block[0], 25)



.. image:: ch1-Copy1_files/ch1-Copy1_46_0.png


.. code:: ipython3

    gp.plot_potential_field(geo_data, fault_block[1], 25)



.. image:: ch1-Copy1_files/ch1-Copy1_47_0.png

