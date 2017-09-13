
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

With the command get data is possible to see all the input data.
However, at the moment the (depositional) order of the formation and the
separation of the series (More explanation about this in the next
notebook) is totally arbitrary.

.. code:: ipython3

    gp.get_data(geo_data).head()




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
          <th rowspan="5" valign="top">interfaces</th>
          <th>0</th>
          <td>800</td>
          <td>200</td>
          <td>-1400</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Reservoir</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>1</th>
          <td>800</td>
          <td>1800</td>
          <td>-1400</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Reservoir</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>2</th>
          <td>600</td>
          <td>1000</td>
          <td>-1050</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Reservoir</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>3</th>
          <td>300</td>
          <td>1000</td>
          <td>-950</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Reservoir</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>4</th>
          <td>2000</td>
          <td>1000</td>
          <td>-1275</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Reservoir</td>
          <td>NaN</td>
          <td>Default serie</td>
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
                          "Rest": ('SecondaryReservoir','Seal', 'Reservoir', 'Overlying')},
                           order_series = ["fault", 'Rest'],
                           order_formations=['MainFault', 
                                             'SecondaryReservoir', 'Seal','Reservoir', 'Overlying',
                                             ]) 




.. parsed-literal::

    <gempy.strat_pile.StratigraphicPile at 0x7fc4b52bff98>




.. image:: ch1_files/ch1_7_1.png


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

    <gempy.strat_pile.StratigraphicPile at 0x7fc4b51316a0>



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
        </tr>
        <tr>
          <th>1</th>
          <td>800</td>
          <td>1000</td>
          <td>-1600</td>
          <td>MainFault</td>
          <td>fault</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1200</td>
          <td>1000</td>
          <td>-400</td>
          <td>MainFault</td>
          <td>fault</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1100</td>
          <td>1000</td>
          <td>-700</td>
          <td>MainFault</td>
          <td>fault</td>
        </tr>
        <tr>
          <th>4</th>
          <td>900</td>
          <td>1000</td>
          <td>-1300</td>
          <td>MainFault</td>
          <td>fault</td>
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
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>917.45</td>
          <td>1000</td>
          <td>-1135.4</td>
          <td>71.565</td>
          <td>270</td>
          <td>1</td>
          <td>MainFault</td>
          <td>fault</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1450</td>
          <td>1000</td>
          <td>-1150</td>
          <td>18.435</td>
          <td>90</td>
          <td>1</td>
          <td>Reservoir</td>
          <td>Rest</td>
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



.. image:: ch1_files/ch1_19_0.png


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

    interp_data = gp.InterpolatorInput(geo_data, u_grade=[3,3])
    print(interp_data)


.. parsed-literal::

    Level of Optimization:  fast_run
    Device:  cpu
    Precision:  float32
    <gempy.DataManagement.InterpolatorInput object at 0x7fc4b505e2e8>


.. code:: ipython3

    interp_data.get_formation_number()




.. parsed-literal::

    {'DefaultBasement': 0,
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

    range 0.8731347322463989 3464.10165466
    Number of drift equations [2 2]
    Covariance at 0 0.018151529133319855
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

    lith_block, fault_block = gp.compute_model(interp_data)

This solution can be plot with the correspondent plotting function.
Blocks:

.. code:: ipython3

    %matplotlib inline
    gp.plot_section(geo_data, lith_block[0], 25, plot_data=True)



.. image:: ch1_files/ch1_30_0.png


Potential field:

.. code:: ipython3

    gp.plot_potential_field(geo_data, lith_block[1], 25)



.. image:: ch1_files/ch1_32_0.png


From the potential fields (of lithologies and faults) it is possible to
extract vertices and simpleces to create the 3D triangles for a vtk
visualization.

.. code:: ipython3

    ver, sim = gp.get_surfaces(interp_data,lith_block[1], fault_block[1], original_scale=True)

.. code:: ipython3

    gp.plot_surfaces_3D(geo_data, ver, sim, alpha=1)

Additionally is possible to update the model and recompute the surfaces
in real time. To do so, we need to pass the data rescaled. To get an
smooth response is important to have the theano optimizer flag in
fast\_run and run theano in the gpu. This can speed up the modeling time
in a factor of 20.

.. code:: ipython3

    ver_s, sim_s = gp.get_surfaces(interp_data,lith_block[1],
                                   fault_block[1],
                                   original_scale=False)

.. code:: ipython3

    gp.plot_surfaces_3D_real_time(interp_data, ver_s, sim_s)

In the same manner we can visualize the fault block:

.. code:: ipython3

    gp.plot_section(geo_data, fault_block[0], 25)



.. image:: ch1_files/ch1_40_0.png


.. code:: ipython3

    gp.plot_potential_field(geo_data, fault_block[1], 25)



.. image:: ch1_files/ch1_41_0.png

