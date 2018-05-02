
Chapter 1: Basics of geological modeling with GemPy
===================================================

--------------

In this first chapter, we will guide through the most important steps of
modeling with GemPy on the base of a relatively simple geological model,
while introducing essential objects and functions. We will illustrate
how to: - import and create input data for modeling in GemPy - return
and visualize input data - generate a 3D geological model in GemPy -
visualize a model directly in GemPy \*\*\*

The example model: Simple stratigraphy and one fault
----------------------------------------------------

Our synthetic example model is defined to be cubic, with an extent of
2000 m in every direction of the 3D space. Lithologically, it includes
five stratigraphic units of sedimentary origin. Here, we list them from
top (youngest) to bottom (oldest):

-  Sandstone (2)
-  Siltstone
-  Shale
-  Sandstone (1)
-  Basement (undefined, default by GemPy)

We assume that these were simply deposited in consequential order and
deformed (tilted and folded) afterwards. Additionally, they are
displaced by a continuous normal fault. The final modeling results
should look somewhat like this, depending on the type of visualization:

.. figure:: ../../readme_images/model_example_duo.png
   :alt: 2D and 3D visualizations of our example model

   2D and 3D visualizations of our example model.

As this example involves a simple sequence of layers and only one fault,
it provides an adequate level of complexity to introduce the basics of
modeling with GemPy. At the end of this chapter, we will show some model
variations and how the modeling workflow has to be adapted accordingly.

Preparing the Python environment
--------------------------------

For modeling with GemPy, we first need to import it. We should also
import any other packages we want to utilize in our Python
environment.Typically, we will also require ``NumPy`` and ``Matplotlib``
when working with GemPy. At this point, we can further customize some
settings as desired, e.g. the size of figures or, as we do here, the way
that ``Matplotlib`` figures are displayed in our notebook
(``%matplotlib inline``).

.. code:: ipython3

    # These two lines are necessary only if GemPy is not installed
    import sys, os
    sys.path.append("../..")
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # Importing GemPy
    import gempy as gp
    
    # Embedding matplotlib figures in the notebooks
    %matplotlib inline
    
    # Importing auxiliary libraries
    import numpy as np
    import matplotlib.pyplot as plt


.. parsed-literal::

    WARNING (theano.tensor.blas): Using NumPy C-API based implementation for BLAS functions.


Importing and creating a set of input data
------------------------------------------

The data used for the construction of a model in GemPy is stored in a
Python object called ``InputData``.

In principle, the input data can be stored in the form of a Python
`pickle <https://docs.python.org/3/library/pickle.html>`__. However,
module version consistency is required. For loading a pickle into GemPy,
you have to make sure that you are using the same version of pickle and
dependent modules (e.g.: ``Pandas``, ``NumPy``) as were used when the
data was originally stored.

``InputData`` can also be generated from raw data that comes in the form
of CSV-files (CSV = comma-separated values). Such files might be
attained by exporting model data from a different program such as
GeoModeller or by simply creating it in spreadsheet software such as
Microsoft Excel or LibreOffice Calc.

In this tutorial, all input data is created by importing such CSV-files.
These exemplary files can be found in the ``input_data`` folder in the
root folder of GemPy. The data comprises :math:`x`-, :math:`y`- and
:math:`z`-positional values for all surface points and orientation
measurements. For the latter, poles, azimuth and polarity are
additionally included. Surface points are furthermore assigned a
formation. This might be a lithological unit such as "Sandstone" or a
structural feature such as "Main Fault". It is decisive to remember
that, in GemPy, interface position points mark the **bottom** of a
layer. If such points are needed to resemble a top of a formation (e.g.
when modeling an intrusion), this can be achieved by defining a
respectively inverted orientation measurement.

As we generate our ``InputData`` from CSV-files, we also have to define
our model's real extent in :math:`x`, :math:`y` and :math:`z`, as well
as declare a desired resolution for each axis. This resolution will in
turn determine the number of voxels used during modeling. Here, we rely
on a medium resolution of 50x50x50, amounting to 125,000 voxels. The
model extent should be chosen in a way that it contains all relevant
data in a representative space. As our model voxels are not cubes, but
prisms, the resolution can take a different shape than the extent. We
don't recommend going much higher than 100 cells in every direction
(1,000,000 voxels), as higher resolutions will become increasingly
difficult to compute.

.. code:: ipython3

    # Importing the data from CSV-files and setting extent and resolution
    geo_data = gp.create_data([0,2000,0,2000,0,2000],[100,100,100], 
                              path_o = os.pardir+"/input_data/tut_chapter1/simple_fault_model_orientations.csv", # importing orientation (foliation) data
                              path_i = os.pardir+"/input_data/tut_chapter1/simple_fault_model_points.csv") # importing point-positional interface data


The input data can then be listed using the command ``get_data``. Note
that the order of formations and respective allocation to series is
still completely arbitrary. We will fix this in the following.

.. code:: ipython3

    gp.get_data(geo_data)




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
          <th>G_x</th>
          <th>G_y</th>
          <th>G_z</th>
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
          <th rowspan="57" valign="top">interfaces</th>
          <th>0</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>800</td>
          <td>50</td>
          <td>750</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Shale</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>1</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>800</td>
          <td>150</td>
          <td>700</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Shale</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>2</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>800</td>
          <td>300</td>
          <td>700</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Shale</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>3</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>800</td>
          <td>500</td>
          <td>800</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Shale</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>4</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>800</td>
          <td>1000</td>
          <td>1000</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Shale</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>5</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>800</td>
          <td>1500</td>
          <td>700</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Shale</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>6</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>800</td>
          <td>1700</td>
          <td>600</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Shale</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>7</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>800</td>
          <td>1950</td>
          <td>650</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Shale</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>8</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0</td>
          <td>1000</td>
          <td>1100</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Shale</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>9</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>300</td>
          <td>1000</td>
          <td>1000</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Shale</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>10</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>450</td>
          <td>1000</td>
          <td>950</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Shale</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>11</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>1100</td>
          <td>1000</td>
          <td>900</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Shale</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>12</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>1400</td>
          <td>1000</td>
          <td>850</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Shale</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>13</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>1700</td>
          <td>1000</td>
          <td>900</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Shale</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>14</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>1500</td>
          <td>500</td>
          <td>800</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Shale</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>15</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>1500</td>
          <td>1500</td>
          <td>750</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Shale</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>16</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>1500</td>
          <td>1500</td>
          <td>450</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Sandstone_1</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>17</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>1500</td>
          <td>500</td>
          <td>500</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Sandstone_1</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>18</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>1700</td>
          <td>1000</td>
          <td>600</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Sandstone_1</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>19</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>1100</td>
          <td>1000</td>
          <td>600</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Sandstone_1</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>20</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>800</td>
          <td>50</td>
          <td>450</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Sandstone_1</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>21</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>800</td>
          <td>150</td>
          <td>400</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Sandstone_1</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>22</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>800</td>
          <td>300</td>
          <td>400</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Sandstone_1</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>23</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>800</td>
          <td>500</td>
          <td>500</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Sandstone_1</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>24</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>800</td>
          <td>1000</td>
          <td>700</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Sandstone_1</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>25</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>800</td>
          <td>1500</td>
          <td>400</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Sandstone_1</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>26</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>800</td>
          <td>1700</td>
          <td>300</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Sandstone_1</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>27</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>800</td>
          <td>1950</td>
          <td>350</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Sandstone_1</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>28</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0</td>
          <td>1000</td>
          <td>800</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Sandstone_1</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>29</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>300</td>
          <td>1000</td>
          <td>700</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Sandstone_1</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>30</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>1400</td>
          <td>1000</td>
          <td>550</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Sandstone_1</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>31</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>900</td>
          <td>150</td>
          <td>920</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Siltstone</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>32</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>900</td>
          <td>300</td>
          <td>920</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Siltstone</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>33</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>900</td>
          <td>1500</td>
          <td>920</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Siltstone</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>34</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>900</td>
          <td>1700</td>
          <td>820</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Siltstone</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>35</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>900</td>
          <td>1950</td>
          <td>870</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Siltstone</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>36</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0</td>
          <td>1000</td>
          <td>1300</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Siltstone</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>37</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>300</td>
          <td>1000</td>
          <td>1200</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Siltstone</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>38</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>600</td>
          <td>1000</td>
          <td>1100</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Siltstone</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>39</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>1100</td>
          <td>1000</td>
          <td>1100</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Siltstone</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>40</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>1400</td>
          <td>1000</td>
          <td>1050</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Siltstone</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>41</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>1700</td>
          <td>1000</td>
          <td>1100</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Siltstone</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>42</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>1500</td>
          <td>500</td>
          <td>1000</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Siltstone</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>43</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>1500</td>
          <td>1500</td>
          <td>950</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Siltstone</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>44</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0</td>
          <td>1000</td>
          <td>1500</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Sandstone_2</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>45</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>300</td>
          <td>1000</td>
          <td>1400</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Sandstone_2</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>46</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>600</td>
          <td>1000</td>
          <td>1300</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Sandstone_2</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>47</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>1100</td>
          <td>1000</td>
          <td>1300</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Sandstone_2</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>48</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>1400</td>
          <td>1000</td>
          <td>1250</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Sandstone_2</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>49</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>1700</td>
          <td>1000</td>
          <td>1300</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Sandstone_2</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>50</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>1500</td>
          <td>500</td>
          <td>1200</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Sandstone_2</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>51</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>1500</td>
          <td>1500</td>
          <td>1150</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Sandstone_2</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>52</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>700</td>
          <td>1000</td>
          <td>900</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Main_Fault</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>53</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>600</td>
          <td>1000</td>
          <td>600</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Main_Fault</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>54</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>500</td>
          <td>1000</td>
          <td>300</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Main_Fault</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>55</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>800</td>
          <td>1000</td>
          <td>1200</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Main_Fault</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>56</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>900</td>
          <td>1000</td>
          <td>1500</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Main_Fault</td>
          <td>NaN</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th rowspan="3" valign="top">orientations</th>
          <th>0</th>
          <td>0.316229</td>
          <td>1e-07</td>
          <td>0.948683</td>
          <td>1000</td>
          <td>1000</td>
          <td>950</td>
          <td>90</td>
          <td>18.435</td>
          <td>Shale</td>
          <td>1</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.316229</td>
          <td>1e-07</td>
          <td>0.948683</td>
          <td>400</td>
          <td>1000</td>
          <td>1400</td>
          <td>90</td>
          <td>18.435</td>
          <td>Sandstone_2</td>
          <td>1</td>
          <td>Default serie</td>
        </tr>
        <tr>
          <th>2</th>
          <td>-0.948683</td>
          <td>1e-07</td>
          <td>0.316229</td>
          <td>500</td>
          <td>1000</td>
          <td>864.602</td>
          <td>270</td>
          <td>71.565</td>
          <td>Main_Fault</td>
          <td>1</td>
          <td>Default serie</td>
        </tr>
      </tbody>
    </table>
    </div>



Declaring the sequential order of geological formations
-------------------------------------------------------

We want our geological units to appear in the correct order relative to
age. Such order might for example be given by a depositional sequence of
stratigraphy, unconformities due to erosion or other lithological
genesis events such as igneous intrusions. A similar age-related order
is to be declared for the faults in our model. In GemPy, the function
*set\_series* is used to assign formations to different sequential
series via declaration in a Python dictionary.

Defining the correct order of series is vital to the construction of the
model! If you are using Python 3.6, the age-related order will already
be defined by the order of key entries, i.e. the first entry is the
youngest series, the last one the oldest. For older versions of Python,
you will have to specify the correct order as a separate list attribute
"*order\_series*" (see cell below).

You can assign several formations to one series. The order of the units
within such as series is only relevant for the color code, thus we
recommend to be consistent. You can define this order via another
attribute "*order\_formations*" or by using the specific command
*set\_order\_formations*. (If the order of the pile differs from the
final result the color of the interfaces and input data will be
different. ?)

Every fault is treated as an independent series and have to be at set at
the **top of the pile**. The relative order between the distinct faults
defines the tectonic relation between them (first entry is the
youngest).

In a model with simple sequential stratigraphy, all layer formations can
be assigned to one single series without a problem. All unit boundaries
and their order would then be given by interface points. However, to
model more complex lithostratigraphical relations and interactions, the
definition of separate series becomes important. For example, you would
need to declare a "newer" series to model an unconformity or an
intrusion that disturbs older stratigraphy.

Our example model comprises four main layers (plus an underlying
basement that is automatically generated by GemPy) and one main normal
fault displacing those layers. Assuming a simple stratigraphy where each
younger unit was deposited onto the underlying older one, we can assign
these layer formations to one series called "Strat\_Series". For the
fault, we declare a respective "Fault\_Series" as the first key entry in
the ``set_series`` dictionary. We could give any other names to these
series, the formations however have to be referred to as named in the
input data.

.. code:: ipython3

    # Assigning series to formations as well as their order (timewise)
    gp.set_series(geo_data, {"Fault_Series":'Main_Fault', 
                          "Strat_Series": ('Sandstone_2','Siltstone', 'Shale', 'Sandstone_1')},
                           order_series = ["Fault_Series", 'Strat_Series'],
                           order_formations=['Main_Fault', 
                                             'Sandstone_2','Siltstone', 'Shale', 'Sandstone_1',
                                             ], verbose=0) 
    
    # unconformity model:
    #gp.set_series(geo_data, {"Fault_Series":'Main_Fault', "Unconf_Series":'Carbonate',
    #                      "Strat_Series": ('Sandstone_2','Siltstone', 'Shale', 'Sandstone_1')},
    #                       order_series = ["Fault_Series", "Unconf_Series", 'Strat_Series'],
    #                       order_formations=['Main_Fault', 'Carbonate',
    #                                         'Sandstone_2','Siltstone', 'Shale', 'Sandstone_1',
    #                                         ], verbose=0) 

The sequence of geoligical series and assigned formations can be
visualized using the function ``get_sequential_pile``. Using a backend
such as ``%matplotlib notebook`` or ``%matplotlib qt5``, the figure
generated by this function becomes interactive, i.e. you can change the
order of series and formations by hand (via ``%matplotlib inline``, the
figure remains static). You can also re-assign a formation to a
different series.

If the backend doen't seem to work properly right away, try executing
the cell twice.

.. code:: ipython3

    %matplotlib inline
    gp.get_sequential_pile(geo_data)




.. parsed-literal::

    <gempy.sequential_pile.StratigraphicPile at 0x7f3eafc59898>




.. image:: ch1_files/ch1_9_1.png


Notice that the colors are order-dependent, i.e. they will remain in the
same order every time the cell is executed, irrespective of
re-assignment of formations. To make sure that every unit is in its
right place, take a look at the legend on the right. If it doesn't show
at first, try dragging the right edge to resize the figure. The legend
will always show you the color currently assigned to a formation (in the
future, every color will have the annotation within its rectangle to
avoid confusion).

Returning information from our input data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Our model input data, here named "*geo\_data*", contains all the
information that is essential for the construction of our model. You can
access different types of information by using correspondent "*get*"
functions.

We can, for example, return the coordinates of our modeling grid via
*get\_grid*:

.. code:: ipython3

    print(gp.get_grid(geo_data))


.. parsed-literal::

    [[   10.    10.    10.]
     [   10.    10.    30.]
     [   10.    10.    50.]
     ..., 
     [ 1990.  1990.  1950.]
     [ 1990.  1990.  1970.]
     [ 1990.  1990.  1990.]]


As mentioned before, GemPy's core algorithm is based on interpolation of
two types of data: - interface (or surface) points and - orientation
measurements

(if you want to know more on how this this interpolation algorithm
works, checkout our chapter on the theory behind GemPy).

We introduced the function *get\_data* above. You can also specify which
kind of data you want to call, by declaring the string attribute
"*dtype*" to be either "interfaces" (surface points) or "foliations"
(orientation measurements).

Interfaces Dataframe:
^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    gp.get_data(geo_data, 'interfaces').head()




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
          <th>formation</th>
          <th>series</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>900</td>
          <td>1000</td>
          <td>1500</td>
          <td>Main_Fault</td>
          <td>Fault_Series</td>
        </tr>
        <tr>
          <th>1</th>
          <td>700</td>
          <td>1000</td>
          <td>900</td>
          <td>Main_Fault</td>
          <td>Fault_Series</td>
        </tr>
        <tr>
          <th>2</th>
          <td>800</td>
          <td>1000</td>
          <td>1200</td>
          <td>Main_Fault</td>
          <td>Fault_Series</td>
        </tr>
        <tr>
          <th>3</th>
          <td>500</td>
          <td>1000</td>
          <td>300</td>
          <td>Main_Fault</td>
          <td>Fault_Series</td>
        </tr>
        <tr>
          <th>4</th>
          <td>600</td>
          <td>1000</td>
          <td>600</td>
          <td>Main_Fault</td>
          <td>Fault_Series</td>
        </tr>
      </tbody>
    </table>
    </div>



Foliations Dataframe:
^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    gp.get_data(geo_data, 'orientations')




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
          <td>1000</td>
          <td>864.602</td>
          <td>-0.948683</td>
          <td>1e-07</td>
          <td>0.316229</td>
          <td>71.565</td>
          <td>270</td>
          <td>1</td>
          <td>Main_Fault</td>
          <td>Fault_Series</td>
        </tr>
        <tr>
          <th>2</th>
          <td>400</td>
          <td>1000</td>
          <td>1400</td>
          <td>0.316229</td>
          <td>1e-07</td>
          <td>0.948683</td>
          <td>18.435</td>
          <td>90</td>
          <td>1</td>
          <td>Sandstone_2</td>
          <td>Strat_Series</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1000</td>
          <td>1000</td>
          <td>950</td>
          <td>0.316229</td>
          <td>1e-07</td>
          <td>0.948683</td>
          <td>18.435</td>
          <td>90</td>
          <td>1</td>
          <td>Shale</td>
          <td>Strat_Series</td>
        </tr>
      </tbody>
    </table>
    </div>



Notice that now all **formations** have been assigned to a **series**
and are displayed in the correct order (from young to old).

Visualizing input data
~~~~~~~~~~~~~~~~~~~~~~

We can also visualize our input data. This might for example be useful
to check if all points and measurements are defined the way we want them
to. Using the function *plot\_data*, we attain a 2D projection of our
data points onto a plane of chosen *direction* (we can choose this
attribute to be either :math:`x`, :math:`y` or :math:`z`).

.. code:: ipython3

    %matplotlib inline
    gp.plot_data(geo_data, direction='y')



.. image:: ch1_files/ch1_18_0.png


Using *plot\_data\_3D*, we can also visualize this data in 3D. Note that
direct 3D visualization in GemPy requires `the Visualization
Toolkit <https://www.vtk.org/>`__ (VTK) to be installed.

All 3D VTK plots in GemPy are interactive. This means that we can drag
and drop any data poit and measurement. The perpendicular axis views in
VTK are particularly useful to move points solely on a desired 2D plane.
Any changes will then be stored permanently in the "InputData"
dataframe. If we want to reset our data points, we will then need to
reload our original input data.

Executing the cell below will open a new window with a 3D interactive
plot of our data.

.. code:: ipython3

    gp.plot_data_3D(geo_data)

Model generation
~~~~~~~~~~~~~~~~

Once we have made sure that we have defined all our primary information
as desired in our object ``DataManagement.InputData`` (named
``geo_data`` in these tutorials), we can continue with the next step
towards creating our geological model: preparing the input data for
interpolation.

This is done by generating an ``InterpolatorInput`` object (named
``interp_data`` in these tutorials) from our ``InputData`` object via
the following function:

.. code:: ipython3

    interp_data = gp.InterpolatorData(geo_data, u_grade=[1,1], output='geology', compile_theano=True, theano_optimizer='fast_compile')
    #print(interp_data)


.. parsed-literal::

    Compiling theano function...
    Compilation Done!
    Level of Optimization:  fast_compile
    Device:  cpu
    Precision:  float32
    Number of faults:  1


This function rescales the extent and coordinates of the original data
and adds mathematical parameters that are needed for conducting the
interpolation. The computation of this step may take a while, as it also
compiles a theano function which is required for the model computation.
However, should this not be needed, we can skip it by declaring
``compile_theano = False`` in the function.

Furthermore, this preparation process includes an assignment of numbers
to each formation. Note that GemPy's always creates a default basement
formation as "0". Afterwards, numbers are allocated from youngest to
oldest as defined by the sequence of series and formations. Using
``get_formation_number()`` on our interpolation data, we can find out
which number has been assigned to which formation:

.. code:: ipython3

    interp_data.geo_data_res.get_formation_number()




.. parsed-literal::

    {'DefaultBasement': 0,
     'Main_Fault': 1,
     'Sandstone_1': 5,
     'Sandstone_2': 2,
     'Shale': 4,
     'Siltstone': 3}



The parameters used for the interpolation can be returned using the
function ``get_kriging_parameters``. These are generated automatically
from the orginal data, but can be changed if needed. However, users
should be careful doing so, if they do not fully understand their
significance.

.. code:: ipython3

    gp.get_kriging_parameters(interp_data) # Maybe move this to an extra part?


.. parsed-literal::

    range 0.911605715751648 3464.10171986
    Number of drift equations [0 3]
    Covariance at 0 0.019786307588219643
    orientations nugget effect 0.009999999776482582
    scalar nugget effect 9.999999974752427e-07


At this point, we have all we need to compute our full model via
``compute_model``. By default, this will return two separate solutions
in the form of arrays. The first gives information on the lithological
formations, the second on the fault network in the model. These arrays
consist of two subarrays as entries each:

1. Lithology block model solution:

   -  Entry [0]: This array shows what kind of lithological formation is
      found in each voxel, as indicated by a respective formation
      number.
   -  Entry [1]: Potential field array that represents the orientation
      of lithological units and layers in the block model.

2. Fault network block model solution:

   -  Entry [0]: Array in which all fault-separated areas of the model
      are represented by a distinct number contained in each voxel.
   -  Entry [1}: Potential field array related to the fault network in
      the block model.

Below, we illustrate these different model solutions and how they can be
used.

.. code:: ipython3

    lith_block, fault_block = gp.compute_model(interp_data)

Direct model visualization in GemPy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Model solutions can be easily visualized in 2D sections in GemPy
directly. Let's take a look at our lithology block:

.. code:: ipython3

    %matplotlib inline
    gp.plot_section(geo_data, lith_block[0], cell_number=50,  direction='y', plot_data=False)



.. image:: ch1_files/ch1_30_0.png


With ``cell_number=25`` and remembering that we defined our resolution
to be 50 cells in each direction, we have chosen a section going through
the middle of our block. We have moved 25 cells in ``direction='y'``,
the plot thus depicts a plane parallel to the :math:`x`- and
:math:`y`-axes. Setting ``plot_data=True``, we could plot original data
together with the results. Changing the values for ``cell_number``\ and
``direction``, we can move through our 3D block model and explore it by
looking at different 2D planes.

We can do the same with out lithological scalar-field solution:

.. code:: ipython3

    gp.plot_scalar_field(geo_data, lith_block[1], cell_number=50, N=15, 
                            direction='y', plot_data=False)



.. image:: ch1_files/ch1_32_0.png


Here, ``N`` defines the number of lines plotted to indicate the scalar
field (default is 20).

For this example, is it interesting to look at the scalar field from
direction of the :math:`z`-axis:

.. code:: ipython3

    gp.plot_scalar_field(geo_data, lith_block[1], cell_number=25, N=15, 
                            direction='z', plot_data=False)



.. image:: ch1_files/ch1_34_0.png


This illustrates well the fold-related deformation of the stratigraphy,
as well as the way the layers are influenced by the fault.

The fault network modeling solutions can be visualized in the same way:

.. code:: ipython3

    gp.plot_section(geo_data, fault_block[0], cell_number=25, plot_data=False)



.. image:: ch1_files/ch1_36_0.png


.. code:: ipython3

    gp.plot_scalar_field(geo_data, fault_block[1], cell_number=25, N=20, 
                            direction='y', plot_data=False)



.. image:: ch1_files/ch1_37_0.png


In the end, it is the combination of lithological and fault network
solutions that provides us with a full geological model.

Surfaces can be visualized as 3D triangle complexes in VTK (see function
``plot_surfaces_3D`` below). To create these triangles, we need to
extract respective vertices and simplices from the potential fields of
lithologies and faults. This process is automatized in GemPy with the
function ``get_surfaces``:

.. code:: ipython3

    ver, sim = gp.get_surfaces(interp_data,lith_block[1], fault_block[1], original_scale=True)
    
    # In python is very easy to export data using for example:
    # np.save('ver_fabian', ver)
    # np.save('sim_fabian', sim)

.. code:: ipython3

    gp.plot_surfaces_3D(geo_data, ver, sim, alpha=1)




.. parsed-literal::

    <gempy.visualization.vtkVisualization at 0x7fdf66dde0f0>



The vertices always cut the edges of the voxels. In the next cell we can
see how vertices and voxels relate:

.. code:: ipython3

    # Cropping a cross-section to visualize in 2D #REDO this part?
    bool_b = np.array(ver[1][:,1] > 999)* np.array(ver[1][:,1] < 1001) 
    bool_r = np.array(ver[1][:,1] > 1039)* np.array(ver[1][:,1] < 1041)
    
    # Plotting section
    gp.plot_section(geo_data, lith_block[0], 50, plot_data=True)
    ax = plt.gca()
    
    # Adding grid
    ax.set_xticks(np.linspace(0, 2000, 100, endpoint=False))
    ax.set_yticks(np.linspace(0, 2000, 100, endpoint=False))
    plt.grid()
    
    plt.ylim(1000,1600)
    plt.xlim(500,1100)
    # Plotting vertices
    ax.plot(ver[1][bool_r][:, 0], ver[1][bool_r][:, 2], '.', color='b', alpha=.9)
    ax.get_xaxis().set_ticklabels([])





.. parsed-literal::

    []




.. image:: ch1_files/ch1_42_1.png


Interactive 3D visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using the rescaled interpolation data, we can also run our 3D VTK
visualization in an interactive mode which allows us to alter and update
our model in real time. Similarly to the interactive 3D visualization of
our input data, the changes are permamently saved (in the
``InterpolationInput`` dataframe object). Addtionally, the resulting
changes in the geological models are re-computed in real time.

Important: To get a smooth response, it is important to have the Theano
optimization in ``fast_run``-mode (you can change this in the GemPy file
``theanograf.py``) and to run Theano in the GPU. This can speed up the
modeling time by a factor of 20.

.. code:: ipython3

    ver_s, sim_s = gp.get_surfaces(interp_data,lith_block[1],
                                   fault_block[1],
                                   original_scale=True)

.. code:: ipython3

    gp.plot_surfaces_3D_real_time(interp_data, ver_s, sim_s)
