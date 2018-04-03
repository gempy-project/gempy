
Chapter 6: Analyzing Topology (WIP)
===================================

.. code:: ipython3

    import sys
    sys.path.append("../../")
    
    import gempy as gp
    
    import numpy as np
    import matplotlib.pyplot as plt
    %matplotlib inline


.. parsed-literal::

    WARNING (theano.tensor.blas): Using NumPy C-API based implementation for BLAS functions.


Creating an example Model
-------------------------

First let's set up a simple example model. For that we initialize the
geo\_data object with the correct model extent and the resolution we
like. Then we load our data points from csv files and set the series and
order the formations (stratigraphic pile).

.. code:: ipython3

    # initialize geo_data object
    geo_data = gp.create_data([0, 3000, 0, 20, 0, 2000], resolution=[50, 3, 67])
    # import data points
    geo_data.import_data_csv("../input_data/ch6_data_interf", 
                             "../input_data/ch6_data_fol")
    
    geo_data.calculate_gradient()


.. code:: ipython3

    geo_data.orientations




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
          <th>...</th>
          <th>Y_std</th>
          <th>Z_std</th>
          <th>dip_std</th>
          <th>azimuth_std</th>
          <th>order_series</th>
          <th>isFault</th>
          <th>formation_number</th>
          <th>annotations</th>
          <th>group_id</th>
          <th>index</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1500.000000</td>
          <td>6.666667</td>
          <td>990.000000</td>
          <td>0.868243</td>
          <td>1.000000e-07</td>
          <td>0.496139</td>
          <td>60.255119</td>
          <td>90.0</td>
          <td>1</td>
          <td>Fault</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>1</td>
          <td>True</td>
          <td>1</td>
          <td>${\bf{x}}_{\beta \,{\bf{1}},0}$</td>
          <td>fault</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1</th>
          <td>506.333333</td>
          <td>9.666667</td>
          <td>1679.333333</td>
          <td>0.258819</td>
          <td>1.000000e-07</td>
          <td>0.965926</td>
          <td>15.000000</td>
          <td>90.0</td>
          <td>1</td>
          <td>Layer 2</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>2</td>
          <td>False</td>
          <td>2</td>
          <td>${\bf{x}}_{\beta \,{\bf{2}},0}$</td>
          <td>l2_a</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2</th>
          <td>2500.000000</td>
          <td>9.666667</td>
          <td>911.000000</td>
          <td>0.258819</td>
          <td>1.000000e-07</td>
          <td>0.965926</td>
          <td>15.000000</td>
          <td>90.0</td>
          <td>1</td>
          <td>Layer 2</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>2</td>
          <td>False</td>
          <td>2</td>
          <td>${\bf{x}}_{\beta \,{\bf{2}},1}$</td>
          <td>l2_a</td>
          <td>1.0</td>
        </tr>
      </tbody>
    </table>
    <p>3 rows × 22 columns</p>
    </div>



.. code:: ipython3

    gp.set_series(geo_data, {"fault":geo_data.get_formations()[np.where(geo_data.get_formations()=="Fault")[0][0]], 
                             "Rest":np.delete(geo_data.get_formations(), np.where(geo_data.get_formations()=="Fault")[0][0])},
                               order_series = ["fault", "Rest"], verbose=0, order_formations=['Fault','Layer 2', 'Layer 3', 'Layer 4', 'Layer 5'])
    


And quickly have a look at the data:

.. code:: ipython3

    gp.plot_data(geo_data)
    plt.xlim(0,3000)
    plt.ylim(0,2000);



.. image:: ch6_files/ch6_7_0.png


Then we can compile our interpolator object and compute our model:

.. code:: ipython3

    interp_data = gp.InterpolatorData(geo_data, u_grade=[0,1])
    lith_block, fault_block = gp.compute_model(interp_data)


.. parsed-literal::

    Compiling theano function...
    Compilation Done!
    Level of Optimization:  fast_compile
    Device:  cpu
    Precision:  float32
    Number of faults:  1


.. code:: ipython3

    gp.plot_section(geo_data, lith_block[0], 0)



.. image:: ch6_files/ch6_10_0.png


Analyzing Topology
------------------

GemPy sports in-built functionality to analyze the topology of its
models. All we need for this is our geo\_data object, lithology block
and the fault block. We input those into *gp.topology\_compute* and get
several useful outputs:

-  an adjacency graph **G**, representing the topological relationships
   of the model
-  the **centroids** of the all the unique topological regions in the
   model (x,y,z coordinates of their center)
-  a list of all the unique labels (labels\_unique)
-  two look-up-tables from the lithology id's to the node labels, and
   vice versa

.. code:: ipython3

    G, centroids, labels_unique, lith_to_labels_lot, labels_to_lith_lot = gp.topology_compute(geo_data, lith_block[0], fault_block)

After computing the model topology, we can overlay the topology graph
over a model section:

.. code:: ipython3

    gp.plot_section(geo_data, lith_block[0], 0)
    gp.plot_topology(geo_data, G, centroids)



.. image:: ch6_files/ch6_14_0.png


So let's say we want to check if the green layer (layer 4) is connected
across the fault. For that we first need to look up which nodes belong
to the layer. In this simple model we could easily do that by looking at
the plot above, but we can also use the look-up-tables provided by the
topology function:

.. code:: ipython3

    lith_to_labels_lot["4"].keys()




.. parsed-literal::

    dict_keys(['3', '8'])



Okay, layer 4 is represented by nodes 3 and 8. We can now put these into
*topology\_check\_adjacency* function, which puts out *True* if the two
nodes share a connection (are adjacent) and *False* if not:

.. code:: ipython3

    gp.topology.check_adjacency(G, 8, 3)




.. parsed-literal::

    False



We can also easily look up to which other nodes a node is adjacent:

.. code:: ipython3

    G.adj[8]




.. parsed-literal::

    {1: {'edge_type': 'fault'},
     2: {'edge_type': 'fault'},
     7: {'edge_type': 'stratigraphic'},
     9: {'edge_type': 'stratigraphic'}}



The adjacency dictionary of the graph shows that node 8 is connected to
nodes 1, 2, 7 and 9. If we go one level deeper in the dictionary, we can
access the type of connection (edge):

.. code:: ipython3

    G.adj[8][2]["edge_type"]




.. parsed-literal::

    'fault'



This way we can directly check if node 8 and 2 (or any other pair of
nodes that share a connection) are connected across a fault, or just
stratigraphically.
