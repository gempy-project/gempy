
Tutorial Chapter 6: Analyzing Topology (WIP)
============================================

.. code:: ipython3

    import sys
    sys.path.append("../../gempy")
    
    import gempy as gp
    
    import numpy as np
    import matplotlib.pyplot as plt
    %matplotlib inline

Creating an example Model
-------------------------

First let's set up a simple example model. For that we initialize the
geo\_data object with the correct model extent and the resolution we
like. Then we load our data points from csv files and set the series and
order the formations (stratigraphic pile).

.. code:: ipython3

    # initialize geo_data object
    geo_data = gp.create_data([0, 3000, 0, 20, 0, 2000], resolution=[100, 3, 67])
    # import data points
    geo_data.import_data_csv("../input_data/ch6_data_interf", 
                             "../input_data/ch6_data_fol"),
    # set series and order formations
    geo_data.set_formation_number()
    gp.set_series(geo_data, {"fault":geo_data.get_formations()[np.where(geo_data.get_formations()=="Fault")[0][0]], 
                             "Rest":np.delete(geo_data.get_formations(), np.where(geo_data.get_formations()=="Fault")[0][0])},
                               order_series = ["fault", "Rest"], verbose=0, order_formations=['Fault','Layer 2', 'Layer 3', 'Layer 4', 'Layer 5'])
    
    geo_data.set_formation_number()
    geo_data.order_table()



.. image:: ch6_files/ch6_3_0.png


And quickly have a look at the data:

.. code:: ipython3

    gp.plot_data(geo_data)
    plt.xlim(0,3000)
    plt.ylim(0,2000);



.. image:: ch6_files/ch6_5_0.png


Then we can compile our interpolator object and compute our model:

.. code:: ipython3

    interp_data = gp.InterpolatorInput(geo_data, u_grade=[0,3])#, verbose=['n_formation'], dtype="float32")
    lith_block, fault_block = gp.compute_model(interp_data)


.. parsed-literal::

    Level of Optimization:  fast_compile
    Device:  cpu
    Precision:  float32


.. code:: ipython3

    gp.plot_section(geo_data, lith_block[0], 0)



.. image:: ch6_files/ch6_8_0.png


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

    G, centroids, labels_unique, lith_to_labels_lot, labels_to_lith_lot = gp.topology_compute(geo_data, lith_block[0], fault_block[0])

After computing the model topology, we can overlay the topology graph
over a model section:

.. code:: ipython3

    gp.plot_section(geo_data, lith_block[0], 0)
    gp.topology_plot(geo_data, G, centroids)



.. image:: ch6_files/ch6_12_0.png


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

    gp.topology_check_adjacency(G, 8, 3)




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

