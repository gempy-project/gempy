
# coding: utf-8

# # Chapter 6: Analyzing Topology (WIP)

# In[1]:


import sys, os
sys.path.append("../../")

import gempy as gp

import numpy as np
import matplotlib.pyplot as plt

#from ..conftest import theano_f_1f
input_path = os.path.dirname(__file__)+'/../../notebooks'

def test_ch6(theano_f_1f):


    # initialize geo_data object
    geo_data = gp.create_data([0, 3000, 0, 20, 0, 2000], resolution=[50, 3, 67])
    # import data points
    geo_data.import_data_csv(input_path+"/input_data/tut_chapter6/ch6_data_interf.csv",
                             input_path+"/input_data/tut_chapter6/ch6_data_fol.csv")


    gp.set_series(geo_data, {"fault":geo_data.get_formations()[np.where(geo_data.get_formations()=="Fault")[0][0]],
                             "Rest":np.delete(geo_data.get_formations(), np.where(geo_data.get_formations()=="Fault")[0][0])},
                               order_series = ["fault", "Rest"], verbose=0, order_formations=['Fault','Layer 2', 'Layer 3', 'Layer 4', 'Layer 5'])


    gp.plot_data(geo_data)
    plt.xlim(0,3000)
    plt.ylim(0,2000);

    interp_data = gp.InterpolatorData(geo_data, u_grade=[0,1])
    lith_block, fault_block = gp.compute_model(interp_data)

    gp.plot_section(geo_data, lith_block[0], 0)

    G, centroids, labels_unique, lith_to_labels_lot, labels_to_lith_lot = gp.topology_compute(
        geo_data, lith_block[0], fault_block)

    gp.plot_section(geo_data, lith_block[0], 0, direction='y')
    gp.plot_topology(geo_data, G, centroids)

    lith_to_labels_lot["4"].keys()

    gp.topology.check_adjacency(G, 8, 3)

    G.adj[8]

    G.adj[8][2]["edge_type"]
