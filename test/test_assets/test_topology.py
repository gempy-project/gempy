import os
import sys
import pytest
import numpy as np
from test.context import gempy as gp
from gempy.assets.topology import compute_topology
from networkx.classes.coreviews import AdjacencyView

TEST_DIR = os.path.dirname(os.path.abspath(__file__))

@pytest.fixture
def topo_geodata():
    # initialize geo_model object

    sys.path.insert(0, TEST_DIR)
    geo_model = gp.create_model("test_topology")

    gp.init_data(geo_model, [0, 3000, 0, 20, 0, 2000], [30, 3, 30],
                 path_i=TEST_DIR+"/input_data/ch6_data_interf.csv",
                 path_o=TEST_DIR+"/input_data/ch6_data_fol.csv")

    gp.map_series_to_surfaces(geo_model,
                                {"fault": "Fault",
                                 "Rest": ('Layer 2', 'Layer 3', 'Layer 4', 'Layer 5')}
                                )
    geo_model.set_is_fault(["fault"])
    geo_model.solutions.lith_block = np.load(TEST_DIR+"/input_data/topology_lith_block.npy")
    geo_model.solutions.fault_blocks = np.load(TEST_DIR+"/input_data/topology_fault_blocks.npy")
    return geo_model


@pytest.fixture
def topo_compute(topo_geodata):
    return compute_topology(topo_geodata)


def test_topo_centroids(topo_compute):
    centroids = topo_compute[1]
    centroids_test = {'2_6': (7.980988593155893, 1.0, 6.612167300380228),
                     '2_5': (7.469387755102041, 1.0, 15.73469387755102),
                     '2_4': (6.533333333333333, 1.0, 19.244444444444444),
                     '2_3': (5.9743589743589745, 1.0, 22.564102564102566),
                     '2_2': (5.634615384615385, 1.0, 26.634615384615383),
                     '1_2': (20.934065934065934, 1.0, 22.186813186813186),
                     '1_3': (21.659574468085108, 1.0, 12.76595744680851),
                     '1_4': (22.466666666666665, 1.0, 9.333333333333334),
                     '1_5': (23.157894736842106, 1.0, 5.973684210526316),
                     '1_6': (23.285714285714285, 1.0, 2.142857142857143)}
    assert centroids == centroids_test, "Topology centroids mismatch."


def test_topo_Gadj(topo_compute):
    Gadj = topo_compute[0].adj
    Gadj_test = AdjacencyView({'2_6': {'2_5': {'edge_type': 'stratigraphic'}, '1_4': {'edge_type': 'fault'}, '1_5': {'edge_type': 'fault'}, '1_6': {'edge_type': 'fault'}}, '2_5': {'2_6': {'edge_type': 'stratigraphic'}, '2_4': {'edge_type': 'stratigraphic'}, '1_3': {'edge_type': 'fault'}, '1_4': {'edge_type': 'fault'}}, '2_4': {'2_5': {'edge_type': 'stratigraphic'}, '2_3': {'edge_type': 'stratigraphic'}, '1_3': {'edge_type': 'fault'}, '1_2': {'edge_type': 'fault'}}, '2_3': {'2_4': {'edge_type': 'stratigraphic'}, '2_2': {'edge_type': 'stratigraphic'}, '1_2': {'edge_type': 'fault'}}, '2_2': {'2_3': {'edge_type': 'stratigraphic'}, '1_2': {'edge_type': 'fault'}}, '1_2': {'2_4': {'edge_type': 'fault'}, '2_3': {'edge_type': 'fault'}, '2_2': {'edge_type': 'fault'}, '1_3': {'edge_type': 'stratigraphic'}}, '1_3': {'2_5': {'edge_type': 'fault'}, '2_4': {'edge_type': 'fault'}, '1_2': {'edge_type': 'stratigraphic'}, '1_4': {'edge_type': 'stratigraphic'}}, '1_4': {'2_6': {'edge_type': 'fault'}, '2_5': {'edge_type': 'fault'}, '1_3': {'edge_type': 'stratigraphic'}, '1_5': {'edge_type': 'stratigraphic'}}, '1_5': {'2_6': {'edge_type': 'fault'}, '1_4': {'edge_type': 'stratigraphic'}, '1_6': {'edge_type': 'stratigraphic'}}, '1_6': {'2_6': {'edge_type': 'fault'}, '1_5': {'edge_type': 'stratigraphic'}}})

    assert Gadj == Gadj_test, "Mismatch in G.adj from topology analysis. Could be (a) general topology misclassification; or (b) wrong edge_type classification."