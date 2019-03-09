import os
import sys
import pytest
import numpy as np
from test.context import gempy as gp
from gempy.assets.topology import topology_compute
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

    gp.map_series_to_formations(geo_model,
                                {"fault": "Fault",
                                 "Rest": ('Layer 2', 'Layer 3', 'Layer 4', 'Layer 5')}
                                )
    geo_model.set_is_fault(["fault"])
    geo_model.solutions.lith_block = np.load(TEST_DIR+"/input_data/topology_lith_block.npy")
    geo_model.solutions.fault_blocks = np.load(TEST_DIR+"/input_data/topology_fault_blocks.npy")
    return geo_model


@pytest.fixture
def topo_compute(topo_geodata):
    return topology_compute(topo_geodata)


def test_topo_centroids(topo_compute):
    centroids = topo_compute[1]
    centroids_test = {1: (7.980988593155893, 1.0, 6.612167300380228),
                      2: (7.469387755102041, 1.0, 15.73469387755102),
                      3: (6.533333333333333, 1.0, 19.244444444444444),
                      4: (5.9743589743589745, 1.0, 22.564102564102566),
                      5: (5.634615384615385, 1.0, 26.634615384615383),
                      6: (20.934065934065934, 1.0, 22.186813186813186),
                      7: (21.659574468085108, 1.0, 12.76595744680851),
                      8: (22.466666666666665, 1.0, 9.333333333333334),
                      9: (23.157894736842106, 1.0, 5.973684210526316),
                      10: (23.285714285714285, 1.0, 2.142857142857143)}
    assert centroids == centroids_test, "Topology centroids mismatch."


def test_topo_labels_unique(topo_compute):
    labels_unique_test = np.array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10], dtype="int64")
    assert (topo_compute[2] == labels_unique_test).all(), "Mismatch in node labels from topology calculation."


def test_topo_lot1(topo_compute):
    lot1_test = {'2': {'5': {}, '6': {}},
                  '3': {'4': {}, '7': {}},
                  '4': {'3': {}, '8': {}},
                  '5': {'2': {}, '9': {}},
                  '6': {'1': {}, '10': {}}}

    assert topo_compute[3] == lot1_test


def test_topo_lot2(topo_compute):
    lot2_test = {1: '6',
                  2: '5',
                  3: '4',
                  4: '3',
                  5: '2',
                  6: '2',
                  7: '3',
                  8: '4',
                  9: '5',
                  10: '6'}
    assert topo_compute[4] == lot2_test


def test_topo_Gadj(topo_compute):
    Gadj = topo_compute[0].adj
    Gadj_test = AdjacencyView({1: {2: {'edge_type': 'fault'}, 8: {'edge_type': 'fault'}, 9: {'edge_type': 'fault'}, 10: {'edge_type': 'stratigraphic'}},
                               2: {1: {'edge_type': 'fault'}, 3: {'edge_type': 'fault'}, 7: {'edge_type': 'fault'}, 8: {'edge_type': 'fault'}},
                               3: {2: {'edge_type': 'fault'}, 4: {'edge_type': 'fault'}, 7: {'edge_type': 'fault'}, 6: {'edge_type': 'fault'}},
                               4: {3: {'edge_type': 'fault'}, 5: {'edge_type': 'fault'}, 6: {'edge_type': 'fault'}},
                               5: {4: {'edge_type': 'fault'}, 6: {'edge_type': 'stratigraphic'}},
                               6: {5: {'edge_type': 'stratigraphic'}, 4: {'edge_type': 'fault'}, 3: {'edge_type': 'fault'}, 7: {'edge_type': 'fault'}},
                               7: {3: {'edge_type': 'fault'}, 2: {'edge_type': 'fault'}, 6: {'edge_type': 'fault'}, 8: {'edge_type': 'fault'}},
                               8: {1: {'edge_type': 'fault'}, 2: {'edge_type': 'fault'}, 7: {'edge_type': 'fault'}, 9: {'edge_type': 'fault'}},
                               9: {1: {'edge_type': 'fault'}, 8: {'edge_type': 'fault'}, 10: {'edge_type': 'fault'}},
                               10: {1: {'edge_type': 'stratigraphic'}, 9: {'edge_type': 'fault'}}})

    assert Gadj == Gadj_test, "Mismatch in G.adj from topology analysis. Could be (a) general topology misclassification; or (b) wrong edge_type classification."