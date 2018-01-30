import os
import sys
import pytest
import numpy as np
from .context import gempy as gp

TEST_DIR = os.path.dirname(os.path.abspath(__file__))

@pytest.fixture
def topo_geodata():
    # initialize geo_data object

    sys.path.insert(0, TEST_DIR)

    geo_data = gp.create_data([0, 3000, 0, 20, 0, 2000], resolution=[3, 3, 3])
    geo_data.import_data_csv(path_i=TEST_DIR+"/input_data/ch6_data_interf.csv",
                             path_o=TEST_DIR+"/input_data/ch6_data_fol.csv")

    gp.set_series(geo_data, {"fault": geo_data.get_formations()[np.where(geo_data.get_formations() == "Fault")[0][0]],
                             "Rest": np.delete(geo_data.get_formations(),
                                               np.where(geo_data.get_formations() == "Fault")[0][0])},
                  order_series=["fault", "Rest"], verbose=0,
                  order_formations=['Fault', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5'])

    return geo_data

@pytest.fixture
def topo_lb_fb():
    lb = np.array([[0., 0., 3., 0.,
                    0., 3., 0., 0.,
                    3., 0., 4., 2.,
                    0., 4., 2., 0.,
                    4., 2., 5., 2.,
                    2., 5., 2., 2.,
                    5., 2., 2.],
                   [899.75524902, 899.79650879, 900.00585938, 899.75524902,
                    899.79644775, 900.00585938, 899.75524902, 899.79650879,
                    900.00585938, 899.74884033, 899.91253662, 900.14904785,
                    899.74884033, 899.91253662, 900.14904785, 899.74884033,
                    899.91253662, 900.14904785, 899.81628418, 900.05279541,
                    900.25317383, 899.81628418, 900.05279541, 900.25317383,
                    899.81628418, 900.05279541, 900.25317383]], dtype="float32")

    fb = np.array([[0., 0., 0., 0.,
                    0., 0., 0., 0.,
                    0., 0., 1., 1.,
                    0., 1., 1., 0.,
                    1., 1., 1., 1.,
                    1., 1., 1., 1.,
                    1., 1., 1.],
                   [949.75537109, 949.78796387, 949.89208984, 949.75537109,
                    949.78796387, 949.89208984, 949.75537109, 949.78796387,
                    949.89208984, 949.89471436, 950.00244141, 950.10699463,
                    949.89471436, 950.00244141, 950.10699463, 949.89471436,
                    950.00244141, 950.10699463, 950.11114502, 950.21447754,
                    950.24395752, 950.11114502, 950.21447754, 950.24395752,
                    950.11114502, 950.21447754, 950.24395752]], dtype="float32")

    return lb, fb


@pytest.fixture
def topo_compute(topo_geodata, topo_lb_fb):
    return gp.topology_compute(topo_geodata, topo_lb_fb[0][0], topo_lb_fb[1])


def test_topo_centroids(topo_compute):
    centroids = topo_compute[1]
    centroids_test = {1: (0.33333333333333331, 1.0, 0.33333333333333331),
                      2: (0.0, 1.0, 2.0),
                      3: (1.0, 1.0, 1.0),
                      4: (1.6666666666666667, 1.0, 1.6666666666666667),
                      5: (2.0, 1.0, 0.0)}
    assert centroids == centroids_test, "Topology centroids mismatch."


def test_topo_labels_unique(topo_compute):
    labels_unique_test = np.array([1, 2, 3, 4, 5], dtype="int64")
    assert (topo_compute[2] == labels_unique_test).all(), "Mismatch in node labels from topology calculation."


def test_topo_lot1(topo_compute):
    lot1_test = {'0': {'1': {}},
                 '2': {'4': {}},
                 '3': {'2': {}},
                 '4': {'3': {}},
                 '5': {'5': {}}}

    assert topo_compute[3] == lot1_test


def test_topo_lot2(topo_compute):
    lot2_test = {1: '0', 2: '3', 3: '4', 4: '2', 5: '5'}
    assert topo_compute[4] == lot2_test


def test_topo_Gadj(topo_compute):
    Gadj = topo_compute[0].adj
    Gadj_test = {1: {2: {'edge_type': 'stratigraphic'}, 3: {'edge_type': 'fault'}, 5: {'edge_type': 'fault'}},
                 2: {1: {'edge_type': 'stratigraphic'}, 4: {'edge_type': 'fault'}}, 3: {1: {'edge_type': 'fault'},
                 4: {'edge_type': 'stratigraphic'}}, 4: {2: {'edge_type': 'fault'}, 3: {'edge_type': 'stratigraphic'},
                 5: {'edge_type': 'stratigraphic'}}, 5: {1: {'edge_type': 'fault'}, 4: {'edge_type': 'stratigraphic'}}}

    assert Gadj == Gadj_test, "Mismatch in G.adj from topology analysis. Could be (a) general topology misclassification; or (b) wrong edge_type classification."