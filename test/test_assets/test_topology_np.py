import pytest
import gempy as gp
from gempy.assets import topology as tp
from pickle import load
import os

input_path = os.path.dirname(__file__) + '/../../notebooks/data'
import numpy as np


@pytest.fixture
def geo_model():
    """Return a GemPy Vista instance with basic geomodel attached."""
    geo_model = gp.create_data(
        [0, 2000, 0, 2000, 0, 2000], [50, 50, 50],
        path_o=input_path + '/input_data/tut_chapter1'
                            '/simple_fault_model_orientations.csv',
        path_i=input_path + '/input_data/tut_chapter1'
                            '/simple_fault_model_points.csv'
    )

    gp.set_series(
        geo_model,
        {"Fault_Series": 'Main_Fault',
         "Strat_Series": ('Sandstone_2', 'Siltstone', 'Shale', 'Sandstone_1')}
    )
    geo_model.set_is_fault(['Fault_Series'])

    with open("input_data/geomodel_sol.p", "rb") as f:
        geo_model.solutions = load(f)
    return geo_model


@pytest.fixture
def edges_test():
    return {(1, 2), (4, 10), (2, 7), (6, 7), (5, 10), (4, 9), (4, 5), (2, 8),
            (3, 8), (8, 9), (9, 10), (3, 9), (2, 3), (1, 6), (1, 7), (3, 4),
            (7, 8)}


@pytest.fixture
def centroids_test():
    return {1: np.array([12.16046912, 24.8310141, 40.65527994]),
            2: np.array([8.93961983, 24.92955647, 28.73667536]),
            3: np.array([7.95101873, 24.71784318, 23.43445153]),
            4: np.array([6.85111073, 24.53773105, 17.19230117]),
            5: np.array([4.92739115, 23.20962489, 7.25340731]),
            6: np.array([35.9145365, 25.00458115, 39.16981059]),
            7: np.array([33.7571116, 24.83114515, 26.98018478]),
            8: np.array([32.89434051, 24.74326391, 21.65431227]),
            9: np.array([32.08333333, 24.65390105, 15.5167301]),
            10: np.array([31.04821962, 23.66595199, 5.89393366])}


@pytest.fixture
def topology(geo_model):
    edges, centroids = tp.compute_topology(geo_model, voxel_threshold=1)
    return edges, centroids


def test_edges_topology(topology, edges_test):
    assert topology[0] == edges_test


def test_centroids_topology(topology, centroids_test):
    centroids = topology[1]
    for key in centroids_test.keys():
        for i in range(2):
            assert pytest.approx(centroids[key][i]) == centroids_test[key][i]
