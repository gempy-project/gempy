import pytest
from gempy.assets import topology as tp
import numpy as np
import gempy as gp
import matplotlib.pyplot as plt


@pytest.fixture(scope='module')
def topology_fabian(one_fault_model_solution):
    """Return a GemPy Vista instance with basic geomodel attached."""
    geo_model = one_fault_model_solution

    edges, centroids = tp.compute_topology(geo_model, voxel_threshold=1)

    return edges, centroids


@pytest.fixture(scope='module')
def topology_jan_unconf(unconformity_model):
    geo_model = unconformity_model
    gp.plot.plot_2d(unconformity_model)
    plt.show()
    edges, centroids = tp.compute_topology(geo_model, voxel_threshold=1)
    return edges, centroids


def assert_centroids(centroids, centroids_test):
    for key in centroids_test.keys():
        for i in range(2):
            assert pytest.approx(centroids[key][i]) == centroids_test[key][i]


@pytest.fixture
def edges_fabian_test():
    return {(1, 2), (4, 10), (2, 7), (6, 7), (5, 10), (4, 9), (4, 5), (2, 8),
            (3, 8), (8, 9), (9, 10), (3, 9), (2, 3), (1, 6), (1, 7), (3, 4),
            (7, 8)}


@pytest.fixture
def centroids_fabian_test():
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


def test_edges_fabian(topology_fabian, edges_fabian_test):
    assert topology_fabian[0] == edges_fabian_test


def test_centroids_fabian(topology_fabian, centroids_fabian_test):
    assert_centroids(topology_fabian[1], centroids_fabian_test)


@pytest.fixture
def edges_jan_unconf_test():
    return {(1, 2), (1, 3), (1, 4), (2, 3), (3, 4)}


@pytest.fixture
def centroids_jan_unconf_test():
    return {1: np.array([24.5, 20.5, 27.5]),
            2: np.array([24.5, 20.5, 21.37869822]),
            3: np.array([24.5, 20.5, 19.12100139]),
            4: np.array([24.5, 20.5, 9.71685136])}


def test_edges_jan_unconf(topology_jan_unconf, edges_jan_unconf_test):
    assert topology_jan_unconf[0] == edges_jan_unconf_test


def test_centroids_jan_unconf(topology_jan_unconf, centroids_jan_unconf_test):
    assert_centroids(topology_jan_unconf[1], centroids_jan_unconf_test)
