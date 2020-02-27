from gempy.assets.geographic import GeographicPoint, broken_function
import pytest  # to add fixtures
import numpy as np  # as another testing environment


def test_geographic_point():
    """Test if call with two points works"""
    x = 2.
    y = 3.
    g = GeographicPoint(x, y)
    assert g.x == 2.
    assert g.y == 3.

def test_geographic_point_3D():
    """Test if call with three points works"""
    x = 2.
    y = 3.
    z = 4.
    g = GeographicPoint(x, y, z)
    assert g.x == 2.
    assert g.y == 3.
    assert g.z == 4.


def test_goegraphic_point_latlong_utm():
    """Test geographic point coordination lat/long"""
    x = 2.
    y = 3.
    z = 4.
    g = GeographicPoint(x, y, z)
    assert g.x == 2.
    assert g.y == 3.
    assert g.z == 4.
    assert g.type == 'nongeo'
    g.utm_to_latlong()
    print(g.x, g.y, g.z)

