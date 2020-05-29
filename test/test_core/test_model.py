import gempy.core.model
import pytest  # to add fixtures and to test error raises
import numpy as np  # as another testing environment


def test_model():
    """Simply check if class can be instantiated"""
    gempy.core.model.Model()


def test_default_name():
    """Test default name of model"""
    geomodel = gempy.core.model.Model()
    # TODO: test for metdata?


def test_default_crs():
    """Default crs should be set to None"""
    geomodel = gempy.core.model.Model()
    assert geomodel.meta.crs is None


def test_crs_property():
    """Set coordinate reference system using crs code"""
    geomodel = gempy.core.model.Model()
    geomodel.meta.crs = '4326'
    assert geomodel.meta.crs == '4326'
