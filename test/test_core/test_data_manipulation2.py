# These two lines are necessary only if GemPy is not installed
import sys, os
sys.path.append("../..")

# Importing GemPy
import gempy as gp


# Importing auxiliary libraries
import numpy as np
import pandas as pn
import matplotlib.pyplot as plt
import pytest

mm = gp.DataMutation()
mm.add_surfaces(['surface1', 'foo1', 'foo2', 'foo3'])

def test_add_surface_points_raise_non_surface():
    with pytest.raises(ValueError):
        mm.add_surface_points(400, 300, -500, 'surface5')


def test_add_surface():
    mm.add_surfaces(['a_foo1', 'a_foo2'])
    with pytest.raises(ValueError, match=r'.* not include old categories.*'):
        mm.add_surfaces('a_foo1')

    mm.add_surfaces('a_foo3')


def test_delete_surface():
    mm.delete_surfaces('foo1')
    mm.delete_surfaces(['surface1', 'foo2'])


def test_rename_surface():
    mm = gp.DataMutation()
    mm.add_surfaces(['surface1', 'foo1', 'foo2', 'foo3'])
    mm.rename_surfaces({'foo1': 'changed'})
    assert mm.surfaces.df.loc[1, 'surface'] == 'changed'


def test_modify_order_surfaces():
    mm = gp.DataMutation()
    mm.add_surfaces(['surface1', 'foo1', 'foo2', 'foo3'])
    mm.modify_order_surfaces(3, 2)
    assert mm.surfaces.df.iloc[2, 0] == 'foo2'


def test_add_surface_values():
    pass


def test_delete_surface_values():
    pass


def test_modify_surface_values():
    mm = gp.DataMutation()
    mm.add_surfaces(['surface1', 'foo1', 'foo2', 'foo3'])
    mm.add_surface_points(400, 300, -500, 'foo2')
    print(mm.surface_points)
    mm.modify_surface_points(0, Y=800)
    print(mm.surface_points)


def test_set_surface_values():
    pass


def test_add_surface_points():
    mm = gp.DataMutation()
    mm.add_surfaces(['surface1', 'foo1', 'foo2', 'foo3'])
    mm.add_surface_points(400, 300, -500, 'foo2')


def test_add_default_orientation():
    mm = gp.DataMutation()
    mm.set_default_surfaces()
    mm.set_default_orientation()


def test_set_is_fault():
    mm = gp.DataMutation()
    mm.add_series(['foo1', 'foo2', 'foo3'])
    mm.set_is_fault(['foo2'])
    mm.set_is_fault(['foo2'], toggle=True)
