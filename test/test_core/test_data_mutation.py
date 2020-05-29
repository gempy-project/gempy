# These two lines are necessary only if GemPy is not installed
# Importing GemPy
import gempy as gp

# Importing auxiliary libraries
import numpy as np
import pytest
import os

mm = gp.ImplicitCoKriging()
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
    mm = gp.ImplicitCoKriging()
    mm.add_surfaces(['surface1', 'foo1', 'foo2', 'foo3'])
    mm.rename_surfaces({'foo1': 'changed'})
    assert mm._surfaces.df.loc[1, 'surface'] == 'changed'


def test_modify_order_surfaces():
    mm = gp.ImplicitCoKriging()
    mm.add_surfaces(['surface1', 'foo1', 'foo2', 'foo3'])
    mm.modify_order_surfaces(3, 2)
    assert mm._surfaces.df.iloc[2, 0] == 'foo2'


def test_add_surface_values():
    pass


def test_delete_surface_values():
    pass


def test_modify_surface_values():
    mm = gp.ImplicitCoKriging()
    mm.add_surfaces(['surface1', 'foo1', 'foo2', 'foo3'])
    mm.add_surface_points(400, 300, -500, 'foo2')
    print(mm._surface_points)
    mm.modify_surface_points(0, Y=800)
    print(mm._surface_points)


def test_set_surface_values():
    pass


def test_add_surface_points():
    mm = gp.ImplicitCoKriging()
    mm.add_surfaces(['surface1', 'foo1', 'foo2', 'foo3'])
    mm.add_surface_points(400, 300, -500, 'foo2')


def test_add_default_orientation():
    mm = gp.ImplicitCoKriging()
    mm.set_default_surfaces()
    mm.set_default_orientation()


def test_set_is_fault():
    mm = gp.ImplicitCoKriging()
    mm.add_features(['foo1', 'foo2', 'foo3'])
    assert (mm._faults.df.index == np.array(['Default series', 'foo1', 'foo2', 'foo3'])).all()
    assert (mm._faults.faults_relations_df.index == ['Default series', 'foo1', 'foo2', 'foo3']).all()
    mm.set_is_fault(['foo2'])
    assert mm._faults.faults_relations_df.loc['foo2', 'foo3'] == True
    assert mm._faults.faults_relations_df.iloc[2, 3] == True
    mm.set_is_fault(['foo2'], toggle=True)


def test_read_data():
    data_path = os.path.dirname(__file__)+'/../../examples/'
    model = gp.Model()
    model.read_data(path_i=data_path + "/data/input_data/tut_chapter1/simple_fault_model_points.csv",
                    path_o=data_path + "/data/input_data/tut_chapter1/simple_fault_model_orientations.csv")

    assert model._surface_points.df.shape[0] == 57