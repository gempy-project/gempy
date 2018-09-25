import gempy.core.data as gd
import gempy.core.gempy_front as gp
import pandas as pn
import numpy as np
import os
import pytest


class TestModel:
    @pytest.fixture(scope='class')
    def test_create_model(self):
        model = gp.Model()
        print(model)
        return model

    def test_set_grid(self):
        model = self.test_create_model()
        grid = gp.create_grid(grid_type='regular_grid', extent=[0,2000,0,2000,0,2000], resolution=[50,50,50])
        model.set_grid(grid)

