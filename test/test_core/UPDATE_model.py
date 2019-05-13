import gempy.core.data as gd
import gempy.core.gempy_api as gp
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

