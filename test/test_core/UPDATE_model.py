import gempy.gempy_api as gp
import pytest


class TestModel:

    @pytest.fixture(scope='class')
    def test_create_model(self):
        model = gp.Model()
        print(model)
        return model

