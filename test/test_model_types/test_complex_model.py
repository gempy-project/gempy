import gempy as gp
import pandas as pn
import numpy as np
import os
import pytest
import matplotlib.pyplot as plt
input_path = os.path.dirname(__file__)+'/../../notebooks/data'


class TestComplexModel:

    def test_init_model(self, model_complex):
        print(model_complex)

    def test_get_data(self, model_complex):
        print(gp.get_data(model_complex))
        print(gp.get_data(model_complex, itype='additional_data'))
