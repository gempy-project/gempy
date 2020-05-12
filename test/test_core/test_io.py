import os
import gempy as gp


def test_load_model():
    cwd = os.path.dirname(__file__)
    data_path = cwd + '/../../examples/'
    geo_model = gp.load_model(r'Tutorial_ch1-8_Onlap_relations',
                              path=data_path + 'data/gempy_models', recompile=False)

