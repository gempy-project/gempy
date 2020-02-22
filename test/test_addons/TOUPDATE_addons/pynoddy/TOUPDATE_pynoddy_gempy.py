import pytest
import numpy as np
import gempy as gp
import sys
import os
input_path = os.path.dirname(__file__)
import gempy.utils.input_manipulation as im


def test_find_interfaces():
    block = np.load(input_path+'/noddy_block.npy')
    bool_block = im.find_interfaces_from_block(block, 1)

    geo_data = gp.create_data([0, 6000,
                               0, 6000,
                               0, 500], resolution=[60, 60 ,6])

    p_df = im.interfaces_from_interfaces_block(bool_block, geo_data.grid.values)

    im.set_interfaces_from_block(geo_data, block)
