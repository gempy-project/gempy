# import pytest
# import gempy.gempy_api as gempy
# import gempy as gp
import enum
import os

input_path = os.path.dirname(__file__) + '/input_data'
input_path2 = os.path.dirname(__file__) + '/../examples/data/input_data/'
import numpy as np

np.random.seed(1234)


class TestSpeed(enum.Enum):
    MILLISECONDS = 0
    SECONDS = 1
    MINUTES = 2
    HOURS = 3


TEST_SPEED = TestSpeed.MINUTES  # * Use seconds for compile errors, minutes before pushing and hours before release
