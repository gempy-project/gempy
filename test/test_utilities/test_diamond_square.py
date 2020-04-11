from gempy.utils.diamond_square import square
import pytest # to add fixtures
import numpy as np # as another testing environment



@pytest.fixture(scope="module")
def data():
    return [1,2,3,4]



def test_square_nocrash():
    square([1])


def test_square(data):
    # without "fixtures"
    # vals = [1,2,3,4]
    # with "fixtures"
    vals = data

    assert square(vals) == [1,4,9,16]
