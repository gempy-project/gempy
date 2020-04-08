import os
import pytest
import gempy as gp

input_path = os.path.dirname(__file__) + '/../../notebooks/data'


@pytest.mark.skipif("TRAVIS" in os.environ and os.environ["TRAVIS"] == "true",
                    reason="Skipping this test on Travis CI.")

from gempy.plot import vista as vs
def test_set_bounds():

