import pytest
import sys, os
import numpy as np

from gempy.addons.gempy_to_rexfile import GemPyToRex, geomodel_to_rex

# sys.path.append("../..")
import gempy
from gempy.addons import gempy_to_rexfile as gtr
from gempy.addons import rex_api
input_path = os.path.dirname(__file__)+'/../input_data'


@pytest.mark.skipif("TRAVIS" in os.environ and os.environ["TRAVIS"] == "true",
                    reason="Skipping this test on Travis CI.")
class TestGemPyToRexClass:
    """Test the class that control the rexfile encoding"""

    def test_grab_mesh(self, unconformity_model_topo):
        gempy_to_rex = GemPyToRex()
        surfaces = gempy_to_rex.grab_meshes(unconformity_model_topo)
        print(surfaces)

    def test_gempymesh_to_rex(self, unconformity_model_topo):
        gempy_to_rex = GemPyToRex()
        surfaces = gempy_to_rex.grab_meshes(unconformity_model_topo)

        gempy_to_rex.gempy_mesh_to_rex(surfaces)

    def test_gempy_to_rex(self, unconformity_model_topo):
        gempy_to_rex = GemPyToRex()
        bytes = gempy_to_rex(unconformity_model_topo, app='RexView')

        print(bytes)

    def test_gempy_to_rex_old(self, unconformity_model_topo):
        bytes2 = geomodel_to_rex(unconformity_model_topo, False)

    def test_gempy_to_rex_comp(self, unconformity_model_topo):
        gempy_to_rex = GemPyToRex()
        bytes = gempy_to_rex(unconformity_model_topo, app='RexView')
        bytes2 = geomodel_to_rex(unconformity_model_topo)
        np.testing.assert_array_almost_equal(bytes, bytes2)