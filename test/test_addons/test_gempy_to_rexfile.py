import pooch
import pytest
import sys, os

from gempy.addons.gempy_to_rexfile import GemPyToRex, geomodel_to_rex

import gempy

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

        gempy_to_rex.gempy_meshes_to_rex(surfaces)

    def test_gempy_to_rex(self, unconformity_model_topo):
        gempy_to_rex = GemPyToRex()
        bytes = gempy_to_rex(unconformity_model_topo, app='RexView')

        print(bytes)

    def test_gempy_to_rex_old(self, unconformity_model_topo):
        bytes2 = geomodel_to_rex(unconformity_model_topo, False)

    def test_gempy_to_rex_with_topo(self):
        model_file = pooch.retrieve(url="https://github.com/cgre-aachen/gempy_data/raw/master/data/gempy_models/combination.zip",
                                    known_hash=None)

        geo_model = gempy.load_model(name='combination', path=model_file)
        gempy_to_rex = GemPyToRex()
        bytes = gempy_to_rex(geo_model)
        print(bytes)