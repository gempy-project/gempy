import pytest
import sys, os

from gempy.addons.gempy_to_rexfile import GemPyToRex

sys.path.append("../..")
import gempy
request = pytest.importorskip("requests")
from gempy.addons import gempy_to_rexfile as gtr
from gempy.addons import rex_api
pyqrcode = pytest.importorskip("pyqrcode")
input_path = os.path.dirname(__file__)+'/../input_data'


@pytest.mark.skipif("TRAVIS" in os.environ and os.environ["TRAVIS"] == "true",
                    reason="Skipping this test on Travis CI.")
class TestGemPyToREX:
    @pytest.fixture(scope='module')
    def geo_model(self, model_horizontal_two_layers):
        geo_data = model_horizontal_two_layers

        # Compute model
        sol = gempy.compute_model(geo_data, compute_mesh_options={'rescale': True})

        return geo_data

    def test_write_header(self):
        header_bytes = gtr.write_file_header_block(3, 1)
        if False:
            gtr.write_file(header_bytes, './rexfiles/header_test')

    def test_write_mesh(self, geo_model):
        mesh_header_size = 128
        file_header_size = 86
        ver, tri = geo_model.solutions.vertices[0], geo_model.solutions.edges[0]

        ver_ravel, tri_ravel, n_vtx_coord, n_triangles = gtr.mesh_preprocess(ver, tri)
        data_block_size_no_header = (n_vtx_coord + n_triangles) * 4 + mesh_header_size

        # Write header
        header_bytes = gtr.write_file_header_block(n_data_blocks=1,
                                            size_data_blocks=data_block_size_no_header+gtr.rexDataBlockHeaderSize,
                                            start_data=file_header_size)

        # Write data block
        data_bytes = gtr.write_data_block_header(size_data=data_block_size_no_header,
                                          data_id=1, data_type=3, version_data=1)

        # Write mesh block
        mesh_header_bytes = gtr.write_mesh_header(n_vtx_coord/3, n_triangles/3,
                                                  start_vtx_coord=gtr.mesh_header_size,
                                                  start_nor_coord=gtr.mesh_header_size + n_vtx_coord*4,
                                                  start_tex_coord=gtr.mesh_header_size + n_vtx_coord*4,
                                                  start_vtx_colors=gtr.mesh_header_size + n_vtx_coord*4,
                                                  start_triangles=gtr.mesh_header_size + n_vtx_coord*4,
                                                  name='test_a')

        mesh_block_bytes = gtr.write_mesh_coordinates(ver_ravel, tri_ravel)

        all_bytes = header_bytes + data_bytes + mesh_header_bytes + mesh_block_bytes

        if False:
            gtr.write_file(all_bytes, './rexfiles/one_mesh_test')

    def TEST_rex_cloud_api(self):
        import datetime
        timestamp = datetime.datetime.now()
        project_name = str(timestamp)
        rex_api.RexAPI(project_name)

    def test_geo_model_to_rex(self, geo_model):
        rex_bytes = gtr.geomodel_to_rex(geo_model)
        print(len(rex_bytes['A']))
        assert len(rex_bytes['A']) == 235706

    def test_rex_bytes_to_file(self, geo_model):
        rex_bytes = gtr.geomodel_to_rex(geo_model)
        gtr.write_rex(rex_bytes, path=os.path.dirname(__file__) + '/rexfiles/gtr_test')

    def test_rex_bytes_to_file_except(self):
        model = gempy.create_data(extent=[0,10,0,10,0,10])
        model.set_default_surfaces()
        model._surfaces.df['isActive'] = False
        with pytest.raises(RuntimeError):
            rex_bytes = gtr.geomodel_to_rex(model)

    @pytest.mark.skip(reason="Needs token and secret. (@leguark they are in notion)")
    def test_plot_ar(self, geo_model):
        tag = gempy.plot.plot_ar(geo_model,
                                 api_token='foo', \
                                 project_name='Alesmodel',
                                 secret='bar')
        print(tag.display_tag(reverse=False))