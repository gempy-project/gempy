import os
import pytest
import gempy as gp

input_path = os.path.dirname(__file__) + '/../../notebooks/data'


@pytest.mark.skipif("TRAVIS" in os.environ and os.environ["TRAVIS"] == "true",
                    reason="Skipping this test on Travis CI.")
class TestVista:
    from gempy.plot import vista as vs

    @pytest.fixture(scope='module')
    def vista_obj(self) -> vs.Vista:
        """Return a GemPy Vista instance with basic geomodel attached."""
        from gempy.plot import vista as vs

        geo_model = gp.create_data(
            [0, 2000, 0, 2000, 0, 2000], [50, 50, 50],
            path_o=input_path + '/input_data/tut_chapter1'
                                '/simple_fault_model_orientations.csv',
            path_i=input_path + '/input_data/tut_chapter1'
                                '/simple_fault_model_points.csv'
        )

        gp.set_series(
            geo_model,
            {"Fault_Series": 'Main_Fault',
             "Strat_Series": ('Sandstone_2', 'Siltstone', 'Shale', 'Sandstone_1')}
        )
        geo_model.set_is_fault(['Fault_Series'])
        gp.set_interpolator(geo_model)
        gp.compute_model(geo_model)
        # with open(os.path.dirname(__file__)+"input_data/geomodel_fabian_sol.p", "rb") as f:
        #     geo_model.solutions = load(f)

        return vs._Vista(geo_model)

    def test_set_bounds(self, vista_obj):
        vista_obj.set_bounds()

    def test_plot_surface_points(self, vista_obj):
        mesh = vista_obj.plot_surface_points("Shale")
        assert vista_obj._actor_exists(mesh[0])

    def test_plot_surface_points_all(self, vista_obj):
        meshes = vista_obj._plot_surface_points_all()
        for mesh in meshes:
            assert vista_obj._actor_exists(mesh)

    def test_plot_orientations(self, vista_obj):
        meshes = vista_obj.plot_orientations("Shale")
        for mesh in meshes:
            assert vista_obj._actor_exists(mesh)

    def test_plot_orientations_all(self, vista_obj):
        meshes = vista_obj._plot_orientations_all()
        for mesh in meshes:
            assert vista_obj._actor_exists(mesh)

    def test_get_surface(self, vista_obj):
        pv = pytest.importorskip("pyvista")

        surface = vista_obj.get_surface("Shale")
        assert type(surface) == pv.PolyData

    def test_plot_surface(self, vista_obj):
        meshes = vista_obj.plot_surface("Shale")
        for mesh in meshes:
            assert vista_obj._actor_exists(mesh)

    def test_plot_surfaces_all(self, vista_obj):
        meshes = vista_obj.plot_surfaces_all()
        for mesh in meshes:
            assert vista_obj._actor_exists(mesh)

    def test_plot_structured_grid_lith(self, vista_obj):
        pv = pytest.importorskip("pyvista")

        mesh = vista_obj.plot_structured_grid("lith")
        assert type(mesh[0]) == pv.StructuredGrid

    def TEST_plot_structured_grid_scalar(self, vista_obj):
        # These test are broken because some times mesh is a list and others are one single value as far as I saw
        # pretty much at random
        pv = pytest.importorskip("pyvista")

        mesh = vista_obj.plot_structured_grid("scalar")
        assert type(mesh[0]) == pv.core.pointset.StructuredGrid

    def TEST_plot_structured_grid_scalar2(self, vista_obj):
        mesh = vista_obj.plot_structured_grid("scalar")
        shape = vista_obj.model.grid.regular_grid.values.shape[0]
        assert mesh[0].points.shape[0] == shape

    # def test_plot_structured_grid_values(vista_obj):
    #     vista_obj.plot_structured_grid("values")
    #     assert type(vista_obj._actors[0]) == pv.StructuredGrid
