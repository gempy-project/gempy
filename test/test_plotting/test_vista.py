import os
import pytest
import gempy as gp
import numpy as np

input_path = os.path.dirname(__file__) + '/../../notebooks/data'


@pytest.mark.skipif("TRAVIS" in os.environ and os.environ["TRAVIS"] == "true",
                    reason="Skipping this test on Travis CI.")
class TestVista:
    @pytest.fixture(scope='module')
    def vista_object(self, one_fault_model):
        """

        Args:
            model_horizontal_two_layers (gp.Model):

        Returns:

        """
        from gempy.plot.vista import GemPyToVista

        return GemPyToVista(one_fault_model, plotter_type='background')

    def test_set_bounds(self, vista_object):
        vista_object.set_bounds()

    def test_select_surface_points(self, vista_object):
        sp = vista_object._select_surface_points(surfaces='all')
        np.testing.assert_almost_equal(sp.loc[4, 'X_r'],  0.486942, 5)

        sp2 = vista_object._select_surface_points(surfaces=['Sandstone_2'])
        with pytest.raises(KeyError):
            sp2.loc[4, 'X_r']

    def test_plot_surface_points_poly(self, vista_object):
        vista_object.live_updating = True
        vista_object.plot_surface_points()
        vista_object.p.show()
        print('foo')