import os
import pytest
import gempy as gp
import numpy as np
import matplotlib.pyplot as plt

input_path = os.path.dirname(__file__) + '/../../notebooks/data'


@pytest.mark.skipif("TRAVIS" in os.environ and os.environ["TRAVIS"] == "true",
                    reason="Skipping this test on Travis CI.")
class TestVista:
    @pytest.fixture(scope='module')
    def vista_object_only_data(self, one_fault_model_no_interp):
        """
        Args:
            one_fault_model:
        """
        from gempy.plot.vista import GemPyToVista

        return GemPyToVista(one_fault_model_no_interp,
                            #plotter_type='background')
                            plotter_type='basic',  off_screen=True)

    @pytest.fixture(scope='module')
    def vista_object_computed(self, one_fault_model_solution):
        """
        Args:
            one_fault_model_solution:
        """
        from gempy.plot.vista import GemPyToVista

        return GemPyToVista(one_fault_model_solution, plotter_type='basic',
                            off_screen=True
                            )

    @pytest.fixture(scope='module')
    def vista_object_computed_topo(self, one_fault_model_solution):
        """
        Args:
            one_fault_model_solution:
        """

        from gempy.plot.vista import GemPyToVista
        one_fault_model_solution.update_additional_data()
        one_fault_model_solution.update_to_interpolator()
        one_fault_model_solution.set_topography()
        gp.compute_model(one_fault_model_solution)

        return GemPyToVista(one_fault_model_solution, plotter_type='basic', off_screen=True)

    def test_set_bounds(self, vista_object_only_data):
        """
        Args:
            vista_object_only_data:
        """
        vista_object_only_data.set_bounds()

    def test_select_surface_points(self, vista_object_only_data):
        """
        Args:
            vista_object_only_data:
        """
        sp = vista_object_only_data._select_surfaces_data(data_df=vista_object_only_data.model._surface_points.df,
                                                          surfaces='all')
        np.testing.assert_almost_equal(sp.loc[4, 'X_r'],  0.486942, 5)

        sp2 = vista_object_only_data._select_surfaces_data(data_df=vista_object_only_data.model._surface_points.df,
                                                           surfaces=['Sandstone_2'])
        with pytest.raises(KeyError):
            sp2.loc[4, 'X_r']

    def test_plot_surface_points_poly_live(self, vista_object_only_data):
        """
        Args:
            vista_object_only_data:
        """
        vista_object_only_data.live_updating = True
        vista_object_only_data.plot_surface_points()
        print('foo')

    def test_plot_surface_points_poly_static(self, vista_object_only_data):
        """
        Args:
            vista_object_only_data:
        """
        vista_object_only_data.live_updating = False
        vista_object_only_data.plot_surface_points()
        print('foo')

    def test_plot_data_static(self, vista_object_only_data):
        vista_object_only_data.plot_data()
        print('foo')

    def test_plot_orientations_poly_live(self, vista_object_only_data):
        """
        Args:
            vista_object_only_data:
        """
        vista_object_only_data.live_updating = True
        vista_object_only_data.plot_orientations()
        print('foo')

    def test_plot_orientations_poly_static(self, vista_object_only_data):
        """
        Args:
            vista_object_only_data:
        """
        vista_object_only_data.live_updating = False
        vista_object_only_data.plot_orientations()
        print('foo')

    def test_plot_surfaces(self, vista_object_computed):
        """
        Args:
            vista_object_computed:
        """
        a = vista_object_computed.plot_surfaces()
        print(a)
        aa = vista_object_computed.plot_surfaces()
        print(aa)
        print('foo')

    def test_plot_topography_high(self, vista_object_computed_topo):
       # vista_object_only_data.model.set_topography()
        """
        Args:
            vista_object_computed_topo:
        """
        vista_object_computed_topo.plot_topography()

    def test_plot_topography(self, vista_object_computed_topo):
        """
        Args:
            vista_object_computed_topo:
        """
        vista_object_computed_topo.plot_topography(scalars='geomap')
        print('foo')

    def test_plot_regular_grid_lith(self, vista_object_computed):
        """
        Args:
            vista_object_computed:
        """
        vista_object_computed.plot_structured_grid('lith', render_topography=False,
                                                   opacity=.8)
        img = vista_object_computed.p.show(screenshot=True)
        plt.imshow(img[1])
        plt.show()

        print('foo')

    def test_plot_regular_grid_scalar_0(self, vista_object_computed):
        # Add all scalar fields to the pyvista object and plot lith
        """
        Args:
            vista_object_computed:
        """
        vista_object_computed.plot_structured_grid('all', render_topography=False)

        # Change active scalar to stratigraphy
        vista_object_computed.set_active_scalar_fields(
            'sf_Strat_Series')

        # Change the color map to the lithology cmap
        vista_object_computed.set_scalar_field_cmap('lith', vista_object_computed.regular_grid_actor)

        # Set active scalar back to lith
        vista_object_computed.set_active_scalar_fields('lith')

        print('foo')

    def test_plot_regular_grid_scalar_topo(self, vista_object_computed_topo):
        """
        Args:
            vista_object_computed_topo:
        """
        vista_object_computed_topo.plot_structured_grid('lith', render_topography=False)
        print('foo')

    def test_plot_regular_grid_scalar_1(self, vista_object_computed):
        """
        Args:
            vista_object_computed:
        """
        vista_object_computed.plot_structured_grid('scalar', render_topography=True)
        print('foo')

    def test_plot_regular_grid_select_field(self, vista_object_computed):
        """
        Args:
            vista_object_computed:
        """
        vista_object_computed.plot_structured_grid('lith')
        with pytest.raises(AttributeError):
            vista_object_computed.set_active_scalar_fields(scalar_field='scalar')
        # vista_object_computed.plot_structured_grid('scalar')
        print('foo')