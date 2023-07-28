import gempy as gp
import gempy_viewer as gpv
from gempy.core.data.enumerators import ExampleModel
from gempy_engine.modules.octrees_topology.octrees_topology_interface import ValueType
from plugins.plotting.helper_functions import plot_block_and_input_2d


def test_backends():
    geo_model: gp.GeoModel = gp.generate_example_model(
        example_model=ExampleModel.ONE_FAULT,
        compute_model=False
    )

    geo_model.interpolation_options.number_octree_levels = 4
    gp.compute_model(geo_model)

    gpv.plot_3d(geo_model, show_data=True, show_boundaries=True, show_lith=False, image=True)
