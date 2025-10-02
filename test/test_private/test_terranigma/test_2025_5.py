import dotenv
import matplotlib.pyplot as plt

import gempy as gp
import numpy as np
from gempy_engine.core.data.stack_relation_type import StackRelationType

dotenv.load_dotenv()

def test_2025_5():
    data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
    path_to_data = data_path + "/data/input_data/jan_models/"
    # Create a GeoModel instance

    n_levels = 4
    extent = 1000

    levels = []
    n_cells = []
    for min_level in [0, 2]:
        data = gp.create_geomodel(
            project_name='fault',
            extent=[0, extent, 0, extent, 0, extent],
            refinement=n_levels,
            importer_helper=gp.data.ImporterHelper(
                path_to_orientations=path_to_data + "model5_orientations.csv",
                path_to_surface_points=path_to_data + "model5_surface_points.csv"
            )
        )

        # Map geological series to surfaces
        gp.map_stack_to_surfaces(
            gempy_model=data,
            mapping_object={
                "Fault_Series": 'fault',
                "Strat_Series": ('rock2', 'rock1')
            }
        )

        # Define fault groups
        data.structural_frame.structural_groups[0].structural_relation = StackRelationType.FAULT
        data.structural_frame.fault_relations = np.array([[0, 1], [0, 0]])

        # Compute the geological model
        data.interpolation_options.evaluation_options.octree_min_level = min_level
        data.interpolation_options.evaluation_options.octree_error_threshold = 0.2
        gp.compute_model(data)

        levels.append([
            k.outputs_centers[-1].grid.octree_grid.dx / data.input_transform.scale[0]
            for k in data.solutions.octrees_output
        ])
        n_cells.append([
            len(k.outputs_centers[-1].grid.octree_grid.values)
            for k in data.solutions.octrees_output
        ])

    print(f"min level 0: {n_cells[0]} cells at {levels[0]}m")
    print(f"min level 2: {n_cells[1]} cells at {levels[1]}m")


if __name__ == "__main__":
    test_2025_5()