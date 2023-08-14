from .initialization_API import create_data_legacy, create_geomodel
from .compute_API import compute_model, compute_model_at
from .map_stack_to_surfaces_API import map_stack_to_surfaces
from .grid_API import set_section_grid, set_active_grid, set_topography_from_random, set_custom_grid
from .examples_generator import generate_example_model
from .faults_API import set_fault_relation, set_is_fault, set_is_finite_fault
from .data_manipulation_API import add_surface_points, add_orientations, delete_surface_points, delete_orientations