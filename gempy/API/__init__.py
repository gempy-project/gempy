from .initialization_API import create_data_legacy, create_geomodel
from .compute_API import compute_model
from .map_stack_to_surfaces_API import map_stack_to_surfaces
from .grid_API import set_section_grid, set_active_grid, set_topography_from_random
from .examples_generator import generate_example_model
from .faults_API import set_fault_relation, set_is_fault, set_is_finite_fault