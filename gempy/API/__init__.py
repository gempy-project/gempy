# Initialization API
from .initialization_API import (
    create_data_legacy,
    create_geomodel
)

# Compute API
from .compute_API import (
    compute_model,
    compute_model_at
)

# Map stack to surfaces API
from .map_stack_to_surfaces_API import map_stack_to_surfaces

# Grid API
from .grid_API import (
    set_active_grid,
    set_custom_grid,
    set_section_grid,
    set_topography_from_random
)

# Examples generator
from .examples_generator import generate_example_model

# Faults API
from .faults_API import (
    set_fault_relation,
    set_is_fault,
    set_is_finite_fault
)

# Data manipulation: points
from ..modules.data_manipulation.manipulate_points import (
    add_orientations,
    add_surface_points,
    delete_orientations,
    delete_surface_points,
    modify_orientations,
    modify_surface_points
)

# Data manipulation: orientations from surface points
from ..modules.data_manipulation.orientations_from_surface_points import (
    create_orientations_from_surface_points
)

# Data manipulation: structural frame
from ..modules.data_manipulation.manipulate_structural_frame import (
    add_structural_group,
    remove_structural_group_by_index,
    remove_structural_group_by_name
)

__all__ = [
    'create_data_legacy', 'create_geomodel', 'compute_model', 'compute_model_at', 'map_stack_to_surfaces',
    'set_section_grid', 'set_active_grid', 'set_topography_from_random', 'set_custom_grid',
    'generate_example_model', 'set_fault_relation', 'set_is_fault', 'set_is_finite_fault',
    'add_surface_points', 'add_orientations', 'delete_surface_points', 'delete_orientations',
    'create_orientations_from_surface_points', 'modify_surface_points', 'modify_orientations',
    'add_structural_group', 'remove_structural_group_by_index', 'remove_structural_group_by_name'
]
