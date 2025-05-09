Code
====

.. toctree::
   :maxdepth: 3


GemPy API
---------
.. currentmodule:: gempy
.. autosummary::
    :toctree: GemPy API
    :template: base.rst

    compute_model
    create_geomodel
    map_stack_to_surfaces
    structural_elements_from_borehole_set
    set_section_grid
    set_active_grid
    set_topography_from_random
    set_topography_from_file
    set_topography_from_subsurface_structured_grid
    set_topography_from_arrays
    set_custom_grid
    set_centered_grid
    generate_example_model
    set_fault_relation
    set_is_fault
    set_is_finite_fault
    add_surface_points
    add_orientations
    delete_surface_points
    delete_orientations
    create_orientations_from_surface_points_coords
    modify_surface_points
    modify_orientations
    add_structural_group
    remove_structural_group_by_index
    remove_structural_group_by_name
    remove_element_by_name
    calculate_gravity_gradient


Plot
----
.. currentmodule:: gempy_viewer
.. autosummary::
    :toctree: Plot
    :template: base.rst

    plot_2d
    plot_3d

Data Classes
============

.. toctree::
    :maxdepth: 3

Modeling Classes
----------------
.. module:: gempy.core.data
.. autosummary::
    :toctree: Modeling Classes
    :template: class.rst

    GeoModel
    StructuralFrame
    StructuralGroup
    StructuralElement
    SurfacePointsTable
    OrientationsTable
    InterpolationOptions
    Grid
    Topography
    Solutions
    RawArraysSolution
    FaultsData
    GeophysicsInput
    Transform

Helper Classes
--------------
.. currentmodule:: gempy.core.data
.. autosummary::
    :toctree: Helper Classes
    :template: class.rst

    ImporterHelper
    GemPyEngineConfig
    ColorsGenerator

Enumerators
-----------
.. currentmodule:: gempy.core.data
.. autosummary::
    :toctree: Modeling Classes
    :template: class.rst
    
    StackRelationType
    GlobalAnisotropy
    AvailableBackends
    FaultsRelationSpecialCase
