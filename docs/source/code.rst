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

    activate_interactive_df
    compute_model
    compute_model_at
    create_data
    create_model
    init_data
    map_series_to_surfaces
    map_stack_to_surfaces
    read_csv
    update_additional_data
    get_data
    get_surfaces
    get_additional_data
    get_interpolator
    get_th_fn


Model
-----
.. currentmodule:: gempy.core.model
.. autosummary::
    :toctree: Model
    :template: class.rst

    Project
    ImplicitCoKriging


Plot
----
.. currentmodule:: gempy
.. autosummary::
    :toctree: Plot
    :template: base.rst

    plot_2d
    plot_3d


Data
----
.. currentmodule:: gempy.core.data_modules.stack
.. autosummary::
    :toctree: Data
    :template: class.rst

    Stack
    Series
    Faults


.. currentmodule:: gempy.core.data
.. autosummary::
    :toctree: Data
    :template: class.rst


    Surfaces
    Structure
    KrigingParameters
    Options
    AdditionalData
    MetaData
    Grid


Grids
-----

.. currentmodule:: gempy.core.grid_modules.grid_types
.. autosummary::
    :toctree: Data/Grid
    :template: class.rst

    RegularGrid
    CustomGrid

.. currentmodule:: gempy.core.grid_modules.topography
.. autosummary::
    :toctree: Data/Grid
    :template: class.rst

    Topography

.. currentmodule:: gempy.core.grid_modules.grid_types
.. autosummary::
    :toctree: Data/Grid
    :template: class.rst

    Sections
    CenteredGrid


Geometric Data
--------------

.. currentmodule:: gempy.core.data_modules.geometric_data
.. autosummary::
    :toctree: Data
    :template: class.rst

    SurfacePoints
    Orientations
    RescaledData


Interpolator
------------
.. currentmodule:: gempy.core.interpolator
.. autosummary::
    :toctree: Interpolator
    :template: class.rst

    InterpolatorModel
    InterpolatorGravity


Solution
--------
.. currentmodule:: gempy.core.solution
.. autosummary::
    :toctree: Solution
    :template: class.rst

    Solution