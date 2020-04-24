Code
====

.. toctree::
   :maxdepth: 2


GemPy API
---------
.. currentmodule:: gempy
.. autosummary::
    :toctree: GemPy API

    activate_interactive_df
    compute_model
    compute_model_at
    create_data
    create_model
    init_data
    map_series_to_surfaces
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

    Project
    ImplicitCoKriging


Plot
____
.. currentmodule:: gempy.plot.plot_api
.. autosummary::
    :toctree: Plot

    plot_2d
    plot_3d


Data
----
.. currentmodule:: gempy.core.data
.. autosummary::
    :toctree: Data

    Grid
    Surfaces
    Structure
    KrigingParameters
    Options
    AdditionalData
    MetaData

.. currentmodule:: gempy.core.data_modules.geometric_data
.. autosummary::
    :toctree: Data

    SurfacePoints
    Orientations
    RescaledData

.. currentmodule:: gempy.core.data_modules.stack
.. autosummary::
    :toctree: Data

    Stack
    Series
    Faults


Interpolator
------------
.. currentmodule:: gempy.core.interpolator
.. autosummary::
    :toctree: Interpolator

    InterpolatorModel
    InterpolatorGravity


Solution
--------
.. currentmodule:: gempy.core.solution
.. autosummary::
    :toctree: Solution

    Solution