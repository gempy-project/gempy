Code
====

.. toctree::
   :maxdepth: 4

Model
-----
.. currentmodule:: model
.. autosummary::
    :toctree: Model

    Model
    DataMutation

Data
----
.. currentmodule:: data
.. autosummary::
    :toctree: Data

    Grid
    Faults
    Series
    Surfaces
    GeometricData
    SurfacePoints
    Orientations
    RescaledData
    Structure
    KrigingParameters
    Options
    AdditionalData

Interpolator
------------
.. currentmodule:: interpolator
.. autosummary::
    :toctree: Interpolator

    Interpolator
    InterpolatorModel
    InterpolatorGravity

Solution
--------
.. currentmodule:: solution
.. autosummary::
    :toctree: Solution

    Solution


Front end
---------
.. currentmodule:: gempy_api
.. autosummary::
    :toctree: GemPy API

    activate_interactive_df
    compute_model
    compute_model_at
    create_data
    create_model
    get_additional_data
    get_interpolator
    get_surfaces
    get_th_fn
    init_data
    load_model
    load_model_pickle
    map_series_to_surfaces
    read_csv
    save_model
    save_model_to_pickle
    set_interpolation_data
    set_orientation_from_surface_points
    set_series
    update_additional_data
