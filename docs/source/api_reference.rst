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
    Solutions
    RawArraysSolution

Helper Classes
--------------
.. currentmodule:: gempy.core.data
.. autosummary::
    :toctree: Helper Classes
    :template: class.rst

    ImporterHelper

Enumerators
-----------
.. currentmodule:: gempy.core.data
.. autosummary::
    :toctree: Modeling Classes
    :template: class.rst
    
    StackRelationType
