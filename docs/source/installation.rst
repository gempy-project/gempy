.. role:: raw-html-m2r(raw)
   :format: html

Installation
------------

Installing ``GemPy``
^^^^^^^^^^^^^^^^^^^^

The latest release version of ``GemPy`` is available via **PyPi**. To install the core version, simply use pip:

.. code-block:: bash

    $ pip install gempy

This will install ``GemPy`` with the minimal required dependencies.

Enhanced Installation Options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For additional features, ``GemPy`` offers enhanced installation options:

- Base Features: For the majority of functionalities, install with base dependencies:

  .. code-block:: bash

      $ pip install gempy[base]

- Optional Features: To access optional features such as data download support, install with optional dependencies:

  .. code-block:: bash

      $ pip install gempy[opt]

- Development Version: For development purposes, including testing tools:

  .. code-block:: bash

      $ pip install gempy[dev]

Note: Some functionalities in ``GemPy`` require `PyTorch` for efficient calculations and `pyvista`. These libraries need to be installed separately as per user requirement due to their complex nature.

Manual Installation
^^^^^^^^^^^^^^^^^^^

For the latest, cutting-edge version of ``GemPy``, you can clone the repository and install it manually:

1. Clone the repository:

   .. code-block:: bash

       $ git clone https://github.com/cgre-aachen/gempy.git

2. Navigate to the root of the cloned repository and install using pip:

   .. code-block:: bash

       $ pip install -e .

Dependencies
^^^^^^^^^^^^

``GemPy`` relies on numerous open-source libraries. Core dependencies are automatically installed with the package. Enhanced functionalities require additional dependencies, structured as follows:

- Core Dependencies: Installed with the basic package.
- Base Dependencies: Additional libraries for extended functionalities (`base-requirements.txt`).
- Optional Dependencies: Libraries for optional features (`optional-requirements.txt`).
- Development Dependencies: Tools needed for development (`dev-requirements.txt`).

