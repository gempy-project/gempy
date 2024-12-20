.. GeMpy documentation master file, created by
   sphinx-quickstart on Wed Dec 14 12:44:40 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. ../logos/gempy1.png
   :width: 30%

About
=====
Open-source software for implicit 3D structural geological modeling in Python.
******************************************************************************

Overview
--------

``GemPy`` is a Python-based, community-driven, **open-source geomodeling library**. It is
capable of constructing complex **3D geological models** including various features such as
fold structures, fault networks and unconformities, based on an underlying
powerful **implicit** approach. From the ground up, ``GemPy`` was designed to be easily embedded in probabilistic frameworks for conducting
uncertainty analysis regarding subsurface structures.

.. Check out the documentation either in `gempy.org <https://www.gempy.org/>`_
 (better option), or `read the docs <http://gempy.readthedocs.io/>`_.

3D models created with GemPy may look like this:

.. image:: ./_images/perth_example.png

Contents:

.. toctree::
   :maxdepth: 2

   self
   installation

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   tutorials/index
   examples/index

.. toctree::
    :maxdepth: 2
    :caption: External examples
    
    external/external_examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api_reference


Features
--------

Geological features
^^^^^^^^^^^^^^^^^^^


.. raw:: html

   <!-- Start with an intro to the geological features - instead of algo -->



``GemPy`` is capable of modeling complex 3D geological scenarios, including:

* Multiple conformal layers (e.g. sequences of sedimentary layers)
* Several sequences of layers, with conformal continuation or unconformities
* Magmatic bodies of (almost) arbitrary shapes
* Faults (offset calculated automatically from affected geological objects)
* Full fault networks (faults affecting faults)
* Folds (affecting single layers or entire layer stacks, including overturned and recumbent folds)

Combining these elements in GemPy allows for the generation of realistic
3D geological models, on a par with most commercial geomodeling software.

.. raw:: html

   <!-- Note: we should inlcude here links to models and/or publications where
   gempy has been used for realistic models!  -->



Interpolation approach
^^^^^^^^^^^^^^^^^^^^^^

The generation of complex structural settings is based on the powerful
interpolation algorithm underlying ``GemPy``\ , a unviersal cokriging method
devoised by `Lajaunie et al. (1997)` and extended by `Calcagno et al. (2008)`\ .
This method is used to interpolate a 3D scalar field, such that geologically
significant interfaces are isosurfces in this field.

The algorithm allows for a direct integration of two of the most relevant
geological input data types:


* **Surface contact points**\ : 3D coordinates of points marking the boundaries
  between different features (e.g. layer interfaces, fault planes, unconformities).
* **Orientation measurements**\ : Orientation of the poles perpendicular to
  the dipping of surfaces at any point in the 3D space.

``GemPy`` also allows for the definition of topological elements such as
combining multiple stratigraphic sequences and
complex fault networks to be considered in the modeling process.

.. image:: ./_images/modeling_principle.png



Integrated visualization
^^^^^^^^^^^^^^^^^^^^^^^^

Models generated with ``GemPy`` can be visualized in several ways:


* direct visualization of 2D model sections (or geological maps) using
  ``matplotlib``, including hillshading and other options for intuitive
  representation of results;
* interactive 3D visualization and model input manipulation using the
  Visualization Toolkit (VTK);
* We also actively develop a link to the fantastic
  `pyvista <https://www.pyvista.org>`_ project
  for even better visualization and model interaction in 3D.

In addition to visualization, the generated models can be exported
in a variety of ways:


* Export of VTK files for further visualization and processing in other
  software such as ParaView;
* Export of triangulated surface meshes (e.g. for further processing in
  meshing programs);
* Export of images (e.g. geological maps).

We are also currently working on a tighter integration with several
meshing libraries, notably `CGAL <https://www.cgal.org>`_ and `gmesh <https://gmsh.info>`_. In addition, we have
established links to several other open-source libraries, including `pygiml <https://www.pygimli.org>`_
for geophysical modeling and inversion. In the current state, however, these
links have to be considered as highly experimental and they are not yet
part of the stable release. If you are interested in these features,
feel free to contact us.

.. image:: _images/vtkFault.png
   :target: https://cgre-aachen.github.io/gempy/_images/sphx_glr_ch1_1_basics_009.png
   :width: 70%


Stochastic geological modeling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One of the most advanced features that sets ``GemPy`` also apart from
available commercial packages is the full integration of stochastic
geological modeling methods.
``GemPy`` was designed from the ground up to support stochastic geological
modeling for uncertainty analysis (e.g. Monte Carlo simulations, Bayesian
inference). This was achieved by writing ``GemPy``'s core architecture
using the numerical computation library `aesara <http://deeplearning.net/software/aesara/>`_
to couple it with the probabilistic programming
framework `PyMC3 <https://pymc-devs.github.io/pymc3/notebooks/getting_started.html>`_.
This enables the use of advanced sampling methods (e.g. Hamiltonian Monte
Carlo) and is of particular relevance when considering uncertainties in
the model input data and making use of additional secondary information
in a Bayesian inference framework.

We can, for example, include uncertainties with respect to the z-position
of layer boundaries in the model space. Simple Monte Carlo simulation
via PyMC will then result in different model realizations.


.. raw:: html

   <!-- Removed images as wobble.gif not anymore included - TODO: include
   new images to represent stochastic modeling capabilities!

   <p align="center"><img src="docs/source/images/gempy_zunc.png" height="300">
   <img src="docs/source/images/model_wobble.gif" height="300"></p>

   -->


aesara allows the automated computation of gradients, opening the door to
the use of advanced gradient-based sampling methods
coupling ``GemPy`` and
`PyMC3 <https://pymc-devs.github.io/pymc3/notebooks/getting_started.html>`_
for advanced stochastic modeling. Also, the use of aesara allows making
use of GPUs through cuda (see the aesara documentation for more information.

Making use of vtk interactivity and `Qgrid <https://github.com/quantopian/qgrid>`_ ,
``GemPy`` provides a functional interface to interact with input data and models.

For a more detailed elaboration of the theory behind ``GemPy``\ , we refer to the
**open access scientific publication**\ :
`\ "GemPy 1.0: open-source stochastic geological modeling and inversion"
by de la Varga et al. (2019) <https://www.geosci-model-dev.net/12/1/2019/gmd-12-1-2019.pdf>`_.

References
----------

* de la Varga, M., Schaaf, A., and Wellmann, F.: GemPy 1.0: `open-source stochastic geological modeling and inversion,` Geosci. Model Dev., 12, 1–32, https://doi.org/10.5194/gmd-12-1-2019, 2019.
* Calcagno, P., Chilès, J. P., Courrioux, G., & Guillen, A. (2008). `Geological modelling from field data and geological knowledge: Part I. Modelling method coupling 3D potential-field interpolation and geological rules.` Physics of the Earth and Planetary Interiors, 171(1-4), 147-157.
* `Lajaunie, C., Courrioux, G., & Manuel, L. (1997). `Foliation fields and 3D cartography in geology: principles of a method based on potential interpolation.` Mathematical Geology, 29(4), 571-584.


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`


.. image:: _static/logos/logo_CGRE.png
   :width: 40%

.. image:: _static/logos/Terranigma.png
   :width: 40%