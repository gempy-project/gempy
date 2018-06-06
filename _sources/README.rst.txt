.. role:: raw-html-m2r(raw)
   :format: html


:raw-html-m2r:`<p align="left"><img src="docs/logos/gempy1.png" width="300"></p>`
=====================================================================================

..

   Open-source, implicit 3D structural geological modeling in Python for uncertainty analysis.



.. image:: https://img.shields.io/badge/python-3-blue.svg
   :target: 
   :alt: PyPI


.. image:: https://img.shields.io/badge/pypi-1.0-blue.svg
   :target: 
   :alt: PyPI


.. image:: https://img.shields.io/badge/license-LGPL%20v3-blue.svg
   :target: 
   :alt: license: LGPL v3


.. image:: https://readthedocs.org/projects/gempy/badge/?version=latest
   :target: http://gempy.readthedocs.io/?badge=latest
   :alt: Documentation Status


.. image:: https://travis-ci.org/cgre-aachen/gempy.svg?branch=master
   :target: 
   :alt: Travis Build


.. image:: https://mybinder.org/badge.svg
   :target: https://mybinder.org/v2/gh/cgre-aachen/gempy/master
   :alt: Binder


.. image:: https://zenodo.org/badge/96211155.svg
   :target: https://zenodo.org/badge/latestdoi/96211155
   :alt: DOI



.. raw:: html

   <p align="center"><img src="docs/source/images/model_examples.png" width="800"></p>


What is it
----------

*GemPy* is a Python-based, open-source library for **implicitly generating 3D structural geological models**. It is capable of
constructing complex 3D geological models of folded structures, fault networks and unconformities. It was designed from the 
ground up to support easy embedding in probabilistic frameworks for the uncertainty analysis of subsurface structures.

Check out the documentaion either in `github pages <https://cgre-aachen.github.io/gempy/index.html>`_ (better option), or `read the docs <http://gempy.readthedocs.io/>`_.

Table of Contents
-----------------


* `Features <##Features>`_
* `Getting Started <##GettingStarted>`_

  * `Dependecies <###Dependecies>`_
  * `Installation <###Installation>`_

* `Documentation <##Documentation>`_
* `References <##References>`_

Features
--------

The core algorithm of *GemPy* is based on a universal cokriging interpolation method devised by
Lajaunie et al. (1997) and extended by Calcagno et al. (2008). Its implicit nature allows the user to automatically 
 generate complex 3D structural geological models through the interpolation of input data:


* *Surface contact points*\ : 3D coordinates of points marking the boundaries between different features (e.g. layer interfaces, fault planes, unconformities).
* *Orientation measurements*\ : Orientation of the poles perpendicular to the dipping of surfaces at any point in the 3D space.

*GemPy* also allows for the definition of topological elements such as combining multiple stratigraphic sequences and 
complex fault networks to be considered in the modeling process.


.. raw:: html

   <p align="center"><img src="docs/source/images/modeling_principle.png" width="600"></p>


*GemPy* itself offers direct visualization of 2D model sections via matplotlib
and in full, interactive 3D using the Visualization Toolkit (VTK). The VTK support also allow to the real time maniulation
 of the 3-D model, allowing for the exact modification of data. Models can also easily be exportes in VTK file format 
for further visualization and processing in other software such as ParaView.


.. raw:: html

   <p align="center"><img src="docs/source/images/gempy-animation.gif" width="600"></p>


*GemPy* was designed from the beginning to support stochastic geological modeling for uncertainty analysis (e.g. Monte Carlo simulations, Bayesian inference). This was achieved by writing *GemPy*\ 's core architecture
using the numerical computation library `Theano <http://deeplearning.net/software/theano/>`_ to couple it with the probabilistic programming framework `PyMC3 <https://pymc-devs.github.io/pymc3/notebooks/getting_started.html>`_.
This enables the use of advanced sampling methods (e.g. Hamiltonian Monte Carlo) and is of particular relevance when considering
uncertainties in the model input data and making use of additional secondary information in a Bayesian inference framework.

We can, for example, include uncertainties with respect to the z-position of layer boundaries
in the model space. Simple Monte Carlo simulation via PyMC will then result in different model realizations:


.. raw:: html

   <p align="center"><img src="docs/source/images/gempy_zunc.png" height="300"> <img src="docs/source/images/model_wobble.gif" height="300"></p>


Theano allows the automated computation of gradients opening the door to the use of advanced gradient-based sampling methods
coupling *GeMpy* and `PyMC3 <https://pymc-devs.github.io/pymc3/notebooks/getting_started.html>`_ for advanced stochastic modeling.
Also, the use of Theano allows making use of GPUs through cuda (see the Theano documentation for more information.

Making use of vtk interactivity and Qgrid (https://github.com/quantopian/qgrid) *GemPy* provides a functional interface to interact with input data and models.


.. raw:: html

   <p align="center"><a href="https://youtu.be/aA4MaHpLWVE?t=67"><img src="https://img.youtube.com/vi/aA4MaHpLWVE/0.jpg" width="600"></p>


For a more detailed elaboration of the theory behind *GemPy*\ , take a look at the upcoming scientific publication
*"GemPy 1.0: open-source stochastic geological modeling and inversion"* by de la Varga et al. (2018).

Besides the main functionality GemPy is powering currently some further projects:

Sandbox
^^^^^^^

New developments in the field of augmented reality, i.e. the superimposition of real and digital objects, offer interesting and diverse possibilities that have hardly been exploited to date.
 The aim of the project is therefore the development and realization of an augmented reality sandbox for interaction with geoscientific data and models.
In this project, methods are to be developed to project geoscientific data (such as the outcrop of a geological layer surface or geophysical measurement data) onto real surfaces.
Thus we extend existing methods in this field by essential aspects. The Augmented Reality Sandbox serves as a development environment for applying these techniques at a later point in time to
outcrops and on a field scale.
The AR Sandbox is based on a container filled with sand, the surface of which can be shaped as required. The topography of the sand surface is continuously scanned by a 3D sensor and a camera.
 In the computer the scanned surface is now blended with a digital geological 3D model (or other data) in real time and an image is calculated, which is projected onto the sand surface by means
  of a beamer. This results in an interactive model with which the user can interact in an intuitive way and which visualizes and comprehend complex three-dimensional facts in an accessible way.
For direct application in research, the simple visualization of geological outcrops in the second phase of the project is supplemented by the possibility of model creation and manipulation.
 For this purpose, a tool is implemented that allows a direct interaction with the data displayed in the sandbox: A position sensor within the tool provides its exact orientation in space,
 the position of the tool in the sandbox is recorded using simple image analysis methods. Thus it is possible to define points and orientations in an intuitive way, which are directly converted
  into implicit geological models by integrating GemPy.
In addition to applications in teaching and research, this development offers great potential as an interactive exhibit with high outreach for the geosciences thanks to its intuitive operation.
The finished sandbox can be used in numerous lectures and public events , but is mainly used as an interface to GemPy software and for rapid prototyping of implicit geological models.


.. raw:: html

   <p align="center"><img src="docs/source/images/Sandbox.gif" width="600"></p>


Remote Geomod: From GoogleEarth to 3-D Geology
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We support this effort here with a full 3-D geomodeling exercise
on the basis of the excellent possibilities offered by open global data sets, implemented in
GoogleEarth, and dedicated geoscientific open-source software and motivate the use of 3-D
geomodeling to address specific geological questions. Initial steps include the selection of
relevant geological surfaces in GoogleEarth and the analysis of determined orientation values
for a selected region This information is subsequently used
to construct a full 3-D geological model with a state-of-the-art interpolation algorithm. Fi-
nally, the generated model is intersected with a digital elevation model to obtain a geological
map, which can then be reimported into GoogleEarth.


.. raw:: html

   <p align="center"><img src="docs/source/images/ge.png" width="900"></p>


Getting Started
---------------

Dependecies
^^^^^^^^^^^

*GemPy* requires Python 3 and makes use of numerous open-source libraries:


* pandas
* tqdm
* scikit_image
* Theano
* matplotlib
* numpy
* pytest
* scipy
* ipython
* seaborn
* setuptools
* scikit_learn
* networkx

Optional:


* ``vtk>=7`` for interactive 3-D visualization 
* ``pymc`` or ``pymc3``
* ``steno3d`` 

Overall we recommend the use of a dedicated Python distribution, such as 
`Anaconda <https://www.continuum.io/what-is-anaconda>`_\ , for hassle-free package installation. 
We are curently working on providing GemPy also via Anaconda Cloud, for easier installation of 
its dependencies.

Installation
^^^^^^^^^^^^

We provide the latest release version of *GemPy* via the **Conda** and **PyPi** package services. We highly
recommend using either Conda or PyPi as both will take care of automatically installing all dependencies.

PyPi
~~~~

``$ pip install gempy``

Manual
~~~~~~

Otherwise you can clone the current repository by downloading is manually or by using Git by calling

``$ git clone https://github.com/cgre-aachen/gempy.git``

and then manually install it using the provided Python install file by calling

``$ python gempy/setup.py install``

in the cloned or downloaded repository folder. Make sure you have installed all necessary dependencies listed above before using *GemPy*.

Documentation
-------------

Extensive documentation for *GemPy* is hosted at `gempy.readthedocs.io <http://gempy.readthedocs.io/>`_\ ,
explaining its capabilities, `the theory behind it <http://gempy.readthedocs.io/Kriging.html>`_ and 
 providing detailed `tutorials <http://gempy.readthedocs.io/tutorial.html>`_ on how to use it.

References
----------


* Calcagno, P., Chilès, J. P., Courrioux, G., & Guillen, A. (2008). Geological modelling from field data and geological knowledge: Part I. Modelling method coupling 3D potential-field interpolation and geological rules. Physics of the Earth and Planetary Interiors, 171(1-4), 147-157.
* Lajaunie, C., Courrioux, G., & Manuel, L. (1997). Foliation fields and 3D cartography in geology: principles of a method based on potential interpolation. Mathematical Geology, 29(4), 571-584.
