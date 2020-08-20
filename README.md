# <p align="left"><img src="docs/logos/gempy1.png" width="300"></p>

> Open-source, implicit 3D structural geological modeling in Python for uncertainty analysis.


[![PyPI](https://img.shields.io/badge/python-3-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/badge/pypi-1.0-blue.svg)](https://pypi.org/project/gempy/)
[![license: LGPL v3](https://img.shields.io/badge/license-LGPL%20v3-blue.svg)](https://github.com/cgre-aachen/gempy/blob/master/LICENSE)
[![Documentation Status](https://assets.readthedocs.org/static/projects/badges/passing-flat.svg)](http://docs.gempy.org)
[![Travis Build](https://travis-ci.org/cgre-aachen/gempy.svg?branch=master)](https://travis-ci.org/github/cgre-aachen/gempy/branches)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/cgre-aachen/gempy/master)
[![DOI](https://zenodo.org/badge/96211155.svg)](https://zenodo.org/badge/latestdoi/96211155)
[![DOCKER](https://img.shields.io/docker/cloud/automated/leguark/gempy.svg)](https://cloud.docker.com/repository/docker/leguark/gempy)

<p align="center"><img src="docs/source/images/model_examples.png" width="800"></p>

## Overview

`GemPy` is a Python-based, **open-source geomodeling library**. It is
capable of constructing complex **3D geological models** of folded
structures, fault networks and unconformities, based on the underlying
powerful **implicit representation** approach. `GemPy` was designed from the
ground up to support easy embedding in probabilistic frameworks for the
uncertainty analysis of subsurface structures.

Check out the documentation either in the main website [gempy.org](https://www.gempy.org/)
(better option), or the specific [docs site](http://docs.gempy.org/).

## Table of Contents

* [Features](#feat)
    * [Geological objects that can be modeled with `gempy`](#geology)
    * [Underlying interpolation approach](#interpolation)
    * [Visualization](#visualization)
    * [Stochastic geological modeling](#stochastic_modeling)
* [Installation](#installation)
* [Documentation](#doc)
* [References](#ref)

<a name="feat"></a>


## Features

<a name="geology"></a>
### Geological objects

<!-- Start with an intro to the geological features - instead of algo -->

`GemPy` enables the modeling of complex 3D geological settings,
on par with many commercial geomodeling packages, including:

- Multiple conformal layers (e.g. sequences of sedimentary layers)
- Several sequences of layers, with conformal continuation or unconformities
- Magmatic bodies of (almost) arbitrary shapes
- Faults (offset calculated automatically from affected geological objects)
- Full fault networks (faults affecting faults)
- Folds (affecting single layers or entire layer stacks, including overturned and recumbent folds)

The combination of these elements allows for the generation of realistic
3-D geological models in most typical geological settings.

<!-- Note: we should inlcude here links to models and/or publications where
gempy has been used for realistic models!  -->

<a name="interpolation"></a>
### Interpolation approach

The generation of complex structural settings is based on the powerful
interpolation algorithm underlying `GemPy`, a universal cokriging method
devoised by Lajaunie et al. (1997) and extended by Calcagno et al. (2008).
This method is used to interpolate a 3D scalar field, such that geologically
significant interfaces are isosurfces in this field.

The algorithm allows for a direct integration of two of the most relevant
geological input data types:

- *Surface contact points*: 3D coordinates of points marking the boundaries
between different features (e.g. layer interfaces, fault planes, unconformities).
- *Orientation measurements*: Orientation of the poles perpendicular to
the dipping of surfaces at any point in the 3D space.

`GemPy` also allows for the definition of topological elements such as
combining multiple stratigraphic sequences and
complex fault networks to be considered in the modeling process.

<p align="center"><img src="docs/source/images/modeling_principle.png" width="600"></p>

<a name="visualization"></a>
### Integrated visualization

Models generated with `GemPy` can be visualized in several ways:

- direct visualization of 2D model sections (or geological maps) using
`matplotlib`, including hillshading and other options for intuitive
representation of results;
- interactive 3D visualization and model input manipulation using the
Visualization Toolkit (VTK);
- We also actively develop a link to the fantastic [
`pyvista`](https://www.pyvista.org) project
for even better visualization and model interaction in 3D.

In addition to visualization, the generated models can be exported
in a variety of ways:

- Export of VTK files for further visualization and processing in other
software such as ParaView;
- Export of triangulated surface meshes (e.g. for further processing in
meshing programs);
- Export of images (e.g. geological maps) for

We are also currently working on a tighter integration with several
meshing libraries, notably [`CGAL`](https://www.cgal.org) and
[`gmsh`](https://gmsh.info). In addition, we have established
links to several other open-source libraries, including [`pygimli`](https://www.pygimli.org)
for geophysical modeling and inversion. In the current state, however, these
links have to be considered as highly experimental and they are not yet
part of the stable release. If you are interested in these features,
feel free to contact us.

<p align="center"><img src="docs/source/images/vtkFault.png" width="600"></p>

<a name="stochastic_modeling"></a>
### Stochastic geological modeling

One of the most advanced features that sets `gempy` also apart from
available commercial packages is the full integration of stochastic
geological modeling methods.
`GemPy` was designed from the ground up to support stochastic geological
modeling for uncertainty analysis (e.g. Monte Carlo simulations, Bayesian
inference). This was achieved by writing `GemPy`'s core architecture
using the numerical computation library [`Theano`](http://deeplearning.net/software/theano/)
to couple it with the probabilistic programming
framework [`PyMC3`](https://pymc-devs.github.io/pymc3/notebooks/getting_started.html).
This enables the use of advanced sampling methods (e.g. Hamiltonian Monte
Carlo) and is of particular relevance when considering uncertainties in
the model input data and making use of additional secondary information
in a Bayesian inference framework.

We can, for example, include uncertainties with respect to the z-position
of layer boundaries in the model space. Simple Monte Carlo simulation
via PyMC will then result in different model realizations.

<!-- Removed images as wobble.gif not anymore included - TODO: include
new images to represent stochastic modeling capabilities!

<p align="center"><img src="docs/source/images/gempy_zunc.png" height="300">
<img src="docs/source/images/model_wobble.gif" height="300"></p>

-->

Theano allows the automated computation of gradients, opening the door to
 the use of advanced gradient-based sampling methods
coupling *GeMpy* and
[PyMC3](https://pymc-devs.github.io/pymc3/notebooks/getting_started.html)
for advanced stochastic modeling. Also, the use of Theano allows making
use of GPUs through cuda (see the Theano documentation for more information.

Making use of vtk interactivity and `Qgrid` (https://github.com/quantopian/qgrid) ,
`GemPy` provides a functional interface to interact with input data and models.

For a more detailed elaboration of the theory behind `GemPy`, we refer to the
**open access scientific publication**:
[*"GemPy 1.0: open-source stochastic geological modeling and inversion"*
by de la Varga et al. (2018)](https://www.geosci-model-dev.net/12/1/2019/gmd-12-1-2019.pdf).

## Installation
<a name="installation"></a>
### Installing GemPy

We provide the latest release version of `GemPy` via the **Conda** and **PyPi** package services. We highly
recommend using PyPi, as it will take care of automatically installing all dependencies.

`$ pip install gempy`

You can also visit `PyPi <https://pypi.org/project/gempy/>`_, or
`GitHub <https://github.com/cgre-aachen/gempy>`

For more details in the installation check:
 `Installation <http://docs.pyvista.org/getting-started/installation.html#install-ref.>`


<a name="depend"></a>

<a name="doc"></a>
## Documentation

Extensive documentation for `GemPy` is hosted at [gempy.readthedocs.io](http://gempy.readthedocs.io/),
explaining its capabilities, [the theory behind it](https://www.geosci-model-dev.net/12/1/2019/) and 
providing detailed [tutorials](https://www.gempy.org/tutorials) on how to use it.

<a name="ref"></a>
## References

* de la Varga, M., Schaaf, A., and Wellmann, F.: GemPy 1.0: open-source stochastic geological modeling and inversion, Geosci. Model Dev., 12, 1-32, https://doi.org/10.5194/gmd-12-1-2019, 2019
* Calcagno, P., Chilès, J. P., Courrioux, G., & Guillen, A. (2008). Geological modelling from field data and geological knowledge: Part I. Modelling method coupling 3D potential-field interpolation and geological rules. Physics of the Earth and Planetary Interiors, 171(1-4), 147-157.
* Lajaunie, C., Courrioux, G., & Manuel, L. (1997). Foliation fields and 3D cartography in geology: principles of a method based on potential interpolation. Mathematical Geology, 29(4), 571-584.
* Wellmann, F., Schaaf, A., de la Varga, M., & von Hagke, C. (2019). [From Google Earth to 3D Geology Problem 2: Seeing Below the Surface of the Digital Earth](
https://www.sciencedirect.com/science/article/pii/B9780128140482000156).
In Developments in Structural Geology and Tectonics (Vol. 5, pp. 189-204). Elsevier.
