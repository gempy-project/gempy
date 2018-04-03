# <p align="left"><img src="docs/logos/gempy1.svg" width="200"></p>

> Open-source, implicit 3D structural geological modeling in Python for uncertainty analysis.


[![PyPI](https://img.shields.io/badge/python-3-blue.svg)]()
[![PyPI](https://img.shields.io/badge/pypi-1.0-blue.svg)]()
[![PyPI](https://img.shields.io/badge/conda-1.0-blue.svg)]()
[![license: LGPL v3](https://img.shields.io/badge/license-LGPL%20v3-blue.svg)]()
[![Documentation Status](https://readthedocs.org/projects/gempy/badge/?version=latest)](http://gempy.readthedocs.io/?badge=latest)
[![Travis Build](https://travis-ci.org/cgre-aachen/gempy.svg?branch=master)]()
[![DOI](https://zenodo.org/badge/96211155.svg)](https://zenodo.org/badge/latestdoi/96211155)


<p align="center"><img src="docs/source/images/model_examples.png" width="800"></p>

*GemPy* is a Python-based, open-source library for implicitly generating 3D structural geological models. It is capable of 
constructing complex 3D geological models of folded structures, fault networks and unconformities. It was designed from the 
ground up to support easy embedding in probabilistic frameworks for the uncertainty analysis of subsurface structures.

## Table of Contents

* [Features](##Features)
* [Getting Started](##GettingStarted)
    * [Dependecies](###Dependecies)
    * [Installation](###Installation)
* [Documentation](##Documentation)
* [References](##References)

## Features

The core algorithm of *GemPy* is based on a universal cokriging interpolation method devised by
Lajaunie et al. (1997) and extended by Calcagno et al. (2008). Its implicit nature allows the user to automatically 
 generate complex 3D structural geological models through the interpolation of input data:

- *Surface contact points*: 3D coordinates of points marking the boundaries between different features (e.g. layer interfaces, fault planes, unconformities).
- *Orientation measurements*: Orientation of the poles perpendicular to the dipping of surfaces at any point in the 3D space.

*GemPy* also allows for the definition of topological elements such as combining multiple stratigraphic sequences and 
complex fault networks to be considered in the modeling process.

<p align="center"><img src="docs/source/images/modeling_principle.png" width="600"></p>

*GemPy* itself offers direct visualization of 2D model sections via matplotlib
and in full, interactive 3D using the Visualization Toolkit (VTK). The VTK support also allow to the real time maniulation
 of the 3-D model, allowing for the exact modification of data. Models can also easily be exportes in VTK file format 
for further visualization and processing in other software such as ParaView.

<p align="center"><img src="docs/source/images/gempy-animation.gif" width="600"></p>

*GemPy* was designed from the beginning to support stochastic geological modeling for uncertainty analysis (e.g. Monte Carlo simulations, Bayesian inference). This was achieved by writing *GemPy*'s core architecture
using the numerical computation library [Theano](http://deeplearning.net/software/theano/) to couple it with the probabilistic programming framework [PyMC3](https://pymc-devs.github.io/pymc3/notebooks/getting_started.html).
This enables the use of advanced sampling methods (e.g. Hamiltonian Monte Carlo) and is of particular relevance when considering
uncertainties in the model input data and making use of additional secondary information in a Bayesian inference framework.

We can, for example, include uncertainties with respect to the z-position of layer boundaries
in the model space. Simple Monte Carlo simulation via PyMC will then result in different model realizations:

<p align="center"><img src="docs/source/images/gempy_zunc.png" height="300"> <img src="docs/source/images/model_wobble.gif" height="300"></p>

Theano allows the automated computation of gradients opening the door to the use of advanced gradient-based sampling methods
coupling *GeMpy* and [PyMC3](https://pymc-devs.github.io/pymc3/notebooks/getting_started.html) for advanced stochastic modeling.
Also, the use of Theano allows making use of GPUs through cuda (see the Theano documentation for more information.

For a more detailed elaboration of the theory behind *GemPy*, take a look at the upcoming scientific publication
*"GemPy 1.0: open-source stochastic geological modeling and inversion"* by de la Varga et al. (2018).

## Getting Started

### Dependecies

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

* `vtk>=7` for interactive 3-D visualization 
* `pymc` or `pymc3`
* `steno3d` 

Overall we recommend the use of a dedicated Python distribution, such as 
[Anaconda](https://www.continuum.io/what-is-anaconda), for hassle-free package installation. 
We are curently working on providing GemPy also via Anaconda Cloud, for easier installation of 
its dependencies.

### Installation

We provide the latest release version of *GemPy* via the **Conda** and **PyPi** package services. We highly
recommend using either Conda or PyPi as both will take care of automatically installing all dependencies.

#### PyPi 

`$ pip install gempy`

#### Manual

Otherwise you can clone the current repository by downloading is manually or by using Git by calling

`$ git clone https://github.com/cgre-aachen/gempy.git`

and then manually install it using the provided Python install file by calling

`$ python gempy/setup.py install`

in the cloned or downloaded repository folder. Make sure you have installed all necessary dependencies listed above before using *GemPy*.

## Documentation

Extensive documentation for *GemPy* is hosted at [gempy.readthedocs.io](http://gempy.readthedocs.io/),
explaining its capabilities, [the theory behind it](http://gempy.readthedocs.io/Kriging.html) and 
 providing detailed [tutorials](http://gempy.readthedocs.io/tutorial.html) on how to use it.

## References

* Calcagno, P., Chilès, J. P., Courrioux, G., & Guillen, A. (2008). Geological modelling from field data and geological knowledge: Part I. Modelling method coupling 3D potential-field interpolation and geological rules. Physics of the Earth and Planetary Interiors, 171(1-4), 147-157.
* Lajaunie, C., Courrioux, G., & Manuel, L. (1997). Foliation fields and 3D cartography in geology: principles of a method based on potential interpolation. Mathematical Geology, 29(4), 571-584.
