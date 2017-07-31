# GemPy

[![PyPI](https://img.shields.io/badge/pypi-v0.9-yellow.svg)]()
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)]()

GemPy is an open-source, Python-based 3-D structural geological modeling software, 
which allows the implicit (i.e. automatic) creation of complex geological models from interface 
and foliation data. It also sports support for stochastic modeling to adress parameter and model
uncertainties.

## Table of Contents

* [Examples](##Examples)
* [Getting Started](##GettingStarted)
    * [Prerequisits](###Prerequisits)
    * [Installation](###Installation)
* [Documentation](##Documentation)

## Examples

GemPy uses interface (i.e. layer interface points, fault plane points) and foliation data 
(i.e. dip values of the surfaces) as input. The following plot shows exemplatory input data of 
four different lithology interfaces and a fault interface (blue), with only two dip 
measurements (arrows).
 
![alt text](/docs/readme_images/readme_input_data.png)

From this input data GemPy can implicitly construct a full 3-D structural geological model:

![alt text](/docs/readme_images/readme_fault_model_block.png)

## Getting Started

### Prerequisits

GemPy requires `Python 3.X` and makes use of numerous open-source libraries:

* `numpy`
* `pandas`
* `matplotlib`
* `seaborn`
* `theano`

Optional:

* `vtk` for interactive 3-D visualization (VTK v. 7.X is required for Python 3)
* `skimage` and `networkx` for 3-D topology analysis and graph handling

Overall we recommend the use of a dedicated Python distribution, such as 
[Anaconda](https://www.continuum.io/what-is-anaconda), for hassle-free package installation. 
We are curently working on providing GemPy also via Anaconda Cloud, for easier installation of 
its dependencies.

### Installation

Installing the latest release version of GemPy itself is easily done using PyPI:

`$ pip install gempy`

Otherwise you can clone the current repository:

`$ git clone https://github.com/cgre-aachen/gempy.git`

And manually install it using the following command in the repository directory:

`$ python install.py`

## Documentation

Extensive documentation for GemPy is hosted at [gempy.readthedocs.io](http://gempy.readthedocs.io/),
explaining its capabilities, [the theory behind it](http://gempy.readthedocs.io/Kriging.html) and 
 providing detailed [tutorials](http://gempy.readthedocs.io/tutorial.html) on how to use it.