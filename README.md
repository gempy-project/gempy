# GemPy

[![PyPI](https://img.shields.io/badge/pypi-v0.9-green.svg)]() 
[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)]() 


GemPy is an open-source, Python-based 3-D structural geological modeling software, 
which allows the implicit (i.e. automatic) creation of complex geological models from interface 
and foliation data. It also sports support for stochastic modeling to adress parameter and model
uncertainties.

## Table of Contents

* [Examples](##Examples)
* [Getting Started](##Getting Started)
    * [Prerequisits](###Prerequisits)
    * [Installation](###Installation)
    
## Examples

Put one or two fancy examples here.

## Getting Started

### Prerequisits

GemPy requires `Python 3.X` and makes use of numerous open-source libraries:

* `numpy`
* `pandas`
* `matplotlib`
* `theano`
* `vtk` for interactive 3-D visualization
* `skimage` and `networkx` for 3-D topology analysis and graph handling

### Installation

Installing the latest release version of GemPy itself is easily done using PyPI:

`$ pip install gempy`

Otherwise you can clone the current repository:

`$ git clone https://github.com/cgre-aachen/gempy.git`

And manually install it using the following command in the repository directory:

`$ python install.py`