# GemPy

[![PyPI](https://img.shields.io/badge/pypi-v0.9-green.svg)]() 
[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)]() 


GemPy is an open-source, Python-based 3-D structural geological modeling software, 
which allows the implicit (i.e. automatic) creation of complex geological models from interface 
and foliation data. It also sports support for stochastic modeling to adress parameter and model
uncertainties.

## Table of Contents

* [Examples](##Examples)
* [Getting Started](##GettingStarted)
    * [Prerequisits](###Prerequisits)
    * [Installation](###Installation)
    
## Examples

Put one or two fancy examples here.
 
![alt text]("/docs/readme_images/readme_input_data.png")
![alt text](/docs/readme_images/readme_fault_model_block.png)

## Getting Started

### Prerequisits

GemPy requires `Python 3.X` and makes use of numerous open-source libraries:

* `numpy`
* `pandas`
* `matplotlib`
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