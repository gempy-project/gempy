# <p align="center"><img src="docs/readme_images/header_combined_slim.png" width="1000"></p>

> Open-source, implicit 3D structural geological modeling in Python.

[![PyPI](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31013/)
[![PyPI](https://img.shields.io/badge/pypi-1.0-blue.svg)](https://pypi.org/project/gempy/)
[![license: EUPL v1.2](https://img.shields.io/badge/license-EUPL%20v1.2-blue.svg)](https://github.com/cgre-aachen/gempy/blob/master/LICENSE)
[![Documentation Status](https://assets.readthedocs.org/static/projects/badges/passing-flat.svg)](http://docs.gempy.org)
[![DOI](https://zenodo.org/badge/96211155.svg)](https://zenodo.org/badge/latestdoi/96211155)

## What's New: GemPy v3 Pre-release!

GemPy v3 is gearing up for its official launch. Our team is diligently working on refining the documentation and adding the final touches. Delve into the exciting new features in the [What's New in GemPy v3](WhatsNewGemPy3.md). Experience GemPy v3 firsthand by installing the pre-release version from PyPi:

`$ pip install gempy --pre`

In the meantime, you can explore the updated documentation [here](https://gempy-project.github.io/temp_gp3_docs/).

## Overview

[GemPy](https://www.gempy.org/) is a Python-based, **open-source geomodeling library**. It is
capable of constructing complex **3D geological models** of folded
structures, fault networks and unconformities, based on the underlying
powerful **implicit representation** approach. 

## Installation

We provide the latest release version of GemPy via PyPi package services. We highly recommend using PyPi,

`$ pip install gempy`


## Requirements

The following versions are required/strongly recommended for the main dependencies of GemPy (as of June 2023):
- python <= 3.10 (required by aesara)
- numpy
 
## Resources

After installation, you can either check the [notebook tutorials](https://docs.gempy.org/getting_started/get_started.html#sphx-glr-getting-started-get-started-py) 
or the [video introduction](https://www.youtube.com/watch?v=n0btC5Zilyc) to get started.

Go to the [documentation site](http://docs.gempy.org/) for further information and enjoy the [tutorials and examples](https://www.gempy.org/tutorials).

For questions and support, please use [discussions](https://github.com/cgre-aachen/gempy/discussions).

If you find a bug or have a feature request, create an [issue](https://github.com/cgre-aachen/gempy/issues).

Follow these [guidelines](https://github.com/cgre-aachen/gempy/blob/WIP_readme-update-march21/CONTRIBUTING.md) to contribute to GemPy.

<a name="ref"></a>
## References 

* de la Varga, M., Schaaf, A., and Wellmann, F. (2019). [GemPy 1.0: open-source stochastic geological modeling and inversion](https://gmd.copernicus.org/articles/12/1/2019/gmd-12-1-2019.pdf), Geosci. Model Dev., 12, 1-32.
* Wellmann, F., & Caumon, G. (2018). [3-D Structural geological models: Concepts, methods, and uncertainties.](https://hal.univ-lorraine.fr/hal-01921494/file/structural_models_for_geophysicsHAL.pdf) In Advances in Geophysics (Vol. 59, pp. 1-121). Elsevier.
* Calcagno, P., Chilès, J. P., Courrioux, G., & Guillen, A. (2008). [Geological modelling from field data and geological knowledge: Part I. Modelling method coupling 3D potential-field interpolation and geological rules](https://www.sciencedirect.com/science/article/abs/pii/S0031920108001258). Physics of the Earth and Planetary Interiors, 171(1-4), 147-157.
* Lajaunie, C., Courrioux, G., & Manuel, L. (1997). [Foliation fields and 3D cartography in geology: principles of a method based on potential interpolation](https://link.springer.com/article/10.1007/BF02775087). Mathematical Geology, 29(4), 571-584.

## Publications using GemPy



* Brisson, S., Wellmann, F., Chudalla, N., von Harten, J., & von Hagke, C. (2023). [Estimating uncertainties in 3-D models of complex fold-and-thrust belts: A case study of the Eastern Alps triangle zone](https://www.sciencedirect.com/science/article/pii/S2590197423000046). Applied Computing and Geosciences, 18, 100115.

* Liang, Z., de la Varga, M., & Wellmann, F. (2023). [Kernel method for gravity forward simulation in implicit probabilistic geologic modeling](https://pubs.geoscienceworld.org/geophysics/article/88/3/G43/621596/Kernel-method-for-gravity-forward-simulation-in?casa_token=VjCR7rYOkKoAAAAA:W81L1AXgW_j9GiYPciBvLIdL8Zo66IzYVYiU6Ri8xLgIjbzTmpcDE74rzmAwnokX_71_XKg). Geophysics, 88(3), G43-G55.

* Kong, S., Oh, J., Yoon, D., Ryu, D. W., & Kwon, H. S. (2023). [Integrating Deep Learning and Deterministic Inversion for Enhancing Fault Detection in Electrical Resistivity Surveys](https://www.mdpi.com/2076-3417/13/10/6250). Applied Sciences, 13(10), 6250.

* Thomas, A. T., Micallef, A., Duan, S., & Zou, Z. (2023). [Characteristics and controls of an offshore freshened groundwater system in the Shengsi region, East China Sea](https://www.frontiersin.org/articles/10.3389/feart.2023.1198215/full). Frontiers in Earth Science, 11, 1198215.

* Haehnel, P., Freund, H., Greskowiak, J. & Massmann, G. (2023) [Development of a three-dimensional hydrogeological model for the island of Norderney (Germany) using GemPy](https://doi.org/10.1002/gdj3.208). Geoscience Data Journal, 00, 1–17. 

* Jüstel, A., de la Varga, M., Chudalla, N., Wagner, J. D., Back, S., & Wellmann, F. (2023). [From Maps to Models-Tutorials for structural geological modeling using GemPy and GemGIS](https://jose.theoj.org/papers/10.21105/jose.00185). Journal of Open Source Education, 6(66), 185.

* Sehsah, H., Eldosouky, A. M., & Pham, L. T. (2022). [Incremental Emplacement of the Sierra Nevada Batholith Constrained by U-Pb Ages and Potential Field Data](https://www.journals.uchicago.edu/doi/full/10.1086/722724?casa_token=pkl8XXrtyokAAAAA:YeIh1t-qwt6AT8yz_vTj4OQapaR1_nZUjS3Az_77VZXlpyfGu0cN5DSzrz6NNjoj4Qv5iud4rdc). The Journal of Geology, 130(5), 381-391.

* Schaaf, A., de la Varga, M., Wellmann, F., & Bond, C. E. (2021). [Constraining stochastic 3-D structural geological models with topology information using approximate Bayesian computation in GemPy 2.1](https://gmd.copernicus.org/articles/14/3899/2021/gmd-14-3899-2021.html). Geosci. Model Dev., 14(6), 3899-3913. doi:10.5194/gmd-14-3899-2021
* Güdük, N., de la Varga, M. Kaukolinna, J. and Wellmann, F. (2021). [Model-Based Probabilistic Inversion Using Magnetic Data: A Case Study on the Kevitsa Deposit](https://www.mdpi.com/2076-3263/11/4/150), _Geosciences_, 11(4):150. https://doi.org/10.3390/geosciences11040150.

* Wu, J., & Sun, B. (2021). [Discontinuous mechanical analysis of manifold element strain of rock slope based on open source Gempy](https://www.e3s-conferences.org/articles/e3sconf/abs/2021/24/e3sconf_caes2021_03084/e3sconf_caes2021_03084.html). In E3S Web of Conferences (Vol. 248, p. 03084). EDP Sciences.

* Stamm, F. A., de la Varga, M., and Wellmann, F. (2019). [Actors, actions, and uncertainties: optimizing decision-making based on 3-D structural geological models](https://se.copernicus.org/articles/10/2015/2019/se-10-2015-2019.html), Solid Earth, 10, 2015–2043.
* Wellmann, F., Schaaf, A., de la Varga, M., & von Hagke, C. (2019). [From Google Earth to 3D Geology Problem 2: Seeing Below the Surface of the Digital Earth](
https://www.sciencedirect.com/science/article/pii/B9780128140482000156).
In Developments in Structural Geology and Tectonics (Vol. 5, pp. 189-204). Elsevier.

Please let us know if your publication is missing!

A continuously growing list of gempy-applications (e.g. listing real-world models) can be found [here](https://hackmd.io/@Japhiolite/B1juPvCxc).

## Gallery

### Geometries

<p>
<table>
<tr>

  <td>
  <a href="https://docs.gempy.org/examples/geometries/1_horizontal_stratigraphic.html#sphx-glr-examples-geometries-1-horizontal-stratigraphic-py">
  <img alt="colormapped image plot thumbnail" src="docs/readme_images/model1_nodata.png" width="300" />
  </a>
  </td>
  
  <td>
  <a href="https://docs.gempy.org/examples/geometries/2_fold.html#sphx-glr-examples-geometries-2-fold-py">
  <img alt="colormapped image plot thumbnail" src="docs/readme_images/model2_nodata.png" width="300" />
  </a>
  </td>
  
   <td>
  <a href="https://docs.gempy.org/examples/geometries/3_recumbent_fold.html#sphx-glr-examples-geometries-3-recumbent-fold-py">
  <img alt="colormapped image plot thumbnail" src="docs/readme_images/model3_nodata.png" width="300" />
  </a>
  </td>

</tr>
<tr>

  <td>
  <a href="https://docs.gempy.org/examples/geometries/4_pinchout.html#sphx-glr-examples-geometries-4-pinchout-py">
  <img alt="colormapped image plot thumbnail" src="docs/readme_images/model4_nodata.png" width="300" />
  </a>
  </td>
  
  <td>
  <a href="https://docs.gempy.org/examples/geometries/5_fault.html#sphx-glr-examples-geometries-5-fault-py">
  <img alt="colormapped image plot thumbnail" src="docs/readme_images/model5_nodata.png" width="300" />
  </a>
  </td>
  
  <td>
  <a href="https://docs.gempy.org/examples/geometries/6_unconformity.html#sphx-glr-examples-geometries-6-unconformity-py">
  <img alt="colormapped image plot thumbnail" src="docs/readme_images/model6_nodata.png" width="300" />
  </a>
  </td>

</tr>
</table>
</p>

### Features

<p>
<table>
<tr>

  <td>
  <a href="https://docs.gempy.org/tutorials/ch1_fundamentals/ch1_3b_cross_sections.html#sphx-glr-tutorials-ch1-fundamentals-ch1-3b-cross-sections-py">
  <img alt="colormapped image plot thumbnail" src="docs/readme_images/sectiontest.png" width="300" />
  </a>
  </td>
  
  <td>
  <a href="https://docs.gempy.org/tutorials/ch1_fundamentals/ch1_7_3d_visualization.html#sphx-glr-tutorials-ch1-fundamentals-ch1-7-3d-visualization-py">
  <img alt="colormapped image plot thumbnail" src="docs/readme_images/data_vis.png" width="300" />
  </a>
  </td>
  
   <td>
  <a href="https://docs.gempy.org/examples/geometries/7_combination.html#sphx-glr-examples-geometries-7-combination-py">
  <img alt="colormapped image plot thumbnail" src="docs/readme_images/scalarfield.png" width="300" />
  </a>
  </td>

</tr>
<tr>

  <td>
  <a href="https://docs.gempy.org/tutorials/ch1_fundamentals/ch1_3b_cross_sections.html#sphx-glr-tutorials-ch1-fundamentals-ch1-3b-cross-sections-py">
  <img alt="colormapped image plot thumbnail" src="docs/readme_images/geomap.png" width="300" />
  </a>
  </td>
  
  <td>
  <a href="https://docs.gempy.org/tutorials/ch4-Topology/ch4-1-Topology.html#sphx-glr-tutorials-ch4-topology-ch4-1-topology-py">
  <img alt="colormapped image plot thumbnail" src="docs/readme_images/topology.png" width="300" />
  </a>
  </td>
  
  <td>
  <a href="https://docs.gempy.org/tutorials/ch4-Topology/ch4-1-Topology.html#sphx-glr-tutorials-ch4-topology-ch4-1-topology-py">
  <img alt="colormapped image plot thumbnail" src="docs/readme_images/topology_matrix.png" width="300" />
  </a>
  </td>

</tr>
</table>
</p>


### Case studies

<p>
<table>
<tr>

  <td>
  <a href="https://docs.gempy.org/examples/real/Alesmodel.html#sphx-glr-examples-real-alesmodel-py">
  <img alt="colormapped image plot thumbnail" src="docs/readme_images/alesmodel.png" width="300" />
  </a>
  </td>
  
  <td>
  <a href="https://docs.gempy.org/examples/real/Perth_basin.html#sphx-glr-examples-real-perth-basin-py">
  <img alt="colormapped image plot thumbnail" src="docs/readme_images/perthmodel.png" width="300" />
  </a>
  </td>
  
   <td>
  <a href="https://docs.gempy.org/examples/real/Greenstone.html#sphx-glr-examples-real-greenstone-py">
  <img alt="colormapped image plot thumbnail" src="docs/readme_images/greenstonemodel.png" width="300" />
  </a>
  </td>

</tr>
</table>
</p>
