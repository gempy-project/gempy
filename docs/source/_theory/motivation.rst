Introduction
============


We commonly capture our knowledge about relevant geological features in
the subsurface in the form of geological models, as 3-D representations
of the geometric structural setting. Computer-aided geo logical modeling
methods have existed for decades, and many advanced and elaborate
commercial packages exist to generate these models (e.g. GoCAD, Petrel,
GeoModeller). But even though these packages partly enable an external
access to the modeling functionality through implemented API’s or
scripting interfaces, it is a significant disadvantage that the source
code is not accessible, and therefore the true inner workings are not
clear. More importantly still, the possibility to extend these methods
is limited—and, especially in the current rapid development of highly
efficient open-source libraries for machine learning and computational
inference (e.g. *TensorFlow*, *Stan*, *pymc*, *PyTorch*, *Infer.NET*),
the integration into other computational frameworks is limited.

Yet, there is to date no fully flexible open-source project which
integrates state-of-the-art geological modeling methods. Conventional
3-D construction tools (CAD, e.g. \ *pythonOCC*, *PyGem*) are only
useful to a limited extent, as they do not consider the specific aspects
of subsurface structures and the inherent sparcity of data. Open source
GIS tools exist (e.g. QGIS, *gdal*), but they are typically limited to
2-D (or 2.5-D) structures and do not facilitate the modeling and
representation of fault networks, complex structures like overturned
folds or dome structures), or combined stratigraphic sequences.

.. figure:: ./figs/Models.png
   :alt:

   Example of models generated using *GemPy*. a) Synthetic model
   representing a reservoir trap, visualized in Paraview
   :cite:`a-stamm2017`; b) Geological model of the Perth
   basin (Australia) rendered using GemPy on the in-built Python in
   Blender (see appendix [blender] for more details), spheres and cones
   represent the input data.

With the aim to close this gap, we present here *GemPy*, an open-source
implementation of a modern and powerful implicit geological modeling
method based on a potential-field approach, found, in turn, on a
Universal CoKriging interpolation
:cite:`a-Lajaunie.1997,Calcagno.2008`. In distinction to
surface-based modeling approaches see :cite:`a-Caumon.2009` for a
good overview,  these approaches allow the direct
interpolation of multiple conformal sequences in a single scalar field,
and the consideration of discontinuities (e.g. metamorphic contacts,
unconformities) through the interaction of multiple sequences
:cite:`a-Lajaunie.1997,Mallet.2004,Calcagno.2008,Caumon:2010jp,Hillier:2014gf`.
Also, these methods allow the construction of complex fault networks and
enable, in addition, a direct global interpolation of all available
geological data in a single step. This last aspect is relevant, as it
facilitates the integration of these methods into diverse other
workflows. Most importantly, we show here how we can integrate the
method into novel and advanced machine learning and Bayesian inference
frameworks :cite:`a-Salvatier:2016ki` for stochastic
geomodeling and Bayesian inversion. Recent developments in this field
have seen a surge in new methods and frameworks, for example using
gradient-based Monte Carlo methods
:cite:`a-duane1987hybrid` or variational
inferences :cite:`a-kucukelbir2016automatic`, making use of
efficient implementations of automatic differentiation
:cite:`a-rall1981automatic` in novel machine learning
frameworks. For this reason, *GemPy* is built on top of *Theano*, which
provides not only the mentioned capacity to compute gradients
efficiently, but also provides optimized compiled code (for more details
see Section [theano]). In addition, we utilize *pandas* for data storage
and manipulation :cite:`a-mckinney2011pandas`, Visualization
Toolkit (*vtk*) Python-bindings for interactive 3-D visualization
:cite:`a-schroeder2004visualization`, the de facto standard
2-D visualization library *Matplotlib*
:cite:`a-hunter2007matplotlib` and *NumPy* for efficient
numerical computations :cite:`a-walt2011numpy`. Our
implementation is specifically intended for combination with other
packages, to harvest efficient implementations in the best possible way.

Especially in this current time of rapid developments of open-source
scientific software packages and powerful machine learning frameworks,
we consider an open-source implementation of a geological modeling tool
as essential. We therefore aim to open up this possibility to a wide
community, by combining state-of-the-art implicit geological modeling
techniques with additional sophisticated Python packages for scientific
programming and data analysis in an open-source ecosystem. The aim is
explicitly not to rival the existing commercial packages with
well-designed graphical user interfaces, underlying databases, and
highly advanced workflows for specific tasks in subsurface engineering,
but to provide access to an advanced modeling algorithm for scientific
experiments in the field of geomodeling.

In the following, we will present the implementation of our code in the
form of core modules, related to the task of geological modeling itself,
and additional assets, which provide the link to external libraries, for
example to facilitate stochastic geomodeling and the inversion of
structural data. Each part is supported/ supplemented with Jupyter
Notebooks that are available as additional online material and part of
the package documentation, which enable the direct testing of our
methods (see Section  [sec:jupyter-notebooks]). These notebooks can also
be executed directly in an online environment (Binder). We encourage the
reader to use these tutorial Jupyter Notebooks to follow along the steps
explained in the following. We encourage the reader to use these
tutorial Jupyter Notebooks to follow along the steps explained in the
following. Finally, we discuss our approach, specifically also with
respect to alternative modeling approaches in the field, and provide an
outlook to our planned future work for this project.

.. bibliography:: small.bib
   :cited:
   :labelprefix: A
   :keyprefix: a-
   :style: unsrt
