.. GeMpy documentation master file, created by
   sphinx-quickstart on Wed Dec 14 12:44:40 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GeMpy's documentation!
=================================
A 3D Structural geologic implicit modelling in python. (v 0.9)
**************************************************************

GeMpy is an opensource project for the generation of 3D structural geological modelling. The algorithm is based on a especial type of Universal cokrigin interpolation created by Laujaunie et al (1997) and developed along the past year by (*add many more papers!!!*). This tool allows the generation of models with relative complex structures just from data. The repository can be found here, https://github.com/nre-aachen/GeMpy.

The results look like this:

.. raw:: html

    <div style="margin-top:10px;">
      <iframe src="https://steno3d.com/embed/A747sS50WZZu75yEm8Yi" width="600" height="400"></iframe>
    </div>

Its implicit nature allows the user to generate complete 3D geological models through the interpolation of:

- Interfaces points: 3D points in space that delimit the different formation in a given setting.
- Foliations: Measurements of the poles (perpendicular to the layer) at any point in the 3D space.

.. image:: ./images/input_example.png

In addition to the interpolation GeMpy has been created with the idea of perform Bayesian Inferences in geological modeling (de la Varga and Wellmann, 2016). Due to this, the whole interpolation algorithm has been written in the optimization packge theano (http://deeplearning.net/software/theano/) what allows the computation of gradients opening the door to the use of advance HMC methods coupling GeMpy and PyMC3 (https://pymc-devs.github.io/pymc3/notebooks/getting_started.html). Also, the use of theano allows the use of the GPU through cuda (see theano doc for more information).

Contents:

.. toctree::
   :maxdepth: 3

   self
   Kriging
   tutorial
   code
   

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

