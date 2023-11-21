.. role:: raw-html-m2r(raw)
   :format: html


Installation
------------


Installing ``GemPy``
^^^^^^^^^^^^^^^^^^^^^^^^

We provide the latest release version of ``GemPy`` via **PyPi** package services. We highly
recommend using `PyPi <https://pypi.org/project/gempy/>`_,

``$ pip install gempy``

as it will take care of automatically installing all the required dependencies - except in
windows that required one extra step.


Manual installation
^^^^^^^^^^^^^^^^^^^
If you want to use the newest, cutting-edge version of ``GemPy`` you can clone the current repository by downloading it manually or by using Git by calling

``$ git clone https://github.com/cgre-aachen/gempy.git``

and then manually install it using the provided Python install file by calling

``$ python gempy/setup.py install``

in the cloned or downloaded repository folder.

Alternatively to running ``setup.py``, you can use pip to handle the installation from the repository and the updating of the path variables.
For this, navigate to the root of the cloned repository and run

``$ pip install -e .``

Make sure you have installed all necessary dependencies listed below before using ``GemPy``.


Dependencies
^^^^^^^^^^^^

``GemPy`` requires Python 3.x and makes use of numerous open-source libraries:

* pandas
* aesara>=1.0.4
* matplotlib
* numpy
* pytest
* seaborn>=0.9
* networkx
* scikit-image>=0.17
* pyvista

Optional requirements:

* gdal
* qgrid==1.3.0
* pymc3
* pyevtk
* pyqrcode
* mplstereonet

Overall we recommend the use of a **dedicated Python distribution**\ , such as
`Anaconda <https://www.continuum.io/what-is-anaconda>`_\ , for hassle-free package installation. 
We are currently working on providing ``GemPy`` also via Anaconda Cloud, for easier installation of
its dependencies.
