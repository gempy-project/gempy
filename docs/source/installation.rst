.. role:: raw-html-m2r(raw)
   :format: html


Installation
------------

:raw-html-m2r:`<a name="installation"></a>`

Installing ``GemPy``
^^^^^^^^^^^^^^^^^^^^^^^^

We provide the latest release version of ``GemPy`` via the **Conda** and **PyPi** package services. We highly
recommend using PyPi, as it will take care of automatically installing all dependencies.

PyPi
~~~~

``$ pip install gempy``

You can also visit `PyPi <https://pypi.org/project/gempy/>`_, or
`GitHub <https://github.com/cgre-aachen/gempy>`_

For more details in the installation check:
`Installation <http://docs.pyvista.org/getting-started/installation.html#install-ref.>`_

:raw-html-m2r:`<a name="depend"></a>`

Dependencies
^^^^^^^^^^^^

``GemPy`` requires Python 3.x and makes use of numerous open-source libraries:

* pandas
* Theano>=1.0.4
* matplotlib
* numpy
* pytest
* seaborn>=0.9
* networkx
* scikit-image>=0.17
* pyvista

Optional requirements:

*gdal
*qgrid==1.3.0
*pymc3
*pyevtk
*pyqrcode
*mplstereonet

Overall we recommend the use of a **dedicated Python distribution**\ , such as
`Anaconda <https://www.continuum.io/what-is-anaconda>`_\ , for hassle-free package installation. 
We are currently working on providing ``GemPy`` also via Anaconda Cloud, for easier installation of
its dependencies.

Conflictive packages.
~~~~~~~~~~~~~~~~~~~~~

Installing Theano (especially under Windows) and vtk can sometimes be difficult.
Here, we provide adivce that should use in most cases (but certainly not all):


.. raw:: html

   <ul>
   <li> ``Theano``: install the following packages before installing theano:

   `conda install mingw libpython m2w64-toolchain`

   Then install Theano via

   `conda install theano`

   If the installation fails at some point try to re-install anaconda for a single user (no administrator priveleges) and with the Path Environment set.
   To use Theano with `numpy version 1.16.0` or following, it has to be updated to `Theano 1.0.4` using

   `pip install theano --upgrade`

   Note that this is not yet available in the conda package manager.
   </li>


   <li>

   ``vtk``: There have been compatibility problems between with the `vtk` package
   and python 3.8. The simplest solution to install it is to
   use `conda install python==3.6` to downgrade the python version and then
   using `pip install vtk`.

   </li>

   </ul>


Docker
^^^^^^

Finally we also provide precompiled Docker images hosted on Docker Hub with all necessary dependencies to get
GemPy up and running (\ **except vtk**\ ).

Docker is an operating-system-level-visualization software,
meaning that we can package a tiny operating system with pre-installed
software into a Docker image. This Docker image can then be shared
with and run by others, enabling them to use intricate dependencies
with just a few commands. For this to work the user needs to have a
working `Docker <https://www.docker.com/>`_ installation.

:raw-html-m2r:`<a name="docker"></a>`

Pull Docker image from DockerHub
""""""""""""""""""""""""""""""""

The easiest way to get remote-geomod running is by running the pre-compiled Docker image (containing everything you
need) directly from the cloud service Docker Hub to get a locally running Docker container. Make sure to set your 
Docker daemon to Linux containers in Docker's context menu.

.. code-block::

   $ docker run -it -p 8899:8899 leguark/gempy


This will automatically pull the Docker image from Docker Hub and run it, opening a command line shell inside of the
running Docker container. There you have access to the file system inside of the container. Note that this pre-compiled
Docker image already contains the GemPy repository. 

Once you are in the docker console if you want to open the tutorials you will need to run:

.. code-block::

   $ jupyter notebook --ip 0.0.0.0 --port 8899 --no-browser --allow-root


Notice that we are running the notebook on the port  8899 to try to avoid conflicts with jupyter servers running in
your system. If everything worked fine, the address to the jupyter notebook will be display on the console. It
has to look something like this (Just be aware of the  brackets):

.. code-block::

   To access the notebook, open this file in a browser:
           file:///root/.local/share/jupyter/runtime/nbserver-286-open.html
   Or copy and paste one of these URLs:
       http://(ce2cdcc55bb0 or 127.0.0.1):8899/?token=97d52c1dc321c42083d8c1b4d


:raw-html-m2r:`<a name="cutting-edge"></a>`

Manual installation of cutting-edge version
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can clone the current repository by downloading is manually or by using Git by calling

``$ git clone https://github.com/cgre-aachen/gempy.git``

and then manually install it using the provided Python install file by calling

``$ python gempy/setup.py install``

in the cloned or downloaded repository folder. Make sure you have installed all
necessary dependencies listed above before using ``GemPy``.

:raw-html-m2r:`<a name="windows"></a>`

Windows installation guide (March 2020)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1) This step is **only important if you want GPU acceleration**. Install CUDA if you do not have it already.

.. code-block::

   - For CUDA > 10 (For RTX cards you need those drivers):
       - Go to your cuda installation (probably
        `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\bin` )
       - Duplicate cublas64_XX and nvrtc64_XX and rename them to cublas64_70 and nvrtc64_70


2) Install Conda (recommended: latest miniconda)

.. code-block::

   - Install in you user
   - Add conda to the main path
   - Add conda enviroment:
       - `conda create --name gempy`
       - `conda init powershell`
       -  As admin `Set-ExecutionPolicy RemoteSigned`
   - After this stage we should have a new empty environment attached to a user


3) Install Theano and associated packages from the Anaconda prompt as administrator, and finally install GemPy 2.0:

.. code-block::

   - `conda update --all`
   - `conda install libpython`
   - `conda install m2w64-toolchain`
   - `conda install git`
   - `conda install -c conda-forge pygpu`
   - `conda install python==3.7` **Downgrade python back to 3.7 until vtk has
   support for python 3.8**
   - `pip install theano==1.0.4`
   - `pip install gempy`


4) Set up Jupyter to work properly with conda environments:

.. code-block::

   - `conda install Jupyter`
   - `conda install nb_conda_kernels`
   - `pip install jupyter-conda`


5) Optional requirements:

.. code-block::

   - `pip install pyvista`
   - `pip install pyevtk`
   - `conda install gdal`


**Note**\ :


* some other packages required by Theano are already included in Anaconda: numpy, scipy, mkl-service, nose, and sphinx.
* ``pydot-ng`` (suggested on Theano web site) yields a lot of errors. I dropped this. It is needed to handle large picture for gif/images and probably it is not needed by GemPy.
* Trying to install all the packages in one go but it does not work, as well as doing the same in Anaconda Navigator, or installing an older Anaconda release with Python 3.5 (Anaconda3 4.2.0) as indicated in some tutorial on Theano.

:raw-html-m2r:`<a name="macosx"></a>`

MacOSX 10.14 installation guide (April 2020)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Note**\ : The following guide is for a standard installation (no GPU support).
It should also work on MacOSX 10.15, but this is not tested, yet.


.. raw:: html

   <ol>

   <li> Install Anaconda

   For a minimal installation, you can install the
   [Miniconda distribution](https://docs.conda.io/en/latest/miniconda.html}.
   The full Anaconda distribution contains some additional features, IDE's
   etc. and is available on the [Anaconda page](https://www.anaconda.com/products/individual).

   </li>

   <li> Create a <code>GemPy</code> conda environment

   We strongly suggtest to create a separate conda environment, to avoid
   conflicts with other Python installations and packages on your system.
   This is easily done in a bash terminal:

   <ul>

   <li>

   `conda create --name gempy`

   </li>
   <li> To activate this environment:

   `conda activate gempy`

   You should now see `(gempy)` at the beginning of the command line. If
   the previous command fails (some known issues), then try:

   `source activate gempy`

   </li>
   </ul>
   </li>

   <li>Install required Python packages

   - `conda update --all`
   - `conda install python==3.7` **Downgrade python back to 3.7 until vtk has support for python 3.8**
   - `pip install theano==1.0.4`

   - Test the `theano` installation: run `python`, then try `import theano`.
   If you get an error (e.g. `stdio.h` not found), then:
   <ul>
   <li> Test if the Xcode command-line tools are installed (info for
   example <a href="https://osxdaily.com/2014/02/12/install-command-line-tools-mac-os-x/">here</a>).
   <li> If this still fails, try installing `theano` through conda-forge instead:

   `conda install -c conda-forge theano`

   </ul>


   **Note**: Theano requires the Xcode command-line tools installed. Overall,
   getting Theano to run can be a bit daunting... we hope to find a better
   method in the future.



   </li>

   <li>Install <code>GemPy</code>:

   - `pip install gempy`

   </li>

   <li> Set up Jupyter to work properly with conda environments:

   - `conda install Jupyter`
   - `conda install nb_conda_kernels`
   - `pip install jupyter-conda`

   </li>

   <li> Optional requirements:

   - `pip install pyvista`
   - `pip install pyevtk`
   - `conda install gdal`

   </li>

   </ol>

