FROM continuumio/miniconda3
RUN conda create -n env python=3.6
RUN conda install mingw
RUN git clone https://github.com/cgre-aachen/gempy.git
WORKDIR gempy
RUN conda install theano gdal qgrid
RUN pip install --upgrade --force-reinstall Theano>=1.0.4
RUN pip install gempy==2.0b.dev2 pandas>=0.21.0 cython pytest seaborn networkx ipywidgets scikit-image
