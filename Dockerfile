FROM continuumio/miniconda3
RUN conda create -n env python=3.6

# ...
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get -y install gcc mono-mcs && \
    rm -rf /var/lib/apt/lists/* \
    apt-get install git
# RUN apt-get install libc-dev
# RUN conda install gcc
RUN git clone https://github.com/cgre-aachen/gempy.git
WORKDIR gempy
RUN conda install theano gdal qgrid
RUN pip install --upgrade --force-reinstall Theano>=1.0.4
RUN pip install gempy==2.0b.dev2 pandas>=0.21.0 cython pytest seaborn networkx ipywidgets scikit-image
