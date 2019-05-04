FROM continuumio/miniconda3
RUN conda create -n env python=3.6
RUN git clone https://github.com/cgre-aachen/gempy.git
WORKDIR gempy
RUN pip install gempy==2.0b.dev1 -r requirements.txt
RUN git+git://github.com/Leguark/scikit-image@master
RUN pip install vtk
RUN conda install gdal qgrid
