# Heavily intspired by docker theano
# Heavily Inspired from https://github.com/jupyter/docker-stacks/tree/master/minimal-notebook
FROM nvidia/cuda:9.0-cudnn7-devel

# This loads nvidia cuda image
# FROM nvidia/cuda

USER root

# Install all OS dependencies for fully functional notebook server
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -yq --no-install-recommends \
    git \
    vim \
    jed \
    emacs \
    wget \
    build-essential \
    python-dev \
    ca-certificates \
    bzip2 \
    unzip \
    libsm6 \
    pandoc \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-fonts-extra \
    texlive-fonts-recommended \
    texlive-generic-recommended \
    sudo \
    locales \
    libxrender1 \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen

# Install Tini
RUN wget --quiet https://github.com/krallin/tini/releases/download/v0.9.0/tini && \
    echo "faafbfb5b079303691a939a747d7f60591f2143164093727e870b289a44d9872 *tini" | sha256sum -c - && \
    mv tini /usr/local/bin/tini && \
    chmod +x /usr/local/bin/tini

# Configure environment
ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH
ENV SHELL /bin/bash
ENV NB_USER gempy
ENV NB_UID 1000
ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8

# Create user with UID=1000 and in the 'users' group
RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
    mkdir -p /opt/conda && \
    chown gempy /opt/conda

USER gempy

# Setup home directory
RUN mkdir /home/$NB_USER/work && \
    mkdir /home/$NB_USER/.jupyter && \
    mkdir /home/$NB_USER/.local && \
    echo "cacert=/etc/ssl/certs/ca-certificates.crt" > /home/$NB_USER/.curlrc

# Install conda as
ENV CONDA_VER latest
ENV CONDA_MD5 7fe70b214bee1143e3e3f0467b71453c
RUN cd /tmp && \
    mkdir -p $CONDA_DIR && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-${CONDA_VER}-Linux-x86_64.sh && \
    /bin/bash Miniconda3-${CONDA_VER}-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-${CONDA_VER}-Linux-x86_64.sh && \
    conda config --set auto_update_conda False && \
    conda clean -tipsy

# Install Jupyter notebook
RUN conda install --quiet --yes \
    terminado \
    mkl-service \
    && conda clean -tipsy

# Install Theano, pygpu
RUN conda install -c conda-forge pygpu
RUN conda install theano gdal

ENV MKL_THREADING_LAYER GNU

USER root

# Configure container startup as root
EXPOSE 8888

# Clone gempy
WORKDIR /home/$NB_USER/work
RUN git clone https://github.com/cgre-aachen/gempy.git --depth 1 --branch master

WORKDIR /home/$NB_USER/work/gempy

# Pull from release
RUN echo '2014122501' >/dev/null && git pull
# This is necessary to get rid off the scan.c file missing

RUN pip install --upgrade --force-reinstall Theano>=1.0.4
RUN pip install -e .
RUN pip install -r optional-requirements.txt

# Pyvista headless display
RUN sudo apt-get -y update && \
    sudo apt-get -y install xvfb & \
    export DISPLAY=:99.0 && \
    export PYVISTA_OFF_SCREEN=true && \
    export PYVISTA_USE_PANEL=true && \
    Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 & \
    sleep 3

RUN sudo apt update && sudo apt -y install python-qt4 libgl1-mesa-glx
# ENTRYPOINT ["tini", "--"]

## Add local files as late as possible to avoid cache busting
## Start notebook server
#COPY start-notebook.sh /usr/local/bin/
#RUN chmod 755 /usr/local/bin/start-notebook.sh
#COPY jupyter_notebook_config_secure.py /home/$NB_USER/.jupyter/jupyter_notebook_config.py
#COPY notebook /home/$NB_USER/work/notebook

## My own change
#
#RUN apt-get update && apt-get install -y \
#        g++ \
#    && \
#    apt-get clean && \
#    rm -rf /var/lib/apt/lists/*
#
#COPY theanorc /home/$NB_USER/.theanorc
#
## Make sure user jovyan owns files in HOME
#RUN chown -R $NB_USER:users /home/$NB_USER
#
## Switch back to jovyan to avoid accidental container runs as root
#USER jovyan
#
#RUN mkdir data && cd data && wget http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist_py3k.pkl.gz -O mnist.pkl.gz
#
#CMD ["start-notebook.sh", "notebook"]
