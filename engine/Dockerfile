ARG UBUNTU_VER=18.04
ARG CONDA_VER=latest
ARG OS_TYPE=x86_64
ARG MAIN_DIR=/usr/src/app/ml_pipeline

FROM ubuntu:${UBUNTU_VER}
# System packages
RUN apt-get update && apt-get install -yq curl wget jq vim

# Use the above args
ARG CONDA_VER
ARG OS_TYPE
# Install miniconda to /miniconda
RUN curl -LO "http://repo.continuum.io/miniconda/Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh"
RUN bash Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh -p /miniconda -b
RUN rm Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda
RUN conda init

# Install packages from conda

ADD environment.yml ${MAIN_DIR}
RUN conda env create -f environment.yml \
    && conda clean -tipsy \
    && find conda -type f,l -name '*.a' -delete \
    && find conda -type f,l -name '*.py' -delete \
    && find conda -type f,l -name '*.js.map' -delete
SHELL ["conda", "run", "-n", "ml_pipelione", "/bin/bash", "-c"]

WORKDIR ${MAIN_DIR}

ENTRYPOINT [ "streamlit", "run", "frontend/main.py" ]