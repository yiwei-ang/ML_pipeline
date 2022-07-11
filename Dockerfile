# Define base image
FROM continuumio/miniconda3

# Set working directory for the project
ENV MAIN_DIR=/usr/src/app/ml_pipeline
ENV PYTHONPATH=${MAIN_DIR}:$PYTHONPATH
COPY . ${MAIN_DIR}

# Create Conda environment from the YAML file
ADD environment.yml ${MAIN_DIR}
RUN /opt/conda/bin/conda env create -f ${MAIN_DIR}/environment.yml \
    && conda clean -tipsy \
    && find /opt/conda -type f,l -name '*.a' -delete \
    && find /opt/conda -type f,l -name '*.pyc' -delete \
    && find /opt/conda -type f,l -name '*.js.map' -delete \
    && rm -rf /opt/conda/pkgs
SHELL ["conda", "run", "-n", "ml_pipeline", "/bin/bash", "-c"]
WORKDIR ${MAIN_DIR}

# exposing default port for streamlit
EXPOSE 8501

ENV PATH /opt/conda/envs/ml_pipeline/bin:$PATH
ENV CONDA_DEFAULT_ENV ml_pipeline

# Make sure the environment is working
RUN echo "Make sure the environment is working."
RUN python -c "import streamlit"

# Run the application after choosing the right conda environment
CMD ["streamlit", "run", "engine/frontend/main.py"]