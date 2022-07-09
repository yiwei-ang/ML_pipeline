# Define base image
FROM continuumio/miniconda3

# Set working directory for the project
WORKDIR /usr/src/app

# Create Conda environment from the YAML file
COPY environment.yml .
COPY requirements.txt .
RUN conda env create -f environment.yml

# exposing default port for streamlit
EXPOSE 8501

# Install dependencies
RUN pip install -r requirements.txt

# Override default shell and use bash
SHELL ["conda", "run", "-n", "ml_pipeline", "/bin/bash", "-c"]

# Activate Conda environment and check if it is working properly
ENTRYPOINT [ "streamlit", "run", "/usr/src/app/ML_pipeline/engine/frontend/main.py" ]