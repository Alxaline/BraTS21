FROM ubuntu:20.04

# Install Python and its tools
RUN apt update && apt install -y --no-install-recommends \
    git \
    tree \
    curl \
    build-essential \
    python3-dev \
    python3-pip \
    python3-setuptools
# Set the working directory in container
WORKDIR /BraTS21

# Install miniconda to /miniconda
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda

#RUN git clone https://github.com/Alxaline/recurrence_gbm.git
RUN conda create -n BraTS21 python=3.7.7

# Initialize conda in bash config fiiles:
RUN conda init bash

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "BraTS21", "/bin/bash", "-c"]

RUN conda env list
RUN pip install -U setuptools
COPY requirements.txt .
RUN pip install -r requirements.txt

# Activate the environment, and make sure it's activated:
RUN echo "Make sure pytorch is installed:"
RUN pip list
RUN python -c "import torch"

# CUDA
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# copy the dependencies file to the o

COPY learning learning
COPY networks networks
COPY src src
COPY tta tta
COPY utils utils

# final weight to copy
COPY final_weights final_weights

RUN chmod a+x .

RUN pwd
ENV PYTHONPATH=/BraTS21/:$PYTHONPATH



RUN tree

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "BraTS21", "python", "-m", "src.main_inference", "--config", "/BraTS21/final_weights/baseline_equiunet_assp_evocor/fold0_ns/config.yaml", "/BraTS21/final_weights/baseline_equiunet_assp_evocor/fold1_ns/config.yaml", "/BraTS21/final_weights/baseline_equiunet_assp_evocor/fold2_ns/config.yaml", "/BraTS21/final_weights/baseline_equiunet_assp_evocor/fold3_ns/config.yaml", "/BraTS21/final_weights/baseline_equiunet_assp_evocor/fold4_ns/config.yaml", "/BraTS21/final_weights/baseline_equiunet_assp_evocor_jaccard/fold0/config.yaml", "/BraTS21/final_weights/baseline_equiunet_assp_evocor_jaccard/fold1/config.yaml", "/BraTS21/final_weights/baseline_equiunet_assp_evocor_jaccard/fold2/config.yaml", "/BraTS21/final_weights/baseline_equiunet_assp_evocor_jaccard/fold3/config.yaml", "/BraTS21/final_weights/baseline_equiunet_assp_evocor_jaccard/fold4/config.yaml", "--on", "test", "-vv", "--replace_value", "--cleaning_areas", "--replace_value_threshold", "300", "--cleaning_areas_threshold", "20", "--device", "0", "--tta", "--docker_test", "--input" , "/input", "--output", "/output", "--num_workers", "0"]


