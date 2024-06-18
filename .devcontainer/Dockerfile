# Dockerfile for veritex
#
# To build image:
# docker build -t veritex_image .
#
# To get a shell after building the image:
# docker run -it veritex_image bash

FROM python:3.7

ARG PYTHON_VERSION=3.7

# Set working directory
WORKDIR /veritex

# Install pip
RUN set -xe \
    && apt-get update -y \
    && apt-get install -y \
    python3-pip \
    python3-dev
RUN pip install --upgrade pip

# Set environment variables
ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1

# Copy pyproject.toml and other project files
COPY pyproject.toml /veritex/
COPY . /veritex/

# Install the package
RUN python3.7 -m pip install -e .

# Optionally, you can uncomment these lines if you have a requirements.txt
# COPY requirements.txt /veritex/
# RUN pip install --no-cache-dir -r requirements.txt