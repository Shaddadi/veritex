# Dockerfile for veritex
#
# To build image:
# docker build -t veritex_image .
#
# To get a shell after building the image:
# docker run -it veritex_image bash

FROM python:3.7

# # set user
# USER root

ARG PYTHON_VERSION=3.7

# set working directory
WORKDIR /veritex

# install pip
RUN set -xe \
    && apt-get update -y \
    && apt-get install -y \
    python3-pip \
    python3-dev
RUN pip install --upgrade pip

# set environment variables
# ENV PYTHONPATH="/veritex:${PYTHONPATH}"
ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1

#COPY ../requirements.txt /tmp/

#RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
#    rm /tmp/requirements.txt
