# Dockerfile for veritex
#
# To build image:
# docker build -t veritex_image .
#
# To get a shell after building the image:
# docker run -it veritex_image bash

FROM python:3.7

# set user
USER root

# set working directory
WORKDIR /veritex

# install pip
RUN set -xe \
    && apt-get update -y \
    && apt-get install -y python3-pip
RUN pip install --upgrade pip

# set environment variables
ENV PYTHONPATH=$PYTHONPATH:/veritex/veritex
ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1

# copy files to docker
COPY . .

# install python package dependencies
RUN pip install .

# # set user
# USER 1001
