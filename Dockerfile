# Dockerfile for veritex
#
# To build image:
# docker build -t veritex_image .
#
# To get a shell after building the image:
# docker run -it veritex_image bash

FROM python:3.7
# FROM mathworks/matlab:r2020b
FROM mathworks/matlab-deps:r2020b

# set working directory
WORKDIR /veritex

COPY requirements.txt requirements.txt

# install pip
RUN set -xe \
    && apt-get update -y \
    && apt-get install -y python3-pip
RUN pip install --upgrade pip

# install python package dependencies
RUN pip install --no-cache-dir -r requirements.txt

# set environment variables
ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1

# copy files to docker
COPY . .
