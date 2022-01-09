# Dockerfile for veritex
#
# To build image:
# docker build -t veritex_image .
#
# To get a shell after building the image:
# docker run -ir veritex_image bash

FROM python:3.8
FROM mathworks/matlab:r2020b
WORKDIR /veritex

# install python package dependencies
RUN pip install --no-cache-dir -r requirements.txt

# set environment variables
ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1

# copy files to docker
COPY . .
