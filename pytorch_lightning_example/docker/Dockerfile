
ARG BASE_CONTAINER=nvidia/cuda:11.6.0-base-ubuntu20.04

FROM $BASE_CONTAINER

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update
RUN apt-get install -y gcc
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version

COPY env.yml /env.yml
RUN conda env create -f /env.yml

RUN mkdir /home/files
