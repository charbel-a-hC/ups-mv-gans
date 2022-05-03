FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
#FROM tensorflow/tensorflow:2.2.2-gpu-jupyter

ARG DEBIAN_FRONTEND=noninteractive

# To save you a headache
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs

ENV PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# Fix Nvidia/Cuda repository key rotation
RUN sed -i '/developer\.download\.nvidia\.com\/compute\/cuda\/repos/d' /etc/apt/sources.list.d/*
RUN sed -i '/developer\.download\.nvidia\.com\/compute\/machine-learning\/repos/d' /etc/apt/sources.list.d/*  
RUN apt-key del 7fa2af80 &&\
	apt-get update && \
	apt-get  install -y wget && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb 

# Install needed libraries
#RUN apt-get update && \
#	apt-get upgrade -y && \
#	apt-get install -y clang-format wget apt-utils build-essential\
#	checkinstall cmake pkg-config yasm git vim curl\
#	autoconf automake libtool libopencv-dev build-essential

# Install python-dev and pip
# Install python dependencies
RUN apt-get update && apt-get install -y \
  locales \
  python3.6 \
  python3-pip \
  wget \
  pkg-config\
  curl

RUN python3 -m pip install --upgrade pip
#ENV PATH=/root/.local/bin:$PATH

RUN --mount=type=cache,target=/root/.cache/pip pip3 install wandb tensorflow-gpu==2.2 scikit-image matplotlib jupyter

WORKDIR /ups-mv-gans-project
COPY . /ups-mv-gans-project
