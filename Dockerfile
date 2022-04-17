FROM tensorflow/tensorflow:2.5.0-gpu

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y software-properties-common

# Install System Dependencies
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    apt install -y \
    git \
    vim \
    curl

ENV PATH=/root/.local/bin:$PATH

WORKDIR /ups-mv-gans-project
COPY Makefile .
RUN make env-docker