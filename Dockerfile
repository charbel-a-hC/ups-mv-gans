FROM tensorflow/tensorflow:2.5.0-gpu
FROM conda/miniconda3:latest

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y software-properties-common

# Install System Dependencies
RUN apt update && \
    apt install -y \
    git \
    vim \
    curl \ 
    make

ENV PATH=/root/.local/bin:$PATH

WORKDIR /ups-mv-gans-project
COPY Makefile .
COPY environment.yml .
#RUN make env-docker