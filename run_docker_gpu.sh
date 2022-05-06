#!/bin/bash

docker run -it --rm --runtime=nvidia --gpus all -v ${PWD}:/ups-mv-gans-project -p 8888:8888 ups-gans bash

