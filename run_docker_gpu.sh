#!/bin/bash

docker run -it --rm --runtime=nvidia --gpus all -v ${PWD}:/ups-mv-gans -p $1:$1 ups-gans bash

