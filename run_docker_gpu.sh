#!/bin/bash

docker run -it --rm --runtime=nvidia --gpus all -v ${PWD}:/ups-mv-gans-project -p 8000:8000 ups-gans bash

