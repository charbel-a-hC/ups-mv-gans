#!bin/bash

docker run -it --rm --gpus=all -v ${PWD}:/ups-mv-gans -p 8888:8888 ups-gans
