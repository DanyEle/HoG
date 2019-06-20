#!/bin/bash

INPUT_IMG=$1

if [ "$1" == "" ]; then
    echo "Positional parameter 1 is empty. Please, provide an input image as parameter 1."
    exit
fi

for i in {1..10}
do
 # echo "------------------"
 # echo "Experiment $i"
  ./Debug/CUDA_Sobel $INPUT_IMG
done
