#!/bin/bash

INPUT_IMG=$1
THREADS_PER_PIXEL=$2

if [ "$1" == "" ]; then
    echo "Positional parameter 1 is empty. Please, provide an input image as parameter 1."
    exit
fi

if [ "$2" == "" ]; then
    echo "Positional parameter 2 is empty. Please, provide the amount of pixels per thread"
    exit
fi


for i in {1..10}
do
 # echo "------------------"
 # echo "Experiment $i"
  ./Debug/CUDA_Sobel $INPUT_IMG $THREADS_PER_PIXEL
done
