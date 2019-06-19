g++ src/main.cpp `pkg-config --cflags --libs opencv` -o output

./output 1 imgs_in/lena.png
