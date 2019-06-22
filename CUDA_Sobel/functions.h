
#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <sys/time.h>
#include <stdbool.h>


typedef unsigned char byte;


void read_file(const char *file_name, byte **buffer, int buffer_size);
void write_file(const char *file_name, byte *buffer, int buffer_size);
int get_image_size(const char *fn, int *x,int *y);
char * array_strings_to_string(const char ** strings, int stringsAmount, int buffer_size);
void get_dimension_from_RGB_vec(int dimension, byte* rgbImage,  byte** dim_vector, int gray_size);
double compute_elapsed_time(struct timeval time_begin, struct timeval time_end);
#endif
