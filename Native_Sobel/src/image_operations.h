/*
 * image_operations.h
 *
 *  Created on: Jun 14, 2019
 *      Author: daniele
 */

#ifndef IMAGE_OPERATIONS_H_
#define IMAGE_OPERATIONS_H_

#include <sys/time.h>
#include "string.h"
#include "stdlib.h"
#include "math.h"
#include "stdio.h"
#include <stdbool.h>

//typedef unsigned char byte;
#define SOBEL_OP_SIZE 9

typedef unsigned char byte;


int rgb_to_gray(byte *rgb, byte **grayImage, int buffer_size);

void itConv(byte *buffer, int buffer_size, int width, int *op, byte **res);

int convolution(byte *X, int *Y, int c_size);

void make_op_mem(byte *buffer, int buffer_size, int width, int cindex, byte *op_mem);

void contour(byte *sobel_h, byte *sobel_v, int gray_size, byte **contour_img);

double compute_elapsed_time(struct timeval time_begin, struct timeval time_end);

void output_gray_scale_image(bool intermediate_output, byte * gray_image, int gray_size, char * str_width, char * str_height, int string_buffer_size, char * png_file_name);

void output_gradient(bool intermediate_output, byte * sobel_res, int gray_size, char * str_width, char * str_height, int string_buffer_size, char * png_file_name);

#endif /* IMAGE_OPERATIONS_H_ */
