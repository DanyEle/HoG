/*
 * image_operations.h
 *
 *  Created on: Jun 14, 2019
 *      Author: daniele
 */

#ifndef IMAGE_OPERATIONS_H_
#define IMAGE_OPERATIONS_H_

typedef unsigned char byte;


int rgbToGray(byte *rgb, byte **grayImage, int buffer_size);

void itConv(byte *buffer, int buffer_size, int width, int *op, byte **res);

int convolution(byte *X, int *Y, int c_size);

void makeOpMem(byte *buffer, int buffer_size, int width, int cindex, byte *op_mem);

void contour(byte *sobel_h, byte *sobel_v, int gray_size, byte **contour_img);


#endif /* IMAGE_OPERATIONS_H_ */
