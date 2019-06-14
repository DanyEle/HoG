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

#endif /* IMAGE_OPERATIONS_H_ */
