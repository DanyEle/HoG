
#include "file_operations.h"

typedef unsigned char byte;



//Input: - rgb image contained in the 'rgb' array
//		 - buffer size: the size of the RGB image
//Output: gray, an array containing the gray-scale image
int rgbToGray(byte *rgb, byte **grayImage, int buffer_size)
{
    // Take size for gray image and allocate memory. Just one dimension for gray-scale image
    int gray_size = buffer_size / 3;
    *grayImage = malloc(sizeof(byte) * gray_size);

    // Make pointers for iteration
    byte *p_rgb = rgb;
    byte *p_gray = *grayImage;

    // Calculate the value for every pixel in gray
    for(int i=0; i<gray_size; i++)
    {
        *p_gray = 0.30*p_rgb[0] + 0.59*p_rgb[1] + 0.11*p_rgb[2];
        p_rgb += 3;
        p_gray++;
    }

    return gray_size;
}
