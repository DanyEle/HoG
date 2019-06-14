//============================================================================
// Name        : main.c
// Author      : Daniele Gadler
// Version     :
// Description : Sobel operator in native C
// Credits to  : Pedro Melgueira
//============================================================================


#include <stdio.h>
#include <stdlib.h>
#include "file_operations.h"
#include "image_operations.h"
#include "math.h"
#include "string.h"

typedef unsigned char byte;


int main( int argc, char** argv )
{

	//###########1. STEP - LOAD THE IMAGE, ITS HEIGHT, WIDTH AND CONVERT IT TO RGB FORMAT#########ÀÀ
	//NB: Only support square pictures
	char * fileInputName = "imgs_in/lena.png";
	char * spaceDiv = " ";
	char * fileInputRGB = "imgs_in/lena.rgb";

	char *pngStrings[4];
	pngStrings[0] = "convert ";
	pngStrings[1] = fileInputName;
	pngStrings[2] = spaceDiv;
	pngStrings[3] = fileInputRGB;

	char * strPngToRGB = arrayStringsToString(pngStrings, 4, 1024);

	printf("Loading input image");
	printf(" [%s] ", fileInputName);
	printf("\n");

	//actually execute the conversion from PNG to RGB, as that format is required for the program
	int status_conversion = system(strPngToRGB);

	if(status_conversion != 0)
	{
		printf("Conversion of input PNG image to RGB was not successful. Program aborting.");
		return -1;
	}
	printf("Converted input image to RGB ");
	printf(" [%s] ", fileInputRGB);
	printf("\n");

	//get the height and width of the input image
	int width = 0;
	int height = 0;

	int status_image = getImageSize(fileInputName, &width, &height);

	if(status_image != 0)
	{
		printf("Loading of input image was not successful. Supported formats are: [PNG/GIF/JPEG]");
	}

	printf("Size of the loaded image : width=%d height=%d \n", width, height);

	//Three dimensions because the input image is in colored format(R,G,B)
	int rgb_size = width * height * 3;
	printf("Total amount of pixels in RGB input image is [%d] \n", rgb_size);
	//Used as a buffer for all pixels of the image
	byte * rgbImage;

	//Load up the input image in RGB format into one single flattened array (rgbImage)
	readFile(fileInputRGB, &rgbImage, rgb_size);

	//#########2. STEP - CONVERT IMAGE TO GRAY-SCALE #################À
	byte * grayImage;
	int gray_size_loaded  = rgbToGray(rgbImage, &grayImage, rgb_size);
	char * file_gray = "imgs_out/lena_gray.gray";
	writeFile(file_gray, grayImage, gray_size_loaded);
	printf("Total amount of pixels in gray-scale image is [%d] \n", gray_size_loaded);

	char *pngGrayStrings[4];

	char * file_png_gray = "imgs_out/lena_gray.png";

	pngGrayStrings[0] = "convert -size 512x512 -depth 8 ";
	pngGrayStrings[1] = file_gray;
	pngGrayStrings[2] = spaceDiv;
	pngGrayStrings[3] = file_png_gray;

	char * strGrayToPNG = arrayStringsToString(pngGrayStrings, 4, 1024);
	status_conversion = system(strGrayToPNG);

	if(status_conversion != 0)
	{
		printf("Conversion of input gray image to PNG was not successful. Program aborting.");
	}

	printf("Converted gray image to PNG ");
	printf(" [%s] " , file_png_gray);
	printf("\n");







/*

	//Now actually convert the RGB image to gray-scale


	// Read file to rgb and get size
    byte *sobel_h_res,
         *sobel_v_res,
         *contour_img;

	int gray_size = sobelFilter(rgbImage, &grayImage, &sobel_h_res, &sobel_v_res, &contour_img, width, height);

	// Write gray image
	char * file_gray = "imgs_out/file_gray.gray";
	writeFile(file_gray, grayImage, gray_size);

	char * file_out_h = "imgs_out/file_h.gray";
	char * file_out_v = "imgs_out/file_v.gray";

	// Write image after each sobel operator
	writeFile(file_out_h, sobel_h_res, gray_size);
	writeFile(file_out_v, sobel_v_res, gray_size);

	char * file_out = "imgs_out/sobel_out.gray";

	// Write sobel img to a file
	writeFile(file_out, contour_img, gray_size);

	return 0;

	*/


}

