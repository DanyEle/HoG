//============================================================================
// Name        : main.c
// Author      : Daniele Gadler
// Version     :
// Description : Sobel operator in native C
//============================================================================


#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "file_operations.h"
#include "image_operations.h"
#include "math.h"
#include "string.h"
#include <sys/time.h>


typedef unsigned char byte;

#define STRING_BUFFER_SIZE 1024


int main( int argc, char** argv )
{
	struct timeval comp_start_load_img, comp_end_load_img;
	gettimeofday(&comp_start_load_img, NULL);

	 if(argc < 2)
	 {
		printf("You did not provide any input image name and thread. Usage: output [input_image_name] . \n");
		return -2;
	 }

	bool intermediate_output = false;

	//###########1. STEP - LOAD THE IMAGE, ITS HEIGHT, WIDTH AND CONVERT IT TO RGB FORMAT#########

	//Specify the input image. Formats supported: png, jpg, GIF.
	//char * fileInputName = "imgs_in/nonna.jpg";

	char * fileInputName = argv[1];
	char * spaceDiv = " ";
	char * fileOutputRGB = "imgs_out/image.rgb";
	char *pngStrings[4] = {"convert ", fileInputName, spaceDiv, fileOutputRGB};
	char * strPngToRGB = arrayStringsToString(pngStrings, 4, STRING_BUFFER_SIZE);
	//Put back
	//printf("Loading input image [%s] \n", fileInputName);

	gettimeofday(&comp_end_load_img, NULL);

	//actually execute the conversion from PNG to RGB, as that format is required for the program
	struct timeval  i_o_start_load_img, i_o_end_load_img;
	gettimeofday(&i_o_start_load_img, NULL);
	int status_conversion = system(strPngToRGB);
	gettimeofday(&i_o_end_load_img, NULL);

	struct timeval  comp_start_image_processing, comp_end_image_processing;

	gettimeofday(&comp_start_image_processing, NULL);

	if(status_conversion != 0)
	{
		printf("ERROR! Conversion of input PNG image to RGB was not successful. Program aborting.\n");
		return -1;
	}
	//Put back
	//printf("Converted input image to RGB [%s] \n", fileOutputRGB);

	//get the height and width of the input image
	int width = 0;
	int height = 0;

	getImageSize(fileInputName, &width, &height);

	//Put back
	//printf("Size of the loaded image: width=%d height=%d \n", width, height);

	//Three dimensions because the input image is in colored format(R,G,B)
	int rgb_size = width * height * 3;
	//Put back
	//printf("Total amount of pixels in RGB input image is [%d] \n", rgb_size);
	//Used as a buffer for all pixels of the image
	byte * rgbImage;

	//Load up the input image in RGB format into one single flattened array (rgbImage)
	readFile(fileOutputRGB, &rgbImage, rgb_size);

	//#########2. STEP - CONVERT IMAGE TO GRAY-SCALE #################À
	char * file_png_gray = "imgs_out/img_gray.png";

	char str_width[100];
	sprintf(str_width, "%d", width);

	char str_height[100];
	sprintf(str_height, "%d", height);

	byte * grayImage;
	int gray_size  = rgbToGray(rgbImage, &grayImage, rgb_size);
	char * file_gray = "imgs_out/img_gray.gray";

	if(intermediate_output)
	{
		writeFile(file_gray, grayImage, gray_size);
		printf("Total amount of pixels in gray-scale image is [%d] \n", gray_size);

		char * pngConvertGray[8] = {"convert -size ", str_width, "x", str_height, " -depth 8 ", file_gray, spaceDiv, file_png_gray};
		char * strGrayToPNG = arrayStringsToString(pngConvertGray, 8, STRING_BUFFER_SIZE);
		system(strGrayToPNG);

		printf("Converted gray image to PNG [%s]\n", file_png_gray);
	}

	//######################3. Step - Compute vertical and horizontal gradient ##########
    byte * sobel_h_res;
    byte * sobel_v_res;

    //kernel for the horizontal axis
    int sobel_h[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};

    itConv(grayImage, gray_size, width, sobel_h, &sobel_h_res);

    char * strGradToPNG;

    if(intermediate_output)
    {
		//output the horizontal axis-gradient to a file
		char * file_out_h_grad = "imgs_out/sobel_horiz_grad.gray";
		//Output the horizontal axis' gradient calculation
		writeFile(file_out_h_grad, sobel_h_res, gray_size);
		printf("Output horizontal gradient to [%s] \n", file_out_h_grad);
		char * fileHorGradPNG = "imgs_out/sobel_horiz_grad.png";
		printf("Converted horizontal gradient: ");
		printf("[%s] \n", fileHorGradPNG);
		//Convert the output file to PNG
		char * pngConvertHor[8] = {"convert -size ", str_width, "x", str_height, " -depth 8 ", file_out_h_grad, spaceDiv, fileHorGradPNG};
		char * strGradToPNG = arrayStringsToString(pngConvertHor, 8, STRING_BUFFER_SIZE);
		system(strGradToPNG);
    }

    //kernel for the vertical axis
    int sobel_v[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

    itConv(grayImage, gray_size, width, sobel_v, &sobel_v_res);

    if(intermediate_output)
    {

		char * file_out_v_grad = "imgs_out/sobel_vert_grad.gray";

		//Output the vertical axis' gradient calculated
		writeFile(file_out_v_grad, sobel_v_res, gray_size);

		printf("Output vertical gradient to [%s] \n", file_out_v_grad);
		char * fileVerGradPNG = "imgs_out/sobel_vert_grad.png";

		char * pngConvertVer[8] = {"convert -size ", str_width, "x", str_height, " -depth 8 ", file_out_v_grad, spaceDiv, fileVerGradPNG};

		strGradToPNG = arrayStringsToString(pngConvertVer, 8, STRING_BUFFER_SIZE);
		system(strGradToPNG);
    }

	//#############4. Step - Compute the countour by putting together the vertical and horizontal gradients####
	byte * countour_img;

    contour(sobel_h_res, sobel_v_res, gray_size, &countour_img);
    char * file_sobel_out = "imgs_out/sobel_countour.gray";

	gettimeofday(&comp_end_image_processing, NULL);

	struct timeval i_o_start_write_gray_image, i_o_end_write_gray_image;

	gettimeofday(&i_o_start_write_gray_image, NULL);
    writeFile(file_sobel_out, countour_img, gray_size);
	gettimeofday(&i_o_end_write_gray_image, NULL);

	struct timeval comp_start_str_conversion, comp_end_str_conversion;
	gettimeofday(&comp_start_str_conversion, NULL);
	//Put back
	//printf("Output countour to [%s] \n", file_sobel_out);
    char * file_sobel_png = "imgs_out/sobel_countour.png";
	char * pngConvertContour[8] = {"convert -size ", str_width, "x", str_height, " -depth 8 ", file_sobel_out, spaceDiv, file_sobel_png};
    char * strSobelToPNG = arrayStringsToString(pngConvertContour, 8, STRING_BUFFER_SIZE);
	gettimeofday(&comp_end_str_conversion, NULL);

	struct timeval i_o_start_png_conversion, i_o_end_png_conversion;
	gettimeofday(&i_o_start_png_conversion, NULL);
   	system(strSobelToPNG);
	gettimeofday(&i_o_end_png_conversion, NULL);

	//Put back
    //printf("Converted countour: [%s] \n", file_sobel_png);
	//printf("SUCCESS! Successfully applied Sobel filter to the input image!\n");

	//#############5. Step - Display the elapsed time in the different parts of the code

	//##I/O time
	double i_o_time_load_img = compute_elapsed_time(i_o_start_load_img, i_o_end_load_img);
	double i_o_time_write_gray_img = compute_elapsed_time(i_o_start_write_gray_image, i_o_end_write_gray_image);
	double i_o_time_write_png_img = compute_elapsed_time(i_o_start_png_conversion, i_o_end_png_conversion);

	double total_time_i_o = i_o_time_load_img + i_o_time_write_gray_img + i_o_time_write_png_img;

	//printf("Time spent on I/O operations from/to disk: [%f] ms\n", total_time_i_o);
	printf("%f \n", total_time_i_o);

	double comp_time_load_img = compute_elapsed_time(comp_start_load_img, i_o_end_load_img);
	double comp_time_img_process = compute_elapsed_time(comp_start_image_processing, comp_end_image_processing);
	double comp_time_str_process = compute_elapsed_time(comp_start_str_conversion, comp_start_str_conversion);

	double total_time_comp = comp_time_load_img + comp_time_img_process + comp_time_str_process;

	printf("%f \n", total_time_comp);


    return 0;
}

