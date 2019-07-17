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

//used to track start and end time
#define get_time(time) (gettimeofday(&time, NULL))

//false --> No vertical gradient and horizontal gradient are output
//true --> Vertical gradient and horizontal gradient are output
#define INTERMEDIATE_OUTPUT false

int main(int argc, char** argv)
{
	//initialize all the timevals at the beginning of the program to avoid cluttered declarations later on
	struct timeval comp_start_load_img, comp_end_load_img;
	struct timeval i_o_start_load_img, i_o_end_load_img;
	struct timeval comp_start_image_processing, comp_end_image_processing;
	struct timeval i_o_start_write_gray_image, i_o_end_write_gray_image;
	struct timeval i_o_start_png_conversion, i_o_end_png_conversion;

	get_time(comp_start_load_img);

	if (argc < 2)
	{
		printf("You did not provide any input image name and thread. Usage: output [input_image_name] . \n");
		return -2;
	}

	//###########1. STEP - LOAD THE IMAGE, ITS HEIGHT, WIDTH AND CONVERT IT TO RGB FORMAT#########

	//Specify the input image. Formats supported: png, jpg, GIF.

	char * file_output_RGB = "imgs_out/image.rgb";
	char * png_strings[4] = { "convert ", argv[1], " ", file_output_RGB };
	char * str_PNG_to_RGB = array_strings_to_string(png_strings, 4,	STRING_BUFFER_SIZE);

	//printf("Loading input image [%s] \n", fileInputName); //debug

	get_time(comp_end_load_img);

	//actually execute the conversion from PNG to RGB, as that format is required for the program
	get_time(i_o_start_load_img);
	int status_conversion = system(str_PNG_to_RGB);
	get_time(i_o_end_load_img);

	get_time(comp_start_image_processing);

	if (status_conversion != 0)
	{
		printf("ERROR! Conversion of input PNG image to RGB was not successful. Program aborting.\n");
		return -1;
	}

	//get the height and width of the input image
	int width = 0;
	int height = 0;

	get_image_size(argv[1], &width, &height);

	//printf("Size of the loaded image: width=%d height=%d \n", width, height); //debug

	//Three dimensions because the input image is in colored format(R,G,B)
	int rgb_size = width * height * 3;

	//Used as a buffer for all pixels of the image
	byte * rgb_image;

	//Load up the input image in RGB format into one single flattened array (rgb_image)
	read_file(file_output_RGB, &rgb_image, rgb_size);

	//#########2. STEP - CONVERT IMAGE TO GRAY-SCALE #################

	//convert the width and height to char *
	char str_width[100];
	sprintf(str_width, "%d", width);

	char str_height[100];
	sprintf(str_height, "%d", height);

	byte * grayImage;

	//convert the RGB vector to gray-scale
	int gray_size = rgb_to_gray(rgb_image, &grayImage, rgb_size);

	//output the gray-scale image to a PNG file if INTERMEDIATE_OUTPUT == true
	output_gray_scale_image(INTERMEDIATE_OUTPUT, grayImage, gray_size,	str_width, str_height, STRING_BUFFER_SIZE, "imgs_out/img_gray.png");

	//######################3. Step - Compute vertical and horizontal gradient ##########
	byte * sobel_h_res;
	byte * sobel_v_res;

	//kernel for the horizontal axis
	int sobel_h[] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };

	itConv(grayImage, gray_size, width, sobel_h, &sobel_h_res);

	//output horizontal gradient to a PNG file if INTERMEDIATE_OUTPUT == true
	output_gradient(INTERMEDIATE_OUTPUT, sobel_h_res, gray_size, str_width,	str_height, STRING_BUFFER_SIZE, "imgs_out/sobel_horiz_grad.png");

	//kernel for the vertical axis
	int sobel_v[] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };

	itConv(grayImage, gray_size, width, sobel_v, &sobel_v_res);

	//output vertical gradient to a PNG file if INTERMEDIATE_OUTPUT == true
	output_gradient(INTERMEDIATE_OUTPUT, sobel_v_res, gray_size, str_width,	str_height, STRING_BUFFER_SIZE, "imgs_out/sobel_vert_grad.png");

	//#############4. Step - Compute the countour by putting together the vertical and horizontal gradients####
	byte * countour_img;
	contour(sobel_h_res, sobel_v_res, gray_size, &countour_img);
	get_time(comp_end_image_processing);

	get_time(i_o_start_write_gray_image);
	write_file("imgs_out/sobel_countour.gray", countour_img, gray_size);
	get_time(i_o_end_write_gray_image);

	//always output the final sobel countour
	get_time(i_o_start_png_conversion);
	output_gradient(true, countour_img, gray_size, str_width, str_height, STRING_BUFFER_SIZE, "imgs_out/sobel_countour.png");
	get_time(i_o_end_png_conversion);

	//#############5. Step - Display the elapsed time in the different parts of the code

	//##I/O time
	double i_o_time_load_img = compute_elapsed_time(i_o_start_load_img,	i_o_end_load_img);
	double i_o_time_write_gray_img = compute_elapsed_time(i_o_start_write_gray_image, i_o_end_write_gray_image);
	double i_o_time_write_png_img = compute_elapsed_time(i_o_start_png_conversion, i_o_end_png_conversion);

	double total_time_i_o = i_o_time_load_img + i_o_time_write_gray_img	+ i_o_time_write_png_img;

	printf("Time spent on I/O operations from/to disk: [%f] ms\n", total_time_i_o); //debug
	//printf("%f \n", total_time_i_o);

	double comp_time_load_img = compute_elapsed_time(comp_start_load_img, i_o_end_load_img);
	double comp_time_img_process = compute_elapsed_time(comp_start_image_processing, comp_end_image_processing);

	double total_time_comp = comp_time_load_img + comp_time_img_process;

	printf("Time spent in computations: [%f] ms\n", total_time_comp); //debug
	//printf("%f \n", total_time_comp);

	//##Overall total time

	double overall_total_time = total_time_comp + total_time_i_o;

	printf("Overall time spent in program: [%f] ms\n", overall_total_time); //debug
	//printf("%f \n", overall_total_time);

	//let's deallocate the memory to avoid any memory leaks
	free(grayImage);
	free(sobel_h_res);
	free(sobel_v_res);
	free(countour_img);
	return 0;
}

