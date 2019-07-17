//============================================================================
// Name        : main.cpp
// Author      : Daniele Gadler
// Version     : 1.0
// Description : Sobel operator in C++, using OpenCV
//============================================================================

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <sys/time.h>


#define STRING_BUFFER_SIZE 1024


using namespace cv;
using namespace std;


//Input: strings: an array containing strings
//		 stringsAmount: the amount of strings present in the array
//	     buffer_size: the size of the buffer for the char* to be created (max length of buffer)
//Output: a string (char*) containing the concatenation of all strings in the array
//passed as input
char * array_strings_to_string(const char ** strings, int stringsAmount, int buffer_size)
{
	char * strConvert = (char*) malloc(buffer_size);

	//first element is just copied
	strcpy(strConvert, strings[0]);

	for(int i = 1; i < stringsAmount; i++)
	{
		//all the following elements are appended
		strcat(strConvert, strings[i]);
	}
	return strConvert;
}


double compute_elapsed_time(struct timeval time_begin, struct timeval time_end)
{
	//time in microseconds (us)
	double time_elapsed_us =  (double) (time_end.tv_usec - time_begin.tv_usec) / 1000000 +  (double) (time_end.tv_sec - time_begin.tv_sec);

	//return time in milliseconds (ms)
	double time_elapsed_ms = time_elapsed_us * 1000;

	return time_elapsed_ms;
}


//argv[1] = input image name
int main( int argc, char** argv )
{
	  struct timeval comp_start_init_vars, comp_end_init_vars;

	  gettimeofday(&comp_start_init_vars, NULL);

	  if(argc < 2) //was 3
	  {
		printf("You did not provide any input image. Please, pass an input image and retry. \n");
		return -2;
	  }

	  //int amountThreads = atoi(argv[1]);
	  char const * file_input_name = argv[1];

	  Mat src, src_gray;
	  Mat grad;
	  int scale = 1;
	  int delta = 0;
	  int ddepth = CV_16S;

	  gettimeofday(&comp_end_init_vars, NULL);

	  struct timeval i_o_start_read_img, i_o_end_read_img;

	  gettimeofday(&i_o_start_read_img, NULL);

	  ///1. Step - load an image from disk
	  src = imread(file_input_name);
	  gettimeofday(&i_o_end_read_img, NULL);

	  struct timeval comp_start_str_sobel, comp_end_str_sobel;
	  gettimeofday(&comp_start_str_sobel, NULL);

	  //if no picture passed as input argument, then just terminate already
	  if( !src.data )
	  { return -1; }

	  const char * spaceDiv = " ";
	  const char * file_output_RGB = "imgs_out/image.rgb";
	  const char *PNG_strings[4] = {"convert ", file_input_name, spaceDiv, file_output_RGB};
	  const char * str_PNG_to_RGB = array_strings_to_string(PNG_strings, 4, STRING_BUFFER_SIZE);
	  //printf("Loading input image [%s] \n", file_input_name);
	  gettimeofday(&comp_end_str_sobel, NULL);

	  struct timeval i_o_start_conv_rgb_png, i_o_end_conv_rgb_png;

	  gettimeofday(&i_o_start_conv_rgb_png, NULL);
	  //actually execute the conversion from PNG to RGB, as that format is required for the program
	  int status_conversion = system(str_PNG_to_RGB);
	  gettimeofday(&i_o_end_conv_rgb_png, NULL);


	  struct timeval comp_start_sobel_filter, comp_end_sobel_filter;

	  gettimeofday(&comp_start_sobel_filter, NULL);

	  if(status_conversion != 0)
	  {
			printf("ERROR! Conversion of input PNG image to RGB was not successful. Program aborting.\n");
			return -1;
	  }

	  //2. Step - Converted the filtered image to grayscale
	  cvtColor( src, src_gray, CV_BGR2GRAY );

	  //3. Step - Gradient over X axis
	  Mat grad_x, grad_y;
	  Mat abs_grad_x, abs_grad_y;

	  Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
	  convertScaleAbs( grad_x, abs_grad_x );

	  //4. Step - Gradient over Y axis
	  Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
	  convertScaleAbs( grad_y, abs_grad_y );

	  //5. Step - approximate the gradient by adding both directional gradients
	  addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

	  //And show the final result...
	  /// Create window
	  //char const * window_name = "Sobel Operator - Simple Edge Detector";
	  //namedWindow( window_name, CV_WINDOW_AUTOSIZE );
	  //imshow( window_name, grad );

	  const char * file_sobel_out = "imgs_out/sobel_countour.png";
	  //printf("Converted countour: [%s] \n", file_sobel_out);
	  gettimeofday(&comp_end_sobel_filter, NULL);

	  struct timeval i_o_start_img_write, i_o_end_img_write;

	  gettimeofday(&i_o_start_img_write, NULL);
	  imwrite(file_sobel_out, grad);
	  //printf("SUCCESS! Successfully applied Sobel filter to the input image!\n");
	  gettimeofday(&i_o_end_img_write, NULL);


	  //6. Step - compute the time elapsed in the 'computation' phase and in the 'i/o' phase
	  //i/o phase
	  double i_o_time_read_img = compute_elapsed_time(i_o_start_read_img, i_o_end_read_img);
	  double i_o_time_conv_img = compute_elapsed_time(i_o_start_conv_rgb_png, i_o_end_conv_rgb_png);
	  double i_o_time_write_img = compute_elapsed_time(i_o_start_img_write, i_o_end_img_write);

	  double total_time_i_o = i_o_time_read_img + i_o_time_conv_img + i_o_time_write_img;

	  //printf("%f \n", total_time_i_o);
	  printf("Time spent on I/O operations from/to disk: [%f] ms\n", total_time_i_o); //debug

	  //compute phase
	  double comp_time_init_vars = compute_elapsed_time(comp_start_init_vars, comp_end_init_vars);
	  double comp_time_str_sobel = compute_elapsed_time(comp_start_str_sobel, comp_end_str_sobel);
	  double comp_time_sobel_filter = compute_elapsed_time(comp_start_sobel_filter, comp_end_sobel_filter);

	  double total_time_compute = comp_time_init_vars + comp_time_str_sobel + comp_time_sobel_filter;

	  printf("Time spent in computations: [%f] ms\n", total_time_compute); //debug
	  //printf("%f \n", total_time_compute);

	  //overall: i/o + compute phase

	  double overall_total_time = total_time_i_o + total_time_compute;

	  printf("Overall time spent in program: [%f] ms\n", overall_total_time); //debug
	  //printf("%f \n", overall_total_time);

	  //let's free the memory
	  src.release();
	  src_gray.release();
	  grad_x.release();
	  grad_y.release();
	  abs_grad_x.release();
	  abs_grad_y.release();
	  grad.release();

	  return 0;
}


