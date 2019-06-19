//============================================================================
// Name        : main.cpp
// Author      : Daniele Gadler
// Version     :
// Description : Sobel operator in C++, using OpenCV
// Source: https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/sobel_derivatives/sobel_derivatives.html
//============================================================================

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <time.h>
#include <chrono>


#define STRING_BUFFER_SIZE 1024


using namespace cv;
using namespace std;


//Input: strings: an array containing strings
//		 stringsAmount: the amount of strings present in the array
//	     buffer_size: the size of the buffer for the char* to be created (max length of buffer)
//Output: a string (char*) containing the concatenation of all strings in the array
//passed as input
char * arrayStringsToString(const char ** strings, int stringsAmount, int buffer_size)
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

//argv[1] = amount of threads
//argv[2] = input image name
int main( int argc, char** argv )
{
	 if(argc < 3)
	 {
		printf("You did not provide any input image name and thread. Usage: output [amount_threads] [input_image_name] and retry. \n");
		return -2;
	 }

	  int amountThreads = atoi(argv[1]);
	  char const * fileInputName = argv[2];

	  //set the amount of threads for testing the performance while computing the sobel operator
	  cv::setNumThreads(amountThreads);

	  Mat src, src_gray;
	  Mat grad;
	  int scale = 1;
	  int delta = 0;
	  int ddepth = CV_16S;

	  //char const * inputFileName = "imgs_in/lena.png";
	  ///1. Step - load an image from disk
	  src = imread(fileInputName);

	  //if no picture passed as input argument, then just terminate already
	  if( !src.data )
	  { return -1; }

	  const char * spaceDiv = " ";
	  const char * fileOutputRGB = "imgs_out/image.rgb";
	  const char *pngStrings[4] = {"convert ", fileInputName, spaceDiv, fileOutputRGB};
	  const char * strPngToRGB = arrayStringsToString(pngStrings, 4, STRING_BUFFER_SIZE);

	  printf("Loading input image [%s] \n", fileInputName);

	  //actually execute the conversion from PNG to RGB, as that format is required for the program
	  int status_conversion = system(strPngToRGB);

	  if(status_conversion != 0)
	  {
			printf("ERROR! Conversion of input PNG image to RGB was not successful. Program aborting.\n");
			return -1;
	  }
	  printf("Converted input image to RGB [%s] \n", fileOutputRGB);

	  //2. Step - Apply a gaussian blur filter to reduce the noise, using kernel size = 3
	  //GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );

	  //3. Step - Converted the filtered image to grayscale
	  cvtColor( src, src_gray, CV_BGR2GRAY );

	  //4. Step - Compute derivatives over the X and Y axes
	  Mat grad_x, grad_y;
	  Mat abs_grad_x, abs_grad_y;

	  //5. Step - Gradient over X axis
	  Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
	  convertScaleAbs( grad_x, abs_grad_x );

	  //5. Step - Gradient over Y axis
	  //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	  Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
	  convertScaleAbs( grad_y, abs_grad_y );

	  //6. Step - approximate the gradient by adding both directional gradients
	  addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

	  //And show the final result...
	  /// Create window
	  //char const * window_name = "Sobel Operator - Simple Edge Detector";
	  //namedWindow( window_name, CV_WINDOW_AUTOSIZE );
	  //imshow( window_name, grad );

	  const char * file_sobel_out = "imgs_out/sobel_countour.png";
	  printf("Converted countour: [%s] \n", file_sobel_out);
	  imwrite(file_sobel_out, grad);
	  printf("SUCCESS! Successfully applied Sobel filter to the input image!\n");

	  return 0;
}


