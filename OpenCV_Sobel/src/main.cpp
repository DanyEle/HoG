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


using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
	  //set the amount of threads for testing the performance while computing the sobel operator
	  cv::setNumThreads(1);
	  auto start = chrono::steady_clock::now();

	  Mat src, src_gray;
	  Mat grad;
	  char const * window_name = "Sobel Operator - Simple Edge Detector";
	  int scale = 1;
	  int delta = 0;
	  int ddepth = CV_16S;

	  ///1. Step - load an image from disk
	  src = imread("lena.png");

	  //if no picture passed as input argument, then just terminate already
	  if( !src.data )
	  { return -1; }

	  //2. Step - Apply a gaussian blur filter to reduce the noise, using kernel size = 3
	  GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );

	  //3. Step - Converted the filtered image to grayscale
	  cvtColor( src, src_gray, CV_BGR2GRAY );

	  /// Create window
	  namedWindow( window_name, CV_WINDOW_AUTOSIZE );

	  //4. Step - Compute derivatives over the X and Y axes
	  Mat grad_x, grad_y;
	  Mat abs_grad_x, abs_grad_y;

	  //5. Step - Gradient over X axis
	  //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
	  Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
	  convertScaleAbs( grad_x, abs_grad_x );

	  //5. Step - Gradient over Y axis
	  //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	  Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
	  convertScaleAbs( grad_y, abs_grad_y );

	  //6. Step - approximate the gradient by adding both directional gradients
	  addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

	  //And show the final result...
	  imshow( window_name, grad );

	  auto end = chrono::steady_clock::now();
	  auto diff = end - start;

	  cout << chrono::duration <double, milli> (diff).count() << " ms" << endl;

	  waitKey(0);

	  return 0;
}
