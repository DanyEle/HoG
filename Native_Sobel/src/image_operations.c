//Credits for this file go to: https://github.com/petermlm/SobelFilter


#include "image_operations.h"
#include "file_operations.h"


typedef unsigned char byte;

#define SOBEL_OP_SIZE 9

//Output: imgs_out/img_gray.png as an image containing the gray-scale input image
void output_gray_scale_image(_Bool intermediate_output, byte * gray_image, int gray_size, char * str_width, char * str_height, int string_buffer_size, char * png_file_name)
{
	if(intermediate_output)
	{
		char * file_gray = "imgs_out/img_gray.gray";
		writeFile(file_gray, gray_image, gray_size);

		char * pngConvertGray[8] = {"convert -size ", str_width, "x", str_height, " -depth 8 ", file_gray, " ", png_file_name};
		char * strGrayToPNG = arrayStringsToString(pngConvertGray, 8, string_buffer_size);
		system(strGrayToPNG);

		printf("Output gray-scale image [%s] \n", file_gray);
	}

}

//Used both for horizontal gradient and vertical gradient
//sobel_res = sobel_h_res or sobel_v_res
void output_gradient(_Bool intermediate_output, byte * sobel_res, int gray_size, char * str_width, char * str_height, int string_buffer_size, char * png_file_name)
{
	  if(intermediate_output)
	  {
			//output the horizontal axis-gradient to an image file
	        char * file_out_grad = "imgs_out/sobel_grad.gray";
			writeFile(file_out_grad, sobel_res, gray_size);
			//Convert the output file to PNG
			char * pngConvert[8] = {"convert -size ", str_width, "x", str_height, " -depth 8 ", file_out_grad, " ", png_file_name};
			char * strGradToPNG = arrayStringsToString(pngConvert, 8, string_buffer_size);
			system(strGradToPNG);
			printf("Output [%s] \n", png_file_name);

	   }
}



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
    for(int i=0; i < gray_size; i++)
    {
    	//Formula according to: https://stackoverflow.com/questions/17615963/standard-rgb-to-grayscale-conversion
        *p_gray = 0.30*p_rgb[0] + 0.59*p_rgb[1] + 0.11*p_rgb[2];
        p_rgb += 3;
        p_gray++;
    }

    return gray_size;
}



void makeOpMem(byte *buffer, int buffer_size, int width, int cindex, byte *op_mem)
{
    int bottom = cindex-width < 0;
    int top = cindex+width >= buffer_size;
    int left = cindex % width == 0;
    int right = (cindex+1) % width == 0;

    op_mem[0] = !bottom && !left  ? buffer[cindex-width-1] : 0;
    op_mem[1] = !bottom           ? buffer[cindex-width]   : 0;
    op_mem[2] = !bottom && !right ? buffer[cindex-width+1] : 0;

    op_mem[3] = !left             ? buffer[cindex-1]       : 0;
    op_mem[4] = buffer[cindex];
    op_mem[5] = !right            ? buffer[cindex+1]       : 0;

    op_mem[6] = !top && !left     ? buffer[cindex+width-1] : 0;
    op_mem[7] = !top              ? buffer[cindex+width]   : 0;
    op_mem[8] = !top && !right    ? buffer[cindex+width+1] : 0;
}


int convolution(byte *X, int *Y, int c_size)
{
    int sum = 0;

    for(int i=0; i<c_size; i++) {
        sum += X[i] * Y[c_size-i-1];
    }

    return sum;
}



void itConv(byte *buffer, int buffer_size, int width, int *op, byte **res)
{
    // Allocate memory for result
    *res = malloc(sizeof(byte) * buffer_size);

    // Temporary memory for each pixel operation
    byte op_mem[SOBEL_OP_SIZE];
    memset(op_mem, 0, SOBEL_OP_SIZE);

    // Make convolution for every pixel
    for(int i=0; i < buffer_size; i++)
    {
        // Make op_mem
        makeOpMem(buffer, buffer_size, width, i, op_mem);

        // Convolution
        (*res)[i] = (byte) abs(convolution(op_mem, op, SOBEL_OP_SIZE));

        /*
         * The abs function is used in here to avoid storing negative numbers
         * in a byte data type array. It wouldn't make a different if the negative
         * value was to be stored because the next time it is used the value is
         * squared.
         */
    }
}



void contour(byte *sobel_h, byte *sobel_v, int gray_size, byte **contour_img)
{
    // Allocate memory for contour_img
    *contour_img = malloc(sizeof(byte) * gray_size);

    // Iterate through every pixel to calculate the contour image
    for(int i = 0; i < gray_size; i++)
    {
        (*contour_img)[i] = (byte) sqrt(pow(sobel_h[i], 2) + pow(sobel_v[i], 2));
    }
}

double compute_elapsed_time(struct timeval time_begin, struct timeval time_end)
{
	//time in microseconds (us)
	double time_elapsed_us =  (double) (time_end.tv_usec - time_begin.tv_usec) / 1000000 +  (double) (time_end.tv_sec - time_begin.tv_sec);

	//return time in milliseconds (ms)
	double time_elapsed_ms = time_elapsed_us * 1000;

	return time_elapsed_ms;
}
