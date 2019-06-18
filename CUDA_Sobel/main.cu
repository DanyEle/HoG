#include <stdio.h>

#include "file_operations.c"

#define STRING_BUFFER_SIZE 1024

#define SOBEL_OP_SIZE 9
#include "string.h"
#include "stdlib.h"
#include "math.h"
#include <stdlib.h>
#include <stdio.h>






#define HANDLE_ERROR( err ) ( HandleError( err, __FILE__, __LINE__ ) )

static void HandleError( cudaError_t err, const char *file, int line )
{
    if (err != cudaSuccess)
      {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}



//called from 'it_conv' function
__device__ int convolution(byte *X, int *Y, int c_size)
{
    int sum = 0;

    for(int i=0; i<c_size; i++) {
        sum += X[i] * Y[c_size-i-1];
    }

    return sum;
}

//called from 'it_conv' function
__device__ void makeOpMem(byte *buffer, int buffer_size, int width, int cindex, byte *op_mem)
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






__global__ void it_conv(byte *buffer, int buffer_size, int width, int *dev_op, byte **dev_res)
{
    // Temporary memory for each pixel operation
    byte op_mem[SOBEL_OP_SIZE];
    memset(op_mem, 0, SOBEL_OP_SIZE);

	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	int i = 0;

    // Make convolution for every pixel. Each pixel --> one thread.
    //for(int i=0; i < buffer_size; i++)
    while(tid < buffer_size)
    {
        // Make op_mem
        makeOpMem(buffer, buffer_size, width, i, op_mem);

        // Convolution
        (*dev_res)[i] = (byte) abs(convolution(op_mem, dev_op, SOBEL_OP_SIZE));

        /*
         * The abs function is used in here to avoid storing negative numbers
         * in a byte data type array. It wouldn't make a different if the negative
         * value was to be stored because the next time it is used the value is
         * squared.
         */
    	tid += blockDim.x * gridDim.x;

    	i = 0;

    }
}




//Input: dev_r_vec, dev_g_vec, dev_b_vec: vectors containing the R,G,B components of the input image
//		 gray_size: amount of pixels in the RGB vector / 3
//Output: dev_gray_image: a vector containing the gray-scale pixels of the resulting image

// CUDA kernel to convert an image to gray-scale
//gray-image's memory needs to be pre-allocated
__global__ void rgb_img_to_gray( byte * dev_r_vec, byte * dev_g_vec, byte * dev_b_vec, byte * dev_gray_image, int gray_size)
{
    //Get the id of thread within a block
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	//pixel-wise operation on the R,G,B vectors
	while(tid < gray_size)
	{
		//r, g, b pixels
		byte p_r = dev_r_vec[tid];
		byte p_g = dev_g_vec[tid];
		byte p_b = dev_b_vec[tid];

		//Formula according to: https://stackoverflow.com/questions/17615963/standard-rgb-to-grayscale-conversion
		dev_gray_image[tid] = 0.30 * p_r + 0.59*p_g + 0.11*p_b;
    	tid += blockDim.x * gridDim.x;
	}
}








int main (void)
{
	//###########1. STEP - LOAD THE IMAGE, ITS HEIGHT, WIDTH AND CONVERT IT TO RGB FORMAT#########

		//Specify the input image. Formats supported: png, jpg, GIF.
		const char * fileInputName = "imgs_in/lena.png";

		const char * spaceDiv = " ";
		const char * fileOutputRGB = "imgs_out/image.rgb";
		const char *pngStrings[4] = {"convert ", fileInputName, spaceDiv, fileOutputRGB};
		const char * strPngToRGB = arrayStringsToString(pngStrings, 4, STRING_BUFFER_SIZE);

		printf("Loading input image [%s] \n", fileInputName);

		//actually execute the conversion from PNG to RGB, as that format is required for the program
		int status_conversion = system(strPngToRGB);

		if(status_conversion != 0)
		{
			printf("Conversion of input PNG image to RGB was not successful. Program aborting.");
			return -1;
		}
		printf("Converted input image to RGB [%s] \n", fileOutputRGB);

		//get the height and width of the input image
		int width = 0;
		int height = 0;

		getImageSize(fileInputName, &width, &height);

		printf("Size of the loaded image : width=%d height=%d \n", width, height);

		//Three dimensions because the input image is in colored format(R,G,B)
		int rgb_size = width * height * 3;
		printf("Total amount of pixels in RGB input image is [%d] \n", rgb_size);
		//Used as a buffer for all pixels of the image
		byte * rgb_image;

		//Load up the input image in RGB format into one single flattened array (rgbImage)
		readFile(fileOutputRGB, &rgb_image, rgb_size);

		//########2. step - convert RGB image to gray-scale

	    int gray_size = rgb_size / 3;
	    byte * rVector, * gVector, * bVector;

	    //now take the RGB image vector and create three separate arrays for the R,G,B dimensions
	    getDimensionFromRGBVec(0, rgb_image,  &rVector, gray_size);
	    getDimensionFromRGBVec(1, rgb_image,  &gVector, gray_size);
	    getDimensionFromRGBVec(2, rgb_image,  &bVector, gray_size);

	    //allocate memory on the device for the r,g,b vectors
	    byte * dev_r_vec, * dev_g_vec, * dev_b_vec;
	    HANDLE_ERROR ( cudaMalloc((void **)&dev_r_vec , gray_size*sizeof(byte)));
	    HANDLE_ERROR ( cudaMalloc((void **)&dev_g_vec, gray_size*sizeof(byte)));
	    HANDLE_ERROR ( cudaMalloc((void **)&dev_b_vec, gray_size*sizeof(byte)));

	    //copy the content of the r,g,b vectors from the host to the device
	    HANDLE_ERROR (cudaMemcpy (dev_r_vec , rVector , gray_size*sizeof(byte), cudaMemcpyHostToDevice));
	    HANDLE_ERROR (cudaMemcpy (dev_g_vec , gVector , gray_size*sizeof(byte), cudaMemcpyHostToDevice));
	    HANDLE_ERROR (cudaMemcpy (dev_b_vec , bVector , gray_size*sizeof(byte), cudaMemcpyHostToDevice));

	    //allocate memory on the device for the output gray image
	    byte * dev_gray_image;
	    HANDLE_ERROR ( cudaMalloc((void **)&dev_gray_image, gray_size*sizeof(byte)));

	    //actually run the kernel to convert input RGB file to gray-scale
	    rgb_img_to_gray <<< width, height>>> (dev_r_vec, dev_g_vec, dev_b_vec, dev_gray_image, gray_size) ;

	    byte gray_image[gray_size];

	    //Now take the device gray vector and bring it back to the host
	    HANDLE_ERROR (cudaMemcpy(gray_image , dev_gray_image , gray_size*sizeof(byte) , cudaMemcpyDeviceToHost));

	    //let's see what's in there, shall we?
	    const char * file_gray = "imgs_out/img_gray.gray";

		writeFile(file_gray, gray_image, gray_size);
		printf("Total amount of pixels in gray-scale image is [%d] \n", gray_size);

		const char * file_png_gray = "imgs_out/img_gray.png";

		char str_width[100];
		sprintf(str_width, "%d", width);

		char str_height[100];
		sprintf(str_height, "%d", height);

		const char * pngConvertGray[8] = {"convert -size ", str_width, "x", str_height, " -depth 8 ", file_gray, spaceDiv, file_png_gray};
		char * strGrayToPNG = arrayStringsToString(pngConvertGray, 8, STRING_BUFFER_SIZE);
		system(strGrayToPNG);

		printf("Converted gray image to PNG [%s]\n", file_png_gray);

		//let's de-allocate memory allocated for input R,G,B vectors
	    cudaFree (dev_r_vec);
	    cudaFree (dev_g_vec);
		cudaFree (dev_b_vec);

		//######################3. Step - Compute vertical and horizontal gradient ##########

		//###Compute the HORIZONTAL GRADIENT#####

   	    //host horizontal kernel
		int sobel_h[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};

		int * dev_sobel_h;

		//allocate memory for device horizontal kernel
		HANDLE_ERROR ( cudaMalloc((void **)&dev_sobel_h , 9*sizeof(int)));

		//copy the content of the host horizontal kernel to the device horizontal kernel
	    HANDLE_ERROR (cudaMemcpy (dev_sobel_h , sobel_h , 9*sizeof(int) , cudaMemcpyHostToDevice));

	    //allocate memory for the resulting horizontal gradient on the device
   	    byte * dev_sobel_h_res;
		HANDLE_ERROR ( cudaMalloc((void **)&dev_sobel_h_res , gray_size*sizeof(byte)));

		//perform horizontal gradient calculation
		it_conv <<< width, height>>> (dev_gray_image, gray_size, width, dev_sobel_h, &dev_sobel_h_res);









}

//Don't forget to clean up the device memory!!
