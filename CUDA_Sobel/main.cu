#include <stdio.h>
#include <chrono>


#include "functions.c"

#define STRING_BUFFER_SIZE 1024

#define SOBEL_OP_SIZE 9
#include "string.h"
#include "stdlib.h"
#include "math.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>



#define HANDLE_ERROR( err ) ( HandleError( err, __FILE__, __LINE__ ) )

static void HandleError( cudaError_t err, const char *file, int line )
{
    if (err != cudaSuccess)
      {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE);
    }
}


__global__ void contour(byte *dev_sobel_h, byte *dev_sobel_v, int gray_size, byte *dev_contour_img)
{
	int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
	int tid_y = threadIdx.y + blockIdx.y * blockDim.y;

	int tid = abs(tid_x - tid_y);


    // Performed on every pixel in parallel to calculate the contour image
    while(tid < gray_size)
    {
        dev_contour_img[tid] = (byte) sqrt(pow((double)dev_sobel_h[tid], 2.0) + pow((double)dev_sobel_v[tid], 2.0));

    	tid += blockDim.x * gridDim.x + blockDim.y * gridDim.y;

    }
}

//called from 'it_conv' function
__device__ int convolution(byte *X, int *Y, int c_size)
{
    int sum = 0;

    for(int i=0; i < c_size; i++) {
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






__global__ void it_conv(byte * buffer, int buffer_size, int width, int * dev_op, byte *dev_res)
{
    // Temporary memory for each pixel operation
    byte op_mem[SOBEL_OP_SIZE];
    memset(op_mem, 0, SOBEL_OP_SIZE);
    int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
	int tid_y = threadIdx.y + blockIdx.y * blockDim.y;

	//simple linearization
	int tid = abs(tid_x - tid_y);

    // Make convolution for every pixel. Each pixel --> one thread.
    while(tid < buffer_size)
    {
        // Make op_mem
        makeOpMem(buffer, buffer_size, width, tid, op_mem);

        dev_res[tid] = (byte) abs(convolution(op_mem, dev_op, SOBEL_OP_SIZE));
        /*
         * The abs function is used in here to avoid storing negative numbers
         * in a byte data type array. It wouldn't make a different if the negative
         * value was to be stored because the next time it is used the value is
         * squared.
         */
    	tid += blockDim.x * gridDim.x + blockDim.y * gridDim.y;
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
	int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
	int tid_y = threadIdx.y + blockIdx.y * blockDim.y;

	//simple linearization of 2D space
	int tid = abs(tid_x - tid_y);

	//pixel-wise operation on the R,G,B vectors
	while(tid < gray_size)
	{
		//r, g, b pixels
		byte p_r = dev_r_vec[tid];
		byte p_g = dev_g_vec[tid];
		byte p_b = dev_b_vec[tid];

		//Formula accordidev_ng to: https://stackoverflow.com/questions/17615963/standard-rgb-to-grayscale-conversion
		dev_gray_image[tid] = 0.30 * p_r + 0.59*p_g + 0.11*p_b;
    	tid += blockDim.x * gridDim.x + blockDim.y * gridDim.y;

	}
}



int main ( int argc, char** argv )
{
		//actual computation
		struct timeval comp_start_load_img, comp_end_load_img;

		gettimeofday(&comp_start_load_img, NULL);
		if(argc < 2)
		{
			printf("You did not provide any input image name. Please, provide an input image name and retry. \n");
			return -2;
		}

		bool intermediate_output = false;

		//###########1. STEP - LOAD THE IMAGE, ITS HEIGHT, WIDTH AND CONVERT IT TO RGB FORMAT#########

		//Specify the input image. Formats supported: png, jpg, GIF.
		//const char * fileInputName = "imgs_in/hua_hua.jpg";
		//Example argv[1] = "imgs_in/hua_hua.pjg";
		const char * fileInputName = argv[1];
		const char * spaceDiv = " ";
		const char * fileOutputRGB = "imgs_out/image.rgb";
		const char *pngStrings[4] = {"convert ", fileInputName, spaceDiv, fileOutputRGB};
		const char * strPngToRGB = arrayStringsToString(pngStrings, 4, STRING_BUFFER_SIZE);

		printf("Loading input image [%s] \n", fileInputName);

		gettimeofday(&comp_end_load_img, NULL);

		struct timeval i_o_start_load_img, i_o_end_load_img;
		gettimeofday(&i_o_start_load_img, NULL);
		//execute the conversion from PNG to RGB, as that format is required for the program
		int status_conversion = system(strPngToRGB);
		gettimeofday(&i_o_end_load_img, NULL);

		struct timeval comp_start_img_conv, comp_end_img_conv;

		gettimeofday(&comp_start_img_conv, NULL);
		if(status_conversion != 0)
		{
			printf("ERROR! Conversion of input PNG image to RGB was not successful. Program aborting.\n");
			return -1;
		}
		printf("Converted input image to RGB [%s] \n", fileOutputRGB);

		//get the height and width of the input image
		int width = 0;
		int height = 0;

		getImageSize(fileInputName, &width, &height);

		printf("Size of the loaded image: width=%d height=%d \n", width, height);

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
	    byte * dev_gray_image;

		gettimeofday(&comp_end_img_conv, NULL);
		struct timeval start_alloc_rgb, end_alloc_rgb;
		gettimeofday(&start_alloc_rgb, NULL);

	    HANDLE_ERROR ( cudaMalloc((void **)&dev_r_vec , gray_size*sizeof(byte)));
	    HANDLE_ERROR ( cudaMalloc((void **)&dev_g_vec, gray_size*sizeof(byte)));
	    HANDLE_ERROR ( cudaMalloc((void **)&dev_b_vec, gray_size*sizeof(byte)));

	    //copy the content of the r,g,b vectors from the host to the device
	    HANDLE_ERROR (cudaMemcpy (dev_r_vec , rVector , gray_size*sizeof(byte), cudaMemcpyHostToDevice));
	    HANDLE_ERROR (cudaMemcpy (dev_g_vec , gVector , gray_size*sizeof(byte), cudaMemcpyHostToDevice));
	    HANDLE_ERROR (cudaMemcpy (dev_b_vec , bVector , gray_size*sizeof(byte), cudaMemcpyHostToDevice));
	    //allocate memory on the device for the output gray image
	    HANDLE_ERROR ( cudaMalloc((void **)&dev_gray_image, gray_size*sizeof(byte)));

		gettimeofday(&end_alloc_rgb, NULL);

		struct timeval comp_start_rgb_to_gray, comp_end_rgb_to_gray;
		gettimeofday(&comp_start_rgb_to_gray, NULL);

	    //actually run the kernel to convert input RGB file to gray-scale
	    rgb_img_to_gray <<< width, height>>> (dev_r_vec, dev_g_vec, dev_b_vec, dev_gray_image, gray_size) ;

	    //TODO: use malloc instead of [gray_size]
	    byte gray_image[gray_size];

		gettimeofday(&comp_end_rgb_to_gray, NULL);

		struct timeval start_gray_vec_copy, end_gray_vec_copy;

		gettimeofday(&start_gray_vec_copy, NULL);
	    //Now take the device gray vector and bring it back to the host
	    HANDLE_ERROR (cudaMemcpy(gray_image , dev_gray_image , gray_size*sizeof(byte) , cudaMemcpyDeviceToHost));
		gettimeofday(&end_gray_vec_copy, NULL);

		struct timeval comp_start_str_alloc, comp_end_str_alloc;

		gettimeofday(&comp_start_str_alloc, NULL);
		char str_width[100];
		sprintf(str_width, "%d", width);

		char str_height[100];
		sprintf(str_height, "%d", height);

		gettimeofday(&comp_end_str_alloc, NULL);

		if(intermediate_output)
		{
			 //let's see what's in there, shall we?
			const char * file_gray = "imgs_out/img_gray.gray";
			writeFile(file_gray, gray_image, gray_size);
			printf("Total amount of pixels in gray-scale image is [%d] \n", gray_size);
			const char * file_png_gray = "imgs_out/img_gray.png";

			const char * pngConvertGray[8] = {"convert -size ", str_width, "x", str_height, " -depth 8 ", file_gray, spaceDiv, file_png_gray};
			char * strGrayToPNG = arrayStringsToString(pngConvertGray, 8, STRING_BUFFER_SIZE);
			system(strGrayToPNG);
			printf("Converted gray image to PNG [%s]\n", file_png_gray);
		}


		struct timeval start_free_rgb, end_free_rgb;

		gettimeofday(&start_free_rgb, NULL);
	    cudaFree (dev_r_vec);
	    cudaFree (dev_g_vec);
		cudaFree (dev_b_vec);
		gettimeofday(&end_free_rgb, NULL);

		//######################3. Step - Compute vertical and horizontal gradient ##########

		//###Compute the HORIZONTAL GRADIENT#####

		struct timeval comp_start_alloc_h_vec, comp_end_alloc_h_vec;

		gettimeofday(&comp_start_alloc_h_vec, NULL);
   	    //host horizontal kernel
		int sobel_h[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
		int * dev_sobel_h;
   	    byte * dev_sobel_h_res;
		gettimeofday(&comp_end_alloc_h_vec, NULL);

		struct timeval start_h_vec_alloc, end_h_vec_alloc;

		gettimeofday(&start_h_vec_alloc, NULL);

		//allocate memory for device horizontal kernel
		HANDLE_ERROR ( cudaMalloc((void **)&dev_sobel_h , SOBEL_OP_SIZE*sizeof(int)));

		//copy the content of the host horizontal kernel to the device horizontal kernel
	    HANDLE_ERROR (cudaMemcpy (dev_sobel_h , sobel_h , SOBEL_OP_SIZE*sizeof(int) , cudaMemcpyHostToDevice));

	    //allocate memory for the resulting horizontal gradient on the device
		HANDLE_ERROR ( cudaMalloc((void **)&dev_sobel_h_res , gray_size*sizeof(byte)));

		gettimeofday(&end_h_vec_alloc, NULL);

		struct timeval comp_start_horiz_grad, comp_end_horiz_grad;

		gettimeofday(&comp_start_horiz_grad, NULL);
		//perform horizontal gradient calculation for every pixel
		it_conv <<< width, height>>> (dev_gray_image, gray_size, width, dev_sobel_h, dev_sobel_h_res);

		//fixed segmentation fault when processing large images
		byte* sobel_h_res = (byte*) malloc(gray_size * sizeof(byte));

		gettimeofday(&comp_end_horiz_grad, NULL);

		//copy the resulting horizontal array from device to host

		struct timeval start_h_vec_copy, end_h_vec_copy;

		gettimeofday(&start_h_vec_copy, NULL);
	    HANDLE_ERROR (cudaMemcpy(sobel_h_res , dev_sobel_h_res , gray_size*sizeof(byte) , cudaMemcpyDeviceToHost));
	    gettimeofday(&end_h_vec_copy, NULL);

		struct timeval start_h_vec_free, end_h_vec_free;

		gettimeofday(&start_h_vec_free, NULL);
	    //free-up the memory for the vectors allocated
	    cudaFree(dev_sobel_h);
	    gettimeofday(&end_h_vec_free, NULL);

	    const char * strGradToPNG;

	    if(intermediate_output)
	    {
			//output the horizontal axis-gradient to a file
			const char * file_out_h_grad = "imgs_out/sobel_horiz_grad.gray";
			//Output the horizontal axis' gradient calculation
			writeFile(file_out_h_grad, sobel_h_res, gray_size);
			printf("Output horizontal gradient to [%s] \n", file_out_h_grad);
			const char * fileHorGradPNG = "imgs_out/sobel_horiz_grad.png";
			printf("Converted horizontal gradient: ");
			printf("[%s] \n", fileHorGradPNG);
			//Convert the output file to PNG
			const char * pngConvertHor[8] = {"convert -size ", str_width, "x", str_height, " -depth 8 ", file_out_h_grad, spaceDiv, fileHorGradPNG};
			const char * strGradToPNG = arrayStringsToString(pngConvertHor, 8, STRING_BUFFER_SIZE);
			system(strGradToPNG);
	    }

		struct timeval comp_start_alloc_v_grad, comp_end_alloc_v_grad;
		gettimeofday(&comp_start_alloc_v_grad, NULL);
		//####Compute the VERTICAL GRADIENT#####
	    int sobel_v[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
		int * dev_sobel_v;
		byte * dev_sobel_v_res;
		gettimeofday(&comp_end_alloc_v_grad, NULL);

		struct timeval start_v_vec_alloc, end_v_vec_alloc;

		gettimeofday(&start_v_vec_alloc, NULL);

		//allocate memory for device vertical kernel
		HANDLE_ERROR (cudaMalloc((void **)&dev_sobel_v , SOBEL_OP_SIZE*sizeof(int)));

		//copy the content of the host vertical kernel to the device vertical kernel
		HANDLE_ERROR (cudaMemcpy (dev_sobel_v , sobel_v , SOBEL_OP_SIZE*sizeof(int) , cudaMemcpyHostToDevice));

		//allocate memory for the resulting vertical gradient on the device
		HANDLE_ERROR (cudaMalloc((void **)&dev_sobel_v_res , gray_size*sizeof(byte)));

		gettimeofday(&end_v_vec_alloc, NULL);

		struct timeval comp_start_vert_grad, comp_end_vert_grad;
		gettimeofday(&comp_start_vert_grad, NULL);

		//perform vertical gradient calculation for every pixel
		it_conv <<<width, height>>> (dev_gray_image, gray_size, width, dev_sobel_v, dev_sobel_v_res);

		//copy the resulting vertical array from device back to host
		//fixed segmentation fault issue with big images
		byte* sobel_v_res = (byte*) malloc(gray_size * sizeof(byte));

		gettimeofday(&comp_end_vert_grad, NULL);

		struct timeval start_v_vec_copy, end_v_vec_copy;

		gettimeofday(&start_v_vec_copy, NULL);
		HANDLE_ERROR (cudaMemcpy(sobel_v_res , dev_sobel_v_res , gray_size*sizeof(byte) , cudaMemcpyDeviceToHost));
		gettimeofday(&end_v_vec_copy, NULL);

		//free-up the memory for the vectors allocated
		struct timeval start_v_vec_free, end_v_vec_free;
		gettimeofday(&start_v_vec_free, NULL);
		cudaFree(dev_sobel_v);
		gettimeofday(&end_v_vec_free, NULL);

		struct timeval comp_start_countour_alloc, comp_end_countour_alloc;
		gettimeofday(&comp_start_countour_alloc, NULL);

		if(intermediate_output)
		{
			const char * file_out_v_grad = "imgs_out/sobel_vert_grad.gray";

			//Output the vertical axis' gradient calculated
			writeFile(file_out_v_grad, sobel_v_res, gray_size);

			printf("Output vertical gradient to [%s] \n", file_out_v_grad);
			const char * fileVerGradPNG = "imgs_out/sobel_vert_grad.png";

			const char * pngConvertVer[8] = {"convert -size ", str_width, "x", str_height, " -depth 8 ", file_out_v_grad, spaceDiv, fileVerGradPNG};

			strGradToPNG = arrayStringsToString(pngConvertVer, 8, STRING_BUFFER_SIZE);
			system(strGradToPNG);
		}


		//#############4. Step - Compute the countour by putting together the vertical and horizontal gradients####

		//allocate device memory for the final vector containing the countour

		byte * dev_countour_img;

		gettimeofday(&comp_end_countour_alloc, NULL);

		struct timeval start_countour_alloc, end_countour_alloc;

		gettimeofday(&start_countour_alloc, NULL);
		HANDLE_ERROR ( cudaMalloc((void **)&dev_countour_img , gray_size*sizeof(byte)));
		gettimeofday(&end_countour_alloc, NULL);

		struct timeval comp_start_countour_merge, comp_end_countour_merge;

		gettimeofday(&comp_start_countour_merge, NULL);
		contour <<< width, height>>> (dev_sobel_h_res, dev_sobel_v_res, gray_size, dev_countour_img);
		//copy the resulting countour image from device back to host
		byte countour_img[gray_size];
		gettimeofday(&comp_end_countour_merge, NULL);

		struct timeval start_countour_copy, end_countour_copy;

		gettimeofday(&start_countour_copy, NULL);
		HANDLE_ERROR (cudaMemcpy(countour_img, dev_countour_img, gray_size*sizeof(byte) , cudaMemcpyDeviceToHost));
		gettimeofday(&end_countour_copy, NULL);

		struct timeval start_free_countour, end_free_countour;

		gettimeofday(&start_free_countour, NULL);
		//free-up all the memory from the allocate vectors
	    cudaFree(dev_sobel_h_res);
	    cudaFree(dev_sobel_v_res);
	    cudaFree(dev_countour_img);
	    gettimeofday(&end_free_countour, NULL);

	    //######Display the resulting countour image

		struct timeval comp_start_countour_str_alloc, comp_end_countour_str_alloc;
		gettimeofday(&comp_start_countour_str_alloc, NULL);
		const char * file_sobel_out = "imgs_out/sobel_countour.gray";
		const char * file_sobel_png = "imgs_out/sobel_countour.png";
		const char * pngConvertContour[8] = {"convert -size ", str_width, "x", str_height, " -depth 8 ", file_sobel_out, spaceDiv, file_sobel_png};
		const char * strSobelToPNG = arrayStringsToString(pngConvertContour, 8, STRING_BUFFER_SIZE);
		gettimeofday(&comp_end_countour_str_alloc, NULL);


		struct timeval i_o_start_write_gray_countour, i_o_end_write_gray_countour;

		gettimeofday(&i_o_start_write_gray_countour, NULL);
		writeFile(file_sobel_out, countour_img, gray_size);
		gettimeofday(&i_o_end_write_gray_countour, NULL);

		printf("Output countour to [%s] \n", file_sobel_out);

		//actually execute the conversion from PNG to RGB, as that format is required for the program
		struct timeval i_o_start_write_img, i_o_end_write_img;
		gettimeofday(&i_o_start_write_img, NULL);
		system(strSobelToPNG);
		gettimeofday(&i_o_end_write_img, NULL);

		printf("Converted countour: [%s] \n", file_sobel_png);
		printf("SUCCESS! Successfully applied Sobel filter to the input image!\n");
		//printf("file loading and writing time: [%d] ms", write_load_total_time);


		//#############5. Step - Display the elapsed time in the different parts of the code

		//##GPU memory movements (cudaMalloc, cudaMemCpy, cudaFree) ##
		double time_alloc_rgb = compute_elapsed_time(start_alloc_rgb, end_alloc_rgb);
		double time_free_rgb = compute_elapsed_time(start_free_rgb, end_free_rgb);
		double time_copy_gray = compute_elapsed_time(start_gray_vec_copy, end_gray_vec_copy);
		//horizontal vector operations
		double time_alloc_h_vec = compute_elapsed_time(start_h_vec_alloc, end_h_vec_alloc);
		double time_copy_h_vec = compute_elapsed_time(start_h_vec_copy, end_h_vec_copy);
		double time_free_h_vec = compute_elapsed_time(start_h_vec_free, end_h_vec_free);
		//vertical vector operations
		double time_alloc_v_vec = compute_elapsed_time(start_v_vec_alloc, end_v_vec_alloc);
		double time_copy_v_vec = compute_elapsed_time(start_v_vec_copy, end_v_vec_copy);
		double time_free_v_vec = compute_elapsed_time(start_v_vec_free, end_v_vec_free);
		//countour image operations
		double time_alloc_countour = compute_elapsed_time(start_countour_alloc, end_countour_alloc);
		double time_copy_countour = compute_elapsed_time(start_countour_copy, end_countour_copy);
		double time_free_countour = compute_elapsed_time(start_free_countour, end_free_countour);

		double total_time_gpu_mem = time_alloc_rgb + time_free_rgb + time_copy_gray + time_alloc_h_vec + time_copy_h_vec + time_free_h_vec +
							  time_alloc_v_vec + time_copy_v_vec + time_free_v_vec + time_alloc_countour + time_copy_countour + time_free_countour;

		printf("Time spent on GPU memory operations: [%f] ms\n", total_time_gpu_mem);

		//##Actual GPU computation##
		double comp_time_load_img = compute_elapsed_time(comp_start_load_img, comp_end_load_img);
		double comp_time_convert_img = compute_elapsed_time(comp_start_img_conv, comp_end_img_conv);
		double comp_time_rgb_to_gray = compute_elapsed_time(comp_start_rgb_to_gray, comp_end_rgb_to_gray);
		double comp_time_str_alloc = compute_elapsed_time(comp_start_str_alloc, comp_end_str_alloc);
		double comp_time_h_alloc = compute_elapsed_time(comp_start_alloc_h_vec, comp_end_alloc_h_vec);
		double comp_time_h_grad = compute_elapsed_time(comp_start_horiz_grad, comp_end_horiz_grad);
		double comp_time_v_alloc = compute_elapsed_time(comp_start_alloc_v_grad, comp_end_alloc_v_grad);
		double comp_time_v_grad = compute_elapsed_time(comp_start_vert_grad, comp_end_vert_grad);
		double comp_time_count_alloc = compute_elapsed_time(comp_start_countour_alloc, comp_end_countour_alloc);
		double comp_time_count_merge = compute_elapsed_time(comp_start_countour_merge, comp_end_countour_merge);
		double comp_time_count_str_alloc = compute_elapsed_time(comp_start_countour_str_alloc, comp_end_countour_str_alloc);

		double total_time_gpu_comp = comp_time_load_img + comp_time_convert_img + comp_time_rgb_to_gray + comp_time_str_alloc + comp_time_h_alloc +
						comp_time_h_grad + comp_time_v_alloc + comp_time_v_grad + comp_time_count_alloc + comp_time_count_merge + comp_time_count_str_alloc;

		printf("Time spent on GPU computation: [%f] ms\n", total_time_gpu_comp);

		//##Input/Output over the disk (image loading and final image writing)##
		double i_o_time_load_img = compute_elapsed_time(i_o_start_load_img, i_o_end_load_img);
		double i_o_time_write_gray_countour = compute_elapsed_time(i_o_start_write_gray_countour, i_o_end_write_gray_countour);
		double i_o_time_write_img = compute_elapsed_time(i_o_start_write_img, i_o_end_write_img);

		double total_time_i_o = i_o_time_load_img + i_o_time_write_gray_countour + i_o_time_write_img;

		printf("Time spent on I/O operations from/to disk: [%f] ms\n", total_time_i_o);

		//##Overall time spent in the program
		double overall_total_time = total_time_gpu_comp + total_time_gpu_mem + total_time_i_o;

		printf("Overall time spent in program [%f] ms \n", overall_total_time);

	    return 0;

}

//Don't forget to clean up the device memory!!
