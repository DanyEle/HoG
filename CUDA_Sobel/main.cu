//============================================================================
// Name        : main.cu
// Author      : Daniele Gadler
// Version     :
// Description : Sobel operator in CUDA
//============================================================================


#include <stdio.h>


#include "functions.c"


//false --> No vertical gradient and horizontal gradient are output
//true --> Vertical gradient and horizontal gradient are output
#define INTERMEDIATE_OUTPUT false
#define SOBEL_OP_SIZE 9
#define STRING_BUFFER_SIZE 1024

#define HANDLE_ERROR( err ) ( HandleError( err, __FILE__, __LINE__ ) )
#define get_time(time) (gettimeofday(&time, NULL))


#include "string.h"
#include "stdlib.h"
#include "math.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>

#include "kernels.cu"


static void HandleError( cudaError_t err, const char *file, int line )
{
    if (err != cudaSuccess)
      {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE);
    }
}



int main ( int argc, char** argv )
{
		//all the time_val declarations are put at the beginning of the file for better code readability
		struct timeval comp_start_load_img, comp_end_load_img;
		struct timeval i_o_start_load_img, i_o_end_load_img;
		struct timeval comp_start_img_conv, comp_end_img_conv;
		struct timeval start_alloc_rgb, end_alloc_rgb;
		struct timeval start_first_cuda_malloc, end_first_cuda_malloc;
		struct timeval comp_start_rgb_to_gray, comp_end_rgb_to_gray;
		struct timeval start_gray_vec_copy, end_gray_vec_copy;
		struct timeval comp_start_str_alloc, comp_end_str_alloc;
		struct timeval start_free_rgb, end_free_rgb;
		struct timeval comp_start_alloc_h_vec, comp_end_alloc_h_vec;
		struct timeval start_h_vec_alloc, end_h_vec_alloc;
		struct timeval comp_start_horiz_grad, comp_end_horiz_grad;
		struct timeval start_h_vec_copy, end_h_vec_copy;
		struct timeval start_h_vec_free, end_h_vec_free;
		struct timeval comp_start_alloc_v_grad, comp_end_alloc_v_grad;
		struct timeval start_v_vec_alloc, end_v_vec_alloc;
		struct timeval comp_start_vert_grad, comp_end_vert_grad;
		struct timeval start_v_vec_copy, end_v_vec_copy;
		struct timeval start_countour_alloc, end_countour_alloc;
		struct timeval start_countour_copy, end_countour_copy;
		struct timeval start_free_countour, end_free_countour;
		struct timeval i_o_start_write_img, i_o_end_write_img;

		//dummy CUDA malloc to "waste" time just at the beginning of the program and not in the middle of the computation
		byte * dummy_array;
		get_time(start_first_cuda_malloc);
		HANDLE_ERROR ( cudaMalloc((void **)&dummy_array , 1*sizeof(byte)));
	    get_time(end_first_cuda_malloc);

		//actual computation begins
		get_time(comp_start_load_img);
		if(argc < 2)
		{
			printf("You did not provide any input image name. Please, provide an input image name and retry. \n");
			return -2;
		}

		//###########1. STEP - LOAD THE IMAGE, ITS HEIGHT, WIDTH AND CONVERT IT TO RGB FORMAT#########

		//Specify the input image. Formats supported: png, jpg, GIF.
		const char * file_output_rgb = "imgs_out/image.rgb";
		const char *png_strings[4] = {"convert ", argv[1], " ", file_output_rgb};
		const char * str_PNG_to_RGB = array_strings_to_string(png_strings, 4, STRING_BUFFER_SIZE);

		//printf("Loading input image [%s] \n", fileInputName); //debug

		get_time(comp_end_load_img);

		get_time(i_o_start_load_img);
		//execute the conversion from PNG to RGB, as that format is required by the program
		int status_conversion = system(str_PNG_to_RGB);
		get_time(i_o_end_load_img);

		get_time(comp_start_img_conv);
		if(status_conversion != 0)
		{
			printf("ERROR! Conversion of input PNG image to RGB was not successful. Program aborting.\n");
			return -1;
		}
		//get the height and width of the input image
		int width = 0;
		int height = 0;

		get_image_size(argv[1], &width, &height);

		//Three dimensions because the input image is in RGB format
		int rgb_size = width * height * 3;

		//Used as a buffer for all pixels of the image
		byte * rgb_image;

		//Load up the input image in RGB format into one single flattened array (rgbImage)
		read_file(file_output_rgb, &rgb_image, rgb_size);

		//########2. step - convert RGB image to gray-scale
	    int gray_size = rgb_size / 3;
	    byte * r_vector, * g_vector, * b_vector;

	    //now take the RGB image vector and create three separate arrays for the R,G,B dimensions
	    get_dimension_from_RGB_vec(0, rgb_image,  &r_vector, gray_size);
	    get_dimension_from_RGB_vec(1, rgb_image,  &g_vector, gray_size);
	    get_dimension_from_RGB_vec(2, rgb_image,  &b_vector, gray_size);

	    //allocate memory on the device for the r,g,b vectors
	    byte * dev_r_vec, * dev_g_vec, * dev_b_vec;
	    byte * dev_gray_image;

		get_time(comp_end_img_conv);
		get_time(start_alloc_rgb);

	    HANDLE_ERROR ( cudaMalloc((void **)&dev_r_vec, gray_size*sizeof(byte)));
	    HANDLE_ERROR ( cudaMalloc((void **)&dev_g_vec, gray_size*sizeof(byte)));
	    HANDLE_ERROR ( cudaMalloc((void **)&dev_b_vec, gray_size*sizeof(byte)));

	    //copy the content of the r,g,b vectors from the host to the device
	    HANDLE_ERROR (cudaMemcpy (dev_r_vec , r_vector , gray_size*sizeof(byte), cudaMemcpyHostToDevice));
	    HANDLE_ERROR (cudaMemcpy (dev_g_vec , g_vector , gray_size*sizeof(byte), cudaMemcpyHostToDevice));
	    HANDLE_ERROR (cudaMemcpy (dev_b_vec , b_vector, gray_size*sizeof(byte), cudaMemcpyHostToDevice));
	    //allocate memory on the device for the output gray image
	    HANDLE_ERROR ( cudaMalloc((void **)&dev_gray_image, gray_size*sizeof(byte)));

		get_time(end_alloc_rgb);

		get_time(comp_start_rgb_to_gray);

	    //actually run the kernel to convert input RGB file to gray-scale
	    rgb_img_to_gray <<< width, height>>> (dev_r_vec, dev_g_vec, dev_b_vec, dev_gray_image, gray_size) ;
	    cudaDeviceSynchronize();

		byte * gray_image = (byte *) malloc(gray_size * sizeof(byte));

		get_time(comp_end_rgb_to_gray);

		get_time(start_gray_vec_copy);
	    //Now take the device gray vector and bring it back to the host
	    HANDLE_ERROR (cudaMemcpy(gray_image , dev_gray_image , gray_size*sizeof(byte) , cudaMemcpyDeviceToHost));
		get_time(end_gray_vec_copy);

		get_time(comp_start_str_alloc);
		char str_width[100];
		sprintf(str_width, "%d", width);

		char str_height[100];
		sprintf(str_height, "%d", height);

		get_time(comp_end_str_alloc);

		//output the gray-scale image to a PNG file if INTERMEDIATE_OUTPUT == true
		output_gray_scale_image(INTERMEDIATE_OUTPUT, gray_image, gray_size, str_width, str_height, STRING_BUFFER_SIZE, "imgs_out/img_gray.png");

		get_time(start_free_rgb);
	    cudaFree (dev_r_vec);
	    cudaFree (dev_g_vec);
		cudaFree (dev_b_vec);
		get_time(end_free_rgb);

		//######################3. Step - Compute vertical and horizontal gradient ##########

		//###Compute the HORIZONTAL GRADIENT#####

		get_time(comp_start_alloc_h_vec);
   	    //host horizontal kernel
		int sobel_h[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
		int * dev_sobel_h;
   	    byte * dev_sobel_h_res;
		get_time(comp_end_alloc_h_vec);

		get_time(start_h_vec_alloc);

		//allocate memory for device horizontal kernel
		HANDLE_ERROR ( cudaMalloc((void **)&dev_sobel_h , SOBEL_OP_SIZE*sizeof(int)));

		//copy the content of the host horizontal kernel to the device horizontal kernel
	    HANDLE_ERROR (cudaMemcpy (dev_sobel_h , sobel_h , SOBEL_OP_SIZE*sizeof(int) , cudaMemcpyHostToDevice));

	    //allocate memory for the resulting horizontal gradient on the device
		HANDLE_ERROR ( cudaMalloc((void **)&dev_sobel_h_res , gray_size*sizeof(byte)));

		get_time(end_h_vec_alloc);

		get_time(comp_start_horiz_grad);
		//perform horizontal gradient calculation for every pixel
		it_conv <<< width, height>>> (dev_gray_image, gray_size, width, dev_sobel_h, dev_sobel_h_res);
	    cudaDeviceSynchronize();

		//fixed segmentation fault when processing large images by using a malloc
		byte* sobel_h_res = (byte*) malloc(gray_size * sizeof(byte));
		get_time(comp_end_horiz_grad);
		//copy the resulting horizontal array from device to host

		get_time(start_h_vec_copy);
	    HANDLE_ERROR (cudaMemcpy(sobel_h_res , dev_sobel_h_res , gray_size*sizeof(byte) , cudaMemcpyDeviceToHost));
	    get_time(end_h_vec_copy);


		get_time(start_h_vec_free);
	    //free-up the memory for the vectors allocated
	    cudaFree(dev_sobel_h);
	    get_time(end_h_vec_free);

	    //output the horizontal gradient to a file if INTERMEDIATE_OUTPUT == true
	    output_gradient(INTERMEDIATE_OUTPUT, sobel_h_res, gray_size, str_width, str_height, STRING_BUFFER_SIZE, "imgs_out/sobel_horiz_grad.png");

		get_time(comp_start_alloc_v_grad);
		//####Compute the VERTICAL GRADIENT#####
	    int sobel_v[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
		int * dev_sobel_v;
		byte * dev_sobel_v_res;
		get_time(comp_end_alloc_v_grad);

		get_time(start_v_vec_alloc);

		//allocate memory for device vertical kernel
		HANDLE_ERROR (cudaMalloc((void **)&dev_sobel_v , SOBEL_OP_SIZE*sizeof(int)));

		//copy the content of the host vertical kernel to the device vertical kernel
		HANDLE_ERROR (cudaMemcpy (dev_sobel_v , sobel_v , SOBEL_OP_SIZE*sizeof(int) , cudaMemcpyHostToDevice));

		//allocate memory for the resulting vertical gradient on the device
		HANDLE_ERROR (cudaMalloc((void **)&dev_sobel_v_res , gray_size*sizeof(byte)));

		get_time(end_v_vec_alloc);

		get_time(comp_start_vert_grad);

		//perform vertical gradient calculation for every pixel
		it_conv <<<width, height>>> (dev_gray_image, gray_size, width, dev_sobel_v, dev_sobel_v_res);
	    cudaDeviceSynchronize();

		//copy the resulting vertical array from device back to host
		//fixed segmentation fault issue with big images
		byte* sobel_v_res = (byte*) malloc(gray_size * sizeof(byte));

		get_time(comp_end_vert_grad);


		get_time(start_v_vec_copy);
		HANDLE_ERROR (cudaMemcpy(sobel_v_res , dev_sobel_v_res , gray_size*sizeof(byte) , cudaMemcpyDeviceToHost));
		get_time(end_v_vec_copy);

		//free-up the memory for the vectors allocated
		struct timeval start_v_vec_free, end_v_vec_free;
		get_time(start_v_vec_free);
		cudaFree(dev_sobel_v);
		get_time(end_v_vec_free);

		struct timeval comp_start_countour_alloc, comp_end_countour_alloc;
		get_time(comp_start_countour_alloc);

	    output_gradient(INTERMEDIATE_OUTPUT, sobel_v_res, gray_size, str_width, str_height, STRING_BUFFER_SIZE, "imgs_out/sobel_vert_grad.png");

		//#############4. Step - Compute the countour by putting together the vertical and horizontal gradients####
		//allocate device memory for the final vector containing the countour

		byte * dev_countour_img;
		get_time(comp_end_countour_alloc);


		get_time(start_countour_alloc);
		HANDLE_ERROR ( cudaMalloc((void **)&dev_countour_img , gray_size*sizeof(byte)));
		get_time(end_countour_alloc);

		struct timeval comp_start_countour_merge, comp_end_countour_merge;

		get_time(comp_start_countour_merge);
		contour <<< width, height>>> (dev_sobel_h_res, dev_sobel_v_res, gray_size, dev_countour_img);
	    cudaDeviceSynchronize();

		//copy the resulting countour image from the device back to host
		byte * countour_img = (byte *) malloc(gray_size * sizeof(byte));

		get_time(comp_end_countour_merge);

		get_time(start_countour_copy);
		HANDLE_ERROR (cudaMemcpy(countour_img, dev_countour_img, gray_size*sizeof(byte) , cudaMemcpyDeviceToHost));
		get_time(end_countour_copy);

		get_time(start_free_countour);
		//free-up all the memory from the allocated vectors
	    cudaFree(dev_sobel_h_res);
	    cudaFree(dev_sobel_v_res);
	    cudaFree(dev_countour_img);
	    get_time(end_free_countour);

	    //######Display the resulting countour image

		get_time(i_o_start_write_img);
	    output_gradient(true, countour_img, gray_size, str_width, str_height, STRING_BUFFER_SIZE, "imgs_out/sobel_countour.png");
		get_time(i_o_end_write_img);


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

		printf("Time spent on GPU memory operations: [%f] ms\n", total_time_gpu_mem); //debug
		//printf("%f \n", total_time_gpu_mem);


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

		double total_time_gpu_comp = comp_time_load_img + comp_time_convert_img + comp_time_rgb_to_gray + comp_time_str_alloc + comp_time_h_alloc +
						comp_time_h_grad + comp_time_v_alloc + comp_time_v_grad + comp_time_count_alloc + comp_time_count_merge ;

		printf("Time spent on GPU computation: [%f] ms\n", total_time_gpu_comp); //debug
		//printf("%f \n", total_time_gpu_comp);

		//##Input/Output over the disk (image loading and final image writing)##
		double i_o_time_load_img = compute_elapsed_time(i_o_start_load_img, i_o_end_load_img);
		double i_o_time_write_img = compute_elapsed_time(i_o_start_write_img, i_o_end_write_img);

		double total_time_i_o = i_o_time_load_img + i_o_time_write_img;

		printf("Time spent on I/O operations from/to disk: [%f] ms\n", total_time_i_o); //debug
		//printf("%f \n", total_time_i_o);

		//##Overall time spent in the program
		double overall_total_time = total_time_gpu_comp + total_time_gpu_mem + total_time_i_o;

		printf("Overall time spent in program [%f] ms \n", overall_total_time); //debug
		//printf("%f \n", overall_total_time);


		double time_first_cuda_malloc = compute_elapsed_time(start_first_cuda_malloc, end_first_cuda_malloc);

		//printf("First cuda malloc has taken [%f] ms\n", time_first_cuda_malloc);

		//let's deallocate the heap memory to avoid any memory leaks
		free(gray_image);
		free(sobel_h_res);
		free(sobel_v_res);
		free(countour_img);

	    return 0;

}
