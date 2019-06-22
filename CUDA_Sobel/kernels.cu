//Input: dev_sobel_h: array containing the sobel horizontal gradient
//		 dev_sobel_v: array containing the sobel verticial gradient
//       gray_size: the amount of pixels in the gray-scale image
//Output: dev_contour_img: the resulting countour pixels of the final image, where the the hor. and ver.gradient
//have been put together
__global__ void contour(byte *dev_sobel_h, byte *dev_sobel_v, int gray_size, byte *dev_contour_img)
{
	int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
	int tid_y = threadIdx.y + blockIdx.y * blockDim.y;

	//Use the abs function as a linearization strategy to avoid having negative thread numbers
	int tid = abs(tid_x - tid_y);


    // Performed on every pixel in parallel to calculate the contour image
    while(tid < gray_size)
    {
    	//g = (g_x^2 + g_y^2)^0.5
        dev_contour_img[tid] = (byte) sqrt(pow((double)dev_sobel_h[tid], 2.0) + pow((double)dev_sobel_v[tid], 2.0));

    	tid += blockDim.x * gridDim.x + blockDim.y * gridDim.y;

    }
}



//Performs convolution within an input region
__device__ int convolution(byte *X, int *Y, int c_size)
{
    int sum = 0;

    for(int i=0; i < c_size; i++)
    {
        sum += X[i] * Y[c_size-i-1];
    }

    return sum;
}

//Input: dev_buffer: all the pixels in the input gray image
//	     buffer_size: the amount of pixels in the gray image
//		 width: the width of the input image
//	     cindex: the current index of the pixel being considered
//Output: op_mem. The current 3x3 region of pixels being considered around cindex
__device__ void make_op_mem(byte *dev_buffer, int buffer_size, int width, int cindex, byte *op_mem)
{
    int bottom = cindex-width < 0;
    int top = cindex+width >= buffer_size;
    int left = cindex % width == 0;
    int right = (cindex+1) % width == 0;

    op_mem[0] = !bottom && !left  ? dev_buffer[cindex-width-1] : 0;
    op_mem[1] = !bottom           ? dev_buffer[cindex-width]   : 0;
    op_mem[2] = !bottom && !right ? dev_buffer[cindex-width+1] : 0;

    op_mem[3] = !left             ? dev_buffer[cindex-1]       : 0;
    op_mem[4] = dev_buffer[cindex];
    op_mem[5] = !right            ? dev_buffer[cindex+1]       : 0;

    op_mem[6] = !top && !left     ? dev_buffer[cindex+width-1] : 0;
    op_mem[7] = !top              ? dev_buffer[cindex+width]   : 0;
    op_mem[8] = !top && !right    ? dev_buffer[cindex+width+1] : 0;
}


//Input: dev_buffer: the gray-scale input image
//		 buffer_size: the amount of gray pixels in the input image
//	     width: the width of the input image
//		 dev_op: the 3x3 kernel used to convolve the gray-scale image
//Output: dev_res: the resulting horizontal/vertical gradient
__global__ void it_conv(byte * dev_buffer, int buffer_size, int width, int * dev_op, byte *dev_res)
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
    	//identify the region in the gray-scale image where the convolution is performed
        make_op_mem(dev_buffer, buffer_size, width, tid, op_mem);

        //actually carry out the convolution
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
		//r, g, b pixels in the input image
		byte p_r = dev_r_vec[tid];
		byte p_g = dev_g_vec[tid];
		byte p_b = dev_b_vec[tid];

		//Formula accordidev_ng to: https://stackoverflow.com/questions/17615963/standard-rgb-to-grayscale-conversion
		dev_gray_image[tid] = 0.30 * p_r + 0.59*p_g + 0.11*p_b;

    	tid += blockDim.x * gridDim.x + blockDim.y * gridDim.y;

	}
}
