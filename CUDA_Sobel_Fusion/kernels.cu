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




//Input: dev_gray_image: a vector containing the pixels of the gray image
//	     buffer_size: the amount of pixels in the dev_gray_image
//		 width: the width of the input image
//	     dev_sobel_h: the horizontal 3x3 kernel by which the image is convolved
//		 dev_sobel_v: the vertical 3x3 kernel by which the image is convolved
//Output: dev_contour_img. the resulting gradient
__global__ void map_fusion(byte * dev_gray_image, int buffer_size, int width, int * dev_sobel_h, int * dev_sobel_v, byte *dev_contour_img)
{
	//area over which the convolution will be performed
    byte op_mem[SOBEL_OP_SIZE];

    memset(op_mem, 0, SOBEL_OP_SIZE);

    int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
	int tid_y = threadIdx.y + blockIdx.y * blockDim.y;

	//simple linearization of the threads' space
	int tid = abs(tid_x - tid_y);

	byte h_res = 0;
	byte v_res = 0;

    // Make convolution for every pixel. Each pixel --> one thread.
    while(tid < buffer_size)
    {
    	//identify the region in the gray-scale image where the convolution is performed
        make_op_mem(dev_gray_image, buffer_size, width, tid, op_mem);
        make_op_mem(dev_gray_image, buffer_size, width, tid, op_mem);

        //horizontal convolution
        //h_res = (byte) abs(convolution(op_mem, dev_sobel_h, SOBEL_OP_SIZE));
        h_res = (byte) abs(convolution(op_mem, dev_sobel_h, SOBEL_OP_SIZE));

        //vertical convolution
        v_res = (byte) abs(convolution(op_mem, dev_sobel_v, SOBEL_OP_SIZE));

        dev_contour_img[tid] = (byte) sqrt(pow((double)h_res, 2.0) + pow((double)v_res, 2.0));

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
