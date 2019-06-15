#include <stdio.h>

#include "file_operations.c"

#define STRING_BUFFER_SIZE 1024




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


//Input: - rgb image contained in the 'rgb' array
//		 - buffer size: the size of the RGB image
//Output: gray, an array containing the gray-scale image
int rgbToGray(byte *rgbImage, byte **grayImage, int gray_size)
{
    // Take size for gray image and allocate memory. Just one dimension for gray-scale image
    *grayImage = (byte*) malloc(sizeof(byte) * gray_size);

    // Make pointers for iteration
    byte *p_rgb = rgbImage;
    byte *p_gray = *grayImage;

    // Calculate the value for every pixel in gray
    for(int i=0; i<gray_size; i++)
    {
    	//Formula according to: https://stackoverflow.com/questions/17615963/standard-rgb-to-grayscale-conversion
        *p_gray = 0.30*p_rgb[0] + 0.59*p_rgb[1] + 0.11*p_rgb[2];
        p_rgb += 3;
        p_gray++;
    }

    return gray_size;
}




// CUDA //kernel to convert an image to gray-scale
//gray-image's memory needs to be pre-allocated
__global__ void rgb_img_to_gray( byte * dev_r_vec, byte * dev_g_vec, byte * dev_b_vec, byte * dev_gray_image, int gray_size)
{
    //Get the id of thread within a block
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

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

	    //actually run the kernel
	    rgb_img_to_gray <<< 512, 512>>> (dev_r_vec, dev_g_vec, dev_b_vec, dev_gray_image, gray_size) ;
	    //__global__ void rgb_img_to_gray( byte * dev_r_vec, byte * dev_g_vec, byte * dev_b_vec, byte * dev_gray_image, int gray_size)


	    //run the rgb_to_gray kernel
	    //rgb_to_gray <<< height * 3, width*3>>> (dev_rgb_image , &dev_gray_image , gray_size) ;

}
