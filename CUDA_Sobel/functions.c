//Partial credits for this file go to: https://github.com/petermlm/SobelFilter
#include "functions.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>



typedef unsigned char byte;


//Input: file_name: the name of the file to be loaded
//  	 buffer_size: amount of RGB pixels to be loaded
//Output: buffer: the RGB pixels of the loaded image. It can be read as:
//buffer[0] = R-pixel
//buffer[1] = G-pixel
//buffer[2] = B-pixel
//buffer += 3 to get the next pixel

void read_file(const char *file_name, byte **buffer, int buffer_size)
{
    //Open file
    FILE *file = fopen(file_name, "r");

    // Allocate memory for buffer containing the file
    *buffer = (byte *) malloc(sizeof(byte) * buffer_size);

    // Read every char of file ONE BY ONE (not the whole thing at once)
    // We do this because this should look closer to the assembly implementation
    for(int i=0; i<buffer_size; i++)
    {
        (*buffer)[i] = fgetc(file);
    }

    // Close the opened file
    fclose(file);
}

/*Input: file_name: the name of the file onto which the buffer content needs to be written
 *		  buffer: an array containing RGB/gray-scale pixels to be written to a file
 *		  buffer_size: the size of the buffer to be written
 *Output: <none>. As a side effect, the content of 'buffer' is written to file_name
 */

void write_file(const char *file_name, byte *buffer, int buffer_size) //was * buffer
{
    // Open
    FILE *file = fopen(file_name, "w");

    // Write all
    for(int i=0; i<buffer_size; i++) {
        fputc(buffer[i], file);
    }

    // Close
    fclose(file);
}


//Input: char * fn: the name of the image file that should be loaded
//Output: 0 if successfully got image size and stored into *x, *y
//		  -1 if an error occurred (i.e.: most likely erroneous image format)
//Credits to: http://www.cplusplus.com/forum/beginner/45217/
int get_image_size(const char *fn, int *x,int *y)
{
    FILE *f=fopen(fn,"rb");
    if (f==0) return -1;
    fseek(f,0,SEEK_END);
    long len=ftell(f);
    fseek(f,0,SEEK_SET);
    if (len<24) {
        fclose(f);
        return -1;
        }
  //cout << fn << endl;
  // Strategy:
  // reading GIF dimensions requires the first 10 bytes of the file
  // reading PNG dimensions requires the first 24 bytes of the file
  // reading JPEG dimensions requires scanning through jpeg chunks
  // In all formats, the file is at least 24 bytes big, so we'll read that always
  unsigned char buf[24]; fread(buf,1,24,f);

  // For JPEGs, we need to read the first 12 bytes of each chunk.
  // We'll read those 12 bytes at buf+2...buf+14, i.e. overwriting the existing buf.
  if (buf[0]==0xFF && buf[1]==0xD8 && buf[2]==0xFF && buf[3]==0xE0 && buf[6]=='J' && buf[7]=='F' && buf[8]=='I' && buf[9]=='F')
  { long pos=2;
    while (buf[2]==0xFF)
    { if (buf[3]==0xC0 || buf[3]==0xC1 || buf[3]==0xC2 || buf[3]==0xC3 || buf[3]==0xC9 || buf[3]==0xCA || buf[3]==0xCB) break;
      pos += 2+(buf[4]<<8)+buf[5];
      if (pos+12>len) break;
      fseek(f,pos,SEEK_SET); fread(buf+2,1,12,f);
    }
  }

  fclose(f);

  // JPEG: (first two bytes of buf are first two bytes of the jpeg file; rest of buf is the DCT frame
  if (buf[0]==0xFF && buf[1]==0xD8 && buf[2]==0xFF)
  { *y = (buf[7]<<8) + buf[8];
    *x = (buf[9]<<8) + buf[10];
    //cout << *x << endl;
    return 0;
  }

  // GIF: first three bytes say "GIF", next three give version number. Then dimensions
  if (buf[0]=='G' && buf[1]=='I' && buf[2]=='F')
  { *x = buf[6] + (buf[7]<<8);
    *y = buf[8] + (buf[9]<<8);
    return 0;
  }

  // PNG: the first frame is by definition an IHDR frame, which gives dimensions
  if ( buf[0]==0x89 && buf[1]=='P' && buf[2]=='N' && buf[3]=='G' && buf[4]==0x0D && buf[5]==0x0A && buf[6]==0x1A && buf[7]==0x0A
    && buf[12]=='I' && buf[13]=='H' && buf[14]=='D' && buf[15]=='R')
  { *x = (buf[16]<<24) + (buf[17]<<16) + (buf[18]<<8) + (buf[19]<<0);
    *y = (buf[20]<<24) + (buf[21]<<16) + (buf[22]<<8) + (buf[23]<<0);
    return 0;
  }

  return -1;
}

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

//Input: dimension = 0 - R
//		 		   = 1 - G
//				   = 2 - B
//Output: dim_vector: the vector extracted from the RGB vector image corresponding to the dimension specified
void get_dimension_from_RGB_vec(int dimension, byte* rgbImage,  byte** dim_vector, int gray_size)
{
	// Take size for gray image and allocate memory. Just one dimension for gray-scale image
	* dim_vector = (byte*) malloc(sizeof(byte) * gray_size);

	// Make pointers for iteration
	byte *p_rgb = rgbImage;
	byte *p_gray = *dim_vector;

	// Calculate the value for every pixel in gray
	for(int i=0; i < gray_size; i++)
	{
		//Formula according to: https://stackoverflow.com/questions/17615963/standard-rgb-to-grayscale-conversion
		*p_gray = p_rgb[dimension];
		p_rgb += 3;
		p_gray++;
	}
}


//Input: time_begin: a struct storing the begin time
//		 time_end: a struct storing the end time
//		Output: the time elapsed as (time_end - time_begin)
double compute_elapsed_time(struct timeval time_begin, struct timeval time_end)
{
	//time in microseconds (us)
	double time_elapsed_us =  (double) (time_end.tv_usec - time_begin.tv_usec) / 1000000 +  (double) (time_end.tv_sec - time_begin.tv_sec);

	//return time in milliseconds (ms)
	double time_elapsed_ms = time_elapsed_us * 1000;

	return time_elapsed_ms;
}


//#####Functions used to output files to disk######



//Input: intermediate_output: true --> the content of gray_image is output to file_gray_name and then converted to png_file_name
//					          false --> the image is not output
//		 buffer_image: an array containing the bytes to be output to file_gray_name and converted to png_file_name
//		 buffer_size: the size of the 'buffer_image' output
//		 str_width: the width of the output image in string format (ex: "512")
//		 str_height: the height of the output image in string format (ex: "512")
//		 str_buffer_size: the amount of bytes to be allocated for producing the string supplied to the OS for conversion to PNG
//	     png_file_name: the
//Output: imgs_out/img_gray.png as an image containing the gray-scale input image
void output_gray_scale_image(bool intermediate_output, byte * gray_image, int gray_size, char * str_width, char * str_height, int string_buffer_size, const char * png_file_name)
{
	if(intermediate_output)
	{
		const char * file_gray = "imgs_out/img_gray.gray";
		write_file(file_gray, gray_image, gray_size);

		const char * PNG_convert_to_gray[8] = {"convert -size ", str_width, "x", str_height, " -depth 8 ", file_gray, " ", png_file_name};
		const char * str_gray_to_PNG = array_strings_to_string(PNG_convert_to_gray, 8, string_buffer_size);
		system(str_gray_to_PNG);

		//printf("Output gray-scale image [%s] \n", file_gray);
	}

}

//Used both for horizontal gradient and vertical gradient
//sobel_res = sobel_h_res or sobel_v_res
void output_gradient(bool intermediate_output, byte * sobel_res, int gray_size, char * str_width, char * str_height, int string_buffer_size, const char * png_file_name)
{
	  if(intermediate_output)
	  {
			//output the horizontal axis-gradient to an image file
	        const char * file_out_grad = "imgs_out/sobel_grad.gray";
			write_file(file_out_grad, sobel_res, gray_size);
			//Convert the output file to PNG
			const char * png_convert[8] = {"convert -size ", str_width, "x", str_height, " -depth 8 ", file_out_grad, " ", png_file_name};
			const char * str_grad_to_PNG = array_strings_to_string(png_convert, 8, string_buffer_size);
			system(str_grad_to_PNG);
			//printf("Output [%s] \n", png_file_name);
	   }
}


