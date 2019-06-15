
#ifndef FILE_OPERATIONS_H
#define FILE_OPERATIONS_H

typedef unsigned char byte;


void readFile(const char *file_name, byte **buffer, int buffer_size);
void writeFile(const char *file_name, byte *buffer, int buffer_size);
int getImageSize(const char *fn, int *x,int *y);
char * arrayStringsToString(const char ** strings, int stringsAmount, int buffer_size);
void getDimensionFromRGBVec(int dimension, byte* rgbImage,  byte** dim_vector, int gray_size);

#endif
