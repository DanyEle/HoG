
#ifndef FILE_OPERATIONS_H
#define FILE_OPERATIONS_H

typedef unsigned char byte;


void readFile(char *file_name, byte **buffer, int buffer_size);
void writeFile(char *file_name, byte *buffer, int buffer_size);
int getImageSize(const char *fn, int *x,int *y);
char * arrayStringsToString(char ** strings, int stringsAmount, int buffer_size);
double compute_elapsed_time(struct timeval time_begin, struct timeval time_end);


#endif
