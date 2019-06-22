
#ifndef FILE_OPERATIONS_H
#define FILE_OPERATIONS_H

typedef unsigned char byte;


void read_file(char *file_name, byte **buffer, int buffer_size);
void write_file(char *file_name, byte *buffer, int buffer_size);
int get_image_size(const char *fn, int *x,int *y);
char * array_strings_to_string(char ** strings, int stringsAmount, int buffer_size);
double compute_elapsed_time(struct timeval time_begin, struct timeval time_end);


#endif
