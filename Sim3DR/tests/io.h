#ifndef IO_H_
#define IO_H_

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <stdio.h>

using namespace std;

#define MAX_PXL_VALUE 255

void load_obj(const char* obj_fp, float* vertices, float* colors, int* triangles, int nver, int ntri);
void load_ply(const char* ply_fp, float* vertices, int* triangles, int nver, int ntri);


void write_ppm(const char *filename, unsigned char *img, int h, int w, int c);

#endif