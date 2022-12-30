#pragma once

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#define max(x, y) (((x) > (y)) ? (x) : (y))
#define min(x, y) (((x) < (y)) ? (x) : (y))
#define clip(_x, _min, _max) min(max(_x, _min), _max)

#define RENDERLIBRARY_API __declspec(dllexport)

extern "C" RENDERLIBRARY_API void _render(const int* triangles,
  const int ntri,
  const float* light,
  const float* directional,
  const float* ambient,
  const float* vertices,
  const int nver,
  unsigned char* image,
  const int h, const int w);
