# pragma once

#include "cuda.h"
#include "cuda_runtime.h"


__global__
void pbotest(uchar4* pbo, int width, int height);

__host__
cudaError_t pbotestRun(uchar4* pbo, int, int);