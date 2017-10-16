#pragma once

#include <cuda_runtime.h>

cudaError_t kernelMain(uchar4* pbo, int iter);
void kernelInit();
void kernelCleanUp();