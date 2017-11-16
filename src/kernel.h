#pragma once

#include <cuda_runtime.h>

extern float4* device_accum_image;

cudaError_t kernelMain(uchar4*, int);
void kernelInit();
void kernelCleanUp();