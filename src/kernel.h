#pragma once

#include <cuda_runtime.h>
#include "world.h"

extern cudaGraphicsResource* viewPBO_cuda;
extern float4* device_accum_image;

cudaError_t kernelMain(uchar4*, Scene&, int);
void kernelInit();
void kernelCleanUp();