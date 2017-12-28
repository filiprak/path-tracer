#pragma once

#include <cuda_runtime.h>
#include "world.h"

extern cudaGraphicsResource* viewPBO_cuda;
extern uchar4 *pbo_dptr;
extern float4* device_accum_image;


cudaError_t kernelMain(uchar4*, Scene&, int);
void kernelInit(const Scene&);
void kernelCleanUp();
void runCUDA(uchar4*, Scene&, int);