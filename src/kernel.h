#pragma once

#include <cuda_runtime.h>
#include <glm\glm.hpp>

extern glm::vec3* dev_image;

cudaError_t kernelMain(uchar4* pbo, int iter);
void kernelInit();
void kernelCleanUp();