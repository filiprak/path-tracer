#include "kernel.h"

#include "pathtracing.cuh"
#include "view.h"
#include "world.h"
#include "camera.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cudaUtility.h"
#include "utility.cuh"
#include "cutil_math.h"

#include <stdio.h>
#include <cstdlib>

float4* device_accum_image = NULL;

__host__
void kernelInit()
{
	int pixel_num = viewWidth * viewHeight;

	cudaMalloc(&device_accum_image, pixel_num * sizeof(float4));
	cudaMemset(device_accum_image, 0, pixel_num * sizeof(float4));

	initPathTracing();
}

__host__
void kernelCleanUp()
{
	cleanUpPathTracing();
	cudaFree(device_accum_image);
}



//Kernel that writes the image to the OpenGL PBO directly.
__global__
void writeImageToPBO(uchar4* pbo, int width, int height, int iter, float4* image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < width && y < height) {
		int index = x + (y * width);

		pbo[index].w = 0.0f;
		pbo[index].x = clamp((image[index].x) / (float)iter, 0.0f, 255.0f);
		pbo[index].y = clamp((image[index].y) / (float)iter, 0.0f, 255.0f);
		pbo[index].z = clamp((image[index].z) / (float)iter, 0.0f, 255.0f);
	}
}


// Helper function for using CUDA to add vectors in parallel.
__host__
cudaError_t kernelMain(uchar4* pbo, int iter)
{
	// Launch a kernel on the GPU with one thread for each element.
	const int blockSideLength = 24;
	const dim3 blockSize(blockSideLength, blockSideLength);
	const dim3 blocksPerGrid(
		(viewWidth + blockSize.x - 1) / blockSize.x,
		(viewHeight + blockSize.y - 1) / blockSize.y);

	if (iter == 1) {
		// clear image
		cudaMemset(device_accum_image, 0, viewWidth * viewHeight * sizeof(float4));
	}

	{ // pathtrace
		int iterHash = utilhash(iter);
		runPathTracing(iterHash);
		cudaDeviceSynchronize();
		checkCudaError("run runPathTracing()");
	}

	// write results to pbo
	writeImageToPBO <<<blocksPerGrid, blockSize>>>(pbo, viewWidth, viewHeight, iter, device_accum_image);
	checkCudaError("run sendImageToPBO<<<>>>()");
    
    cudaDeviceSynchronize();
	checkCudaError("kernelMain<<< >>>()");

    return cudaGetLastError();
}