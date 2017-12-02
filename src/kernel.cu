#include "kernel.h"

#include "pathtracing.cuh"
#include "view.h"
#include "world.h"
#include "camera.h"
#include "device_launch_parameters.h"
#include "cudaUtility.h"
#include "cutility.cuh"
#include "cutil_math.h"

#include <stdio.h>
#include <cstdlib>

float4* device_accum_image = NULL;

__host__
void kernelInit()
{
	cudaMalloc(&device_accum_image, scene.camera.projection.num_pixels * sizeof(float4));
	cudaMemset(device_accum_image, 0, scene.camera.projection.num_pixels * sizeof(float4));

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
void writeImageToPBO(uchar4* pbo, float gamma, int width, int height, int iter, float4* image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < width && y < height) {
		int index = x + (y * width);

		float inv_iter = 1 / (float)iter;
		pbo[index].w = 0.0f;
		pbo[index].x = 255.0f * powf(clamp((image[index].x) * inv_iter, 0.0f, 1.0f), gamma);
		pbo[index].y = 255.0f * powf(clamp((image[index].y) * inv_iter, 0.0f, 1.0f), gamma);
		pbo[index].z = 255.0f * powf(clamp((image[index].z) * inv_iter, 0.0f, 1.0f), gamma);
	}
}


// Helper function for using CUDA to add vectors in parallel.
__host__
cudaError_t kernelMain(uchar4* pbo, int iter)
{
	Camera& cam = scene.camera;
	// Launch a kernel on the GPU with one thread for each element.
	const int blockSideLength = 24;
	const dim3 blockSize(blockSideLength, blockSideLength);
	const dim3 blocksPerGrid(
		(cam.projection.width + blockSize.x - 1) / blockSize.x,
		(cam.projection.height + blockSize.y - 1) / blockSize.y);

	if (iter == 1) {
		// clear image
		cudaMemset(device_accum_image, 0, cam.projection.num_pixels * sizeof(float4));
	}

	{ // pathtrace
		int iterHash = wang_hash(utilhash(iter));
		int jitterHash = wang_hash(iter);

		runPathTracing(iterHash, jitterHash);
		cudaDeviceSynchronize();
		checkCudaError("run runPathTracing()");
	}

	// write results to pbo
	writeImageToPBO << <blocksPerGrid, blockSize >> >(	pbo,
														cam.projection.gamma_corr,
														cam.projection.width,
														cam.projection.height,
														iter,
														device_accum_image);
	checkCudaError("run sendImageToPBO<<<>>>()");
    
    cudaDeviceSynchronize();
	checkCudaError("kernelMain<<< >>>()");

    return cudaGetLastError();
}