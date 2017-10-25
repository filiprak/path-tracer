#include "kernel.h"

#include "pathtracing.h"
#include "view.h"
#include "world.h"
#include "camera.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cudaUtility.h"

#include <stdio.h>
#include <cstdlib>
#include <thrust/random.h>

float3* dev_image = NULL;

__host__
void kernelInit()
{
	int pixel_num = viewWidth * viewHeight;

	cudaMalloc(&dev_image, pixel_num * sizeof(float3));
	cudaMemset(dev_image, 0, pixel_num * sizeof(float3));

	initPathTracing();
}

__host__
void kernelCleanUp()
{
	cleanUpPathTracing();

	cudaFree(dev_image);
}

__host__ __device__
inline unsigned int utilhash(unsigned int a) {
	a = (a + 0x7ed55d16) + (a << 12);
	a = (a ^ 0xc761c23c) ^ (a >> 19);
	a = (a + 0x165667b1) + (a << 5);
	a = (a + 0xd3a2646c) ^ (a << 9);
	a = (a + 0xfd7046c5) + (a << 3);
	a = (a ^ 0xb55a4f09) ^ (a >> 16);
	return a;
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
	int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
	return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__
void writeImageToPBO(uchar4* pbo, int width, int height, int iter, float3* dev_image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < width && y < height) {
		int index = x + (y * width);

		/*thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
		thrust::uniform_real_distribution<float> u01(0, 255.0);

		float r1 = u01(rng);
		float r2 = u01(rng);
		float r3 = u01(rng);*/

		pbo[index].w = 0;
		pbo[index].x = dev_image[index].x;
		pbo[index].y = dev_image[index].y;
		pbo[index].z = dev_image[index].z;
	}
}



// Helper function for using CUDA to add vectors in parallel.
__host__
cudaError_t kernelMain(uchar4* pbo, int iter)
{
	// Launch a kernel on the GPU with one thread for each element.
	const int blockSideLength = 8;
	const dim3 blockSize(blockSideLength, blockSideLength);
	const dim3 blocksPerGrid(
		(viewWidth + blockSize.x - 1) / blockSize.x,
		(viewHeight + blockSize.y - 1) / blockSize.y);

	{ // pathtrace
		runPathTracing();
		cudaDeviceSynchronize();
		checkCudaError("run runPathTracing()");
	}

	writeImageToPBO <<<blocksPerGrid, blockSize>>>(pbo, viewWidth, viewHeight, iter, dev_image);
	checkCudaError("run sendImageToPBO<<<>>>()");
    
    cudaDeviceSynchronize();
	checkCudaError("kernelMain<<< >>>()");

    return cudaGetLastError();
}