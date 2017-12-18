#include "device_launch_parameters.h"
#include "cutil_math.h"
#include "pbotest.h"
#include "cudaUtility.h"
#include "world.h"


__global__
void pbotest(uchar4* pbo, int width, int height) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < width && y < height) {
		int index = x + (y * width);

		float gradientx = (float)x / (float)width;
		float gradienty = (float)y / (float)height;

		pbo[index].w = 0.0f;
		pbo[index].x = 255.0f * gradientx;
		pbo[index].y = 0.0f;
		pbo[index].z = 255.0f * gradienty;
	}
}

__host__
cudaError_t pbotestRun(uchar4* pbo, int w, int h)
{
	const int blockSideLength = 8;
	const dim3 blockSize(blockSideLength, blockSideLength);
	const dim3 blocksPerGrid(
		(w + blockSize.x - 1) / blockSize.x,
		(h + blockSize.y - 1) / blockSize.y);

	pbotest<<<blocksPerGrid, blockSize>>>(pbo, w, h);
	checkCudaError("run pbotest<<<>>>()");
	cudaDeviceSynchronize();

	return cudaGetLastError();
}