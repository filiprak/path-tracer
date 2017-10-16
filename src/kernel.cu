
#include "view.h"
#include "world.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cudaUtility.h"

#include <stdio.h>
#include <cstdlib>
#include <thrust/random.h>
#include <glm\glm.hpp>

// temp image memory
static glm::vec3* dev_image = NULL;
static Ray* prim_rays = NULL;

__host__
void kernelCleanUp()
{
	cudaFree(dev_image);
	cudaFree(prim_rays);
}

__host__
void kernelInit()
{
	int pixel_num = viewWidth * viewHeight;

	cudaMalloc(&dev_image, pixel_num * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixel_num * sizeof(glm::vec3));

	cudaMalloc(&prim_rays, pixel_num * sizeof(Ray));
	cudaMemset(prim_rays, 0, pixel_num * sizeof(Ray));
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
void writeImageToPBO(uchar4* pbo, int width, int height, int iter, glm::vec3* dev_image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < width && y < height) {
		int index = x + (y * width);

		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
		thrust::uniform_real_distribution<float> u01(0, 255.0);

		float r1 = u01(rng);
		float r2 = u01(rng);
		float r3 = u01(rng);

		dev_image[index] += glm::vec3(r1, r2, r3);

		// Each thread writes one pixel location in the texture (textel)
		pbo[index].w = 0;
		pbo[index].x = r1;
		pbo[index].y = r2;
		pbo[index].z = r3;

		if (x % 20 == 0 || y % 20 == 0) {
			pbo[index].x = 0.0;
			pbo[index].y = 0.0;
			pbo[index].z = 0.0;
		}
	}
}

// Init primary rays
__global__
void kernelPrimaryRays(Camera cam, Ray* rays)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.projection.width && y < cam.projection.height) {
		int index = x + (y * cam.projection.width);

		float perPixel = 2.0f / (float)cam.projection.height;
		float halfScreenHeight = 1.0f;
		float halfScreenWidth = 1.0f * (float)cam.projection.width / (float)cam.projection.height;

		glm::vec3 screenDistanceVec = cam.projection.viewer_dist * glm::normalize(cam.direction);
		glm::vec3 pixVector = (halfScreenWidth - x * perPixel - perPixel / 2.0f) * glm::normalize(glm::cross(cam.up, cam.direction)) +
			(halfScreenHeight - y * perPixel - perPixel / 2.0f) * glm::normalize(cam.up);

		rays[index].direction = glm::normalize(screenDistanceVec + pixVector);
		if ((x == 0 && y == 0) || (x == 0 && y == 749) || (x == 999 && y == 0) || (x == 999 && y == 749)) {
			printf("\nRay(%d, %d) = [%.9f, %.9f, %.9f]\n  pixVector = [%f, %f, %f]\n\n",
				x, y, rays[index].direction.x, rays[index].direction.y, rays[index].direction.z,
				pixVector.x, pixVector.y, pixVector.z);
		}
		rays[index].originPoint = cam.position;
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

	if (iter == 0) {
		Camera cam = createCamera();

		kernelPrimaryRays << <blocksPerGrid, blockSize >> >(cam, prim_rays);
		checkCudaError("run kernelPrimaryRays<<< >>>()");
	}

	writeImageToPBO << <blocksPerGrid, blockSize >> >(pbo, viewWidth, viewHeight, iter, dev_image);
	checkCudaError("run sendImageToPBO<<< >>>()");
    
    cudaDeviceSynchronize();
	checkCudaError("kernelMain<<< >>>()");

    return cudaGetLastError();
}