#include "pathtracing.h"
#include "kernel.h"
#include "stdio.h"
#include "view.h"
#include "intersect.h"
#include "cudaUtility.h"
#include "world_load.h"
#include "cutil_math.h"

// memory
static Ray* prim_rays = NULL;

// func decl
__host__
void loadWorldObjects(Camera cam, WorldObject* wobjects);

__global__
void generatePrimaryRays(Camera cam, Ray* rays);


// Init pathtracing
__host__
void initPathTracing() {
	loadSceneWorldObjects();

	int pixel_num = viewWidth * viewHeight;

	cudaOk(cudaMalloc(&prim_rays, pixel_num * sizeof(Ray)));
	cudaOk(cudaMemset(prim_rays, 0, pixel_num * sizeof(Ray)));

	const int blockSideLength = 8;
	const dim3 blockSize(blockSideLength, blockSideLength);
	const dim3 blocksPerGrid(
		(viewWidth + blockSize.x - 1) / blockSize.x,
		(viewHeight + blockSize.y - 1) / blockSize.y);

	generatePrimaryRays << <blocksPerGrid, blockSize >> >(scene.camera, prim_rays);
	cudaDeviceSynchronize();
	checkCudaError("generatePrimaryRays<<<>>>()");
}

__host__
void cleanUpPathTracing()
{
	cudaOk(cudaFree(prim_rays));
	freeWorldObjects();
}


// Init primary rays
__global__
void generatePrimaryRays(Camera cam, Ray* rays)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.projection.width && y < cam.projection.height) {
		int index = x + (y * cam.projection.width);

		float perPixel = 2.0f / (float)cam.projection.height;
		float halfScreenHeight = 1.0f;
		float halfScreenWidth = 1.0f * (float)cam.projection.width / (float)cam.projection.height;

		float3 screenDistanceVec = cam.projection.viewer_dist * normalize(cam.direction);
		float3 pixVector = (halfScreenWidth - x * perPixel - perPixel / 2.0f) * normalize(cross(cam.up, cam.direction)) +
			(halfScreenHeight - y * perPixel - perPixel / 2.0f) * normalize(cam.up);

		rays[index].direction = normalize(screenDistanceVec + pixVector);
		if ((x == 0 && y == 0) ||
			(x == 0 && y == cam.projection.height - 1) ||
			(x == cam.projection.width - 1 && y == 0) ||
			(x == cam.projection.width - 1 && y == cam.projection.height - 1)) {

			printf("\nRay(%d, %d) = [%.9f, %.9f, %.9f]\n  pixVector = [%f, %f, %f]\n\n",
				x, y, rays[index].direction.x, rays[index].direction.y, rays[index].direction.z,
				pixVector.x, pixVector.y, pixVector.z);
		}
		rays[index].originPoint = cam.position;
	}
}

// Trace rays
__global__
void traceRays(Scene scene, Ray* primary_rays, float3* image)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	Camera& cam = scene.camera;
	

	if (x < cam.projection.width && y < cam.projection.height) {
		int index = x + (y * cam.projection.width);

		float4 rec_color;

		if (rayIntersectsScene(primary_rays[index], scene, rec_color)) {
			image[index] = make_float3(rec_color.x, rec_color.y, rec_color.z);

		} else {
			image[index] = make_float3(0.0);
		}
	}
}

__host__
void runPathTracing()
{
	cuassert(scene.dv_wobjects_ptr != NULL);

	const int blockSideLength = 24;
	const dim3 blockSize(blockSideLength, blockSideLength);
	const dim3 blocksPerGrid(
		(viewWidth + blockSize.x - 1) / blockSize.x,
		(viewHeight + blockSize.y - 1) / blockSize.y);

	if (scene.camera.changed) {
		generatePrimaryRays << <blocksPerGrid, blockSize >> >(scene.camera, prim_rays);
		cudaDeviceSynchronize();
		checkCudaError("generatePrimaryRays<<<>>>()");

		scene.camera.changed = false;
	}

	traceRays << <blocksPerGrid, blockSize >> >(scene, prim_rays, dev_image);
	cudaDeviceSynchronize();
	checkCudaError("traceRays<<<>>>()");
}
