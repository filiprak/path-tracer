#include "pathtracing.h"
#include "kernel.h"
#include "stdio.h"
#include "view.h"
#include "intersect.h"
#include "cudaUtility.h"

#include "glm\gtc\matrix_transform.hpp"
#include "glm\gtc\matrix_inverse.hpp"

// memory
static Ray* prim_rays = NULL;
static WorldObject* host_world_objects;
static WorldObject* device_world_objects;

static glm::vec3* vert = NULL;
static int* ind = NULL;

// func decl
__host__
void loadWorldObjects(Camera cam, WorldObject* wobjects);

__global__
void generatePrimaryRays(Camera cam, Ray* rays);


// Init pathtracing
__host__
void initPathTracing() {

	int pixel_num = viewWidth * viewHeight;

	cudaMalloc(&prim_rays, pixel_num * sizeof(Ray));
	cudaMemset(prim_rays, 0, pixel_num * sizeof(Ray));

	cudaMalloc(&vert, 8 * sizeof(glm::vec3));
	cudaMalloc(&ind, 36 * sizeof(int));

	host_world_objects = (WorldObject*)malloc(1 * sizeof(WorldObject));

	loadWorldObjects(scene.camera, host_world_objects);
	cudaDeviceSynchronize();
	checkCudaError("loadWorldObjects()");

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
	cudaFree(prim_rays);
	
	cudaFree(vert);
	cudaFree(ind);
	cudaFree(device_world_objects);

	free(host_world_objects);
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

// Load world objects
__host__
void loadWorldObjects(Camera cam, WorldObject* wobjects) {
	
	cudaMalloc(&device_world_objects, 1 * sizeof(WorldObject));

	wobjects[0].materialType = Light;
	wobjects[0].transformMat = glm::translate(glm::mat4() , glm::vec3(- 2.0, 2.0, -2.0));
	wobjects[0].inversedTransMat = glm::inverse(wobjects[0].inversedTransMat);

	wobjects[0].vertices[0] = glm::vec3(-0.5, -0.5, 0.5);
	wobjects[0].vertices[1] = glm::vec3(0.5, -0.5, 0.5);
	wobjects[0].vertices[2] = glm::vec3(0.5, 0.5, 0.5);
	wobjects[0].vertices[3] = glm::vec3(-0.5, 0.5, 0.5);
	wobjects[0].vertices[4] = glm::vec3(-0.5, -0.5, -0.5);
	wobjects[0].vertices[5] = glm::vec3(0.5, -0.5, -0.5);
	wobjects[0].vertices[6] = glm::vec3(0.5, 0.5, -0.5);
	wobjects[0].vertices[7] = glm::vec3(-0.5, 0.5, -0.5);

	int indices[36] = {
	// front
	0, 1, 2,
	2, 3, 0,
	// top
	1, 5, 6,
	6, 2, 1,
	// back
	7, 6, 5,
	5, 4, 7,
	// bottom
	4, 0, 3,
	3, 7, 4,
	// left
	4, 5, 1,
	1, 0, 4,
	// right
	3, 2, 6,
	6, 7, 3,
	};

	for (int i = 0; i < 36; ++i) {
		wobjects[0].indices[i] = indices[i];
	}

	cudaMemcpy(device_world_objects, host_world_objects, 1 * sizeof(WorldObject), cudaMemcpyHostToDevice);
}


// Trace rays
__global__
void traceRays(Camera cam, WorldObject* wobjs, Ray* primary_rays, glm::vec3* image)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.projection.width && y < cam.projection.height) {
		int index = x + (y * cam.projection.width);

		glm::vec3 inters_point;
		if (rayIntersectsObject(primary_rays[index], wobjs[0], inters_point)) {
			image[index] = glm::vec3(255.0);

		} else {
			image[index] = glm::vec3(0.0);
		}
	}
}

__host__
void runPathTracing()
{
	const int blockSideLength = 8;
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

	traceRays << <blocksPerGrid, blockSize >> >(scene.camera, device_world_objects, prim_rays, dev_image);
	cudaDeviceSynchronize();
	checkCudaError("traceRays<<<>>>()");
}
