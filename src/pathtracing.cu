#include "pathtracing.cuh"
#include "kernel.h"
#include "stdio.h"
#include "view.h"
#include "radiance.cuh"
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

__device__
inline void init_ray(Ray& r, float3& orig, float3& dir) {
	r.direction = dir;
	r.originPoint = orig;
	r.inv_direction = make_float3(1 / dir.x, 1 / dir.y, 1 / dir.z);
	r.sign[0] = (r.inv_direction.x < 0);
	r.sign[1] = (r.inv_direction.y < 0);
	r.sign[2] = (r.inv_direction.z < 0);
}

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

		Ray ray;
		init_ray(ray, cam.position, normalize(screenDistanceVec + pixVector));
		rays[index] = ray;
		/*if ((x == 0 && y == 0) ||
			(x == 0 && y == cam.projection.height - 1) ||
			(x == cam.projection.width - 1 && y == 0) ||
			(x == cam.projection.width - 1 && y == cam.projection.height - 1)) {

			printf("\nRay(%d, %d) = [%.9f, %.9f, %.9f]\n  pixVector = [%f, %f, %f]\n\n",
				x, y, rays[index].direction.x, rays[index].direction.y, rays[index].direction.z,
				pixVector.x, pixVector.y, pixVector.z);
		}*/
	}
}

// init random number generator
__device__
void initRandomCuda(int iterHash, curandState& randState) {
	int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y)
		+ (threadIdx.y * blockDim.x) + threadIdx.x;

	curand_init(iterHash + threadId, 0, 0, &randState);
}

// Trace rays
__global__
void tracePaths(int iterHash, Scene scene, Ray* primary_rays, float4* acc_image)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	Camera& cam = scene.camera;
	
	if (x < cam.projection.width && y < cam.projection.height) {
		int index = x + (y * cam.projection.width);

		// init random number generator
		curandState curand_state;
		initRandomCuda(iterHash, curand_state);

		// gather radiance from rays path for current pixel
		float4 radiance = make_float4(gatherRadiance(primary_rays[index], scene, &curand_state));

		acc_image[index] += radiance;
	}
}

//DEBUG
__global__
void debugTexture(Scene scene) {
	for (int i = 0; i < scene.num_wobjects; ++i) {
		WorldObject& obj = scene.dv_wobjects_ptr[i];

		if (obj.type == TriangleMeshObj) {
			MeshGeometryData* gd = (MeshGeometryData*)obj.geometry_data;
			int tex = gd->triangles_tex;
			printf("texture id: %d\n", tex);
			for (int t = 0; t < gd->num_triangles; t++)
			{
				for (int s = 0; s < 6; s++)
				{
					float4 a = tex1Dfetch<float4>(tex, 6 * t + s);
					printf("[%d+%d]: [%f,%f,%f,%f]\n", t, s, a.x, a.y, a.z, a.w);
				}
				printf("\n");
			}
			return;
		}
	}
}

__host__
void runPathTracing(int iterHash)
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

		/*debugTexture<<<1,1>>>(scene);
		checkCudaError("debugTexture<<<1,1>>>(scene)");*/

		scene.camera.changed = false;
	}

	tracePaths << <blocksPerGrid, blockSize >> >(iterHash, scene, prim_rays, device_accum_image);
	cudaDeviceSynchronize();
	checkCudaError("traceRays<<<>>>()");
}
