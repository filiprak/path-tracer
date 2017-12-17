#include "pathtracing.cuh"
#include "kernel.h"
#include "stdio.h"
#include "radiance.cuh"
#include "cudaUtility.h"
#include "world_load.h"
#include "cutil_math.h"


// primary rays jittered through pixel with uniform sampling
static Ray* dv_prim_rays;

// primary rays hitting pixels right in the middle point
static float3* dv_pix_midpoints;
static int sizeof_prim_rays;

// func decl
__host__
void loadWorldObjects(Camera cam, WorldObject* wobjects);

__global__
void generatePrimaryRays(Camera cam, Ray* rays);

__device__
inline void init_ray(Ray* r, float3& orig, float3& dir) {
	r->direction = dir;
	r->originPoint = orig;
	r->inv_direction = make_float3(1 / dir.x, 1 / dir.y, 1 / dir.z);
	r->sign[0] = (r->inv_direction.x < 0);
	r->sign[1] = (r->inv_direction.y < 0);
	r->sign[2] = (r->inv_direction.z < 0);
}

__device__
inline void change_ray_dir(Ray& r, float3& new_dir) {
	r.direction = new_dir;
	r.inv_direction = make_float3(1 / new_dir.x, 1 / new_dir.y, 1 / new_dir.z);
	r.sign[0] = (r.inv_direction.x < 0);
	r.sign[1] = (r.inv_direction.y < 0);
	r.sign[2] = (r.inv_direction.z < 0);
}

// Init pathtracing
__host__
void initPathTracing() {
	Camera& cam = scene.camera;
	sizeof_prim_rays = cam.projection.num_pixels * sizeof(Ray);

	cudaOk(cudaMalloc(&dv_prim_rays, sizeof_prim_rays));
	cudaOk(cudaMemset(dv_prim_rays, 0, sizeof_prim_rays));
	cudaOk(cudaMalloc(&dv_pix_midpoints, scene.camera.projection.num_pixels * sizeof(float3)));
	cudaOk(cudaMemset(dv_pix_midpoints, 0, scene.camera.projection.num_pixels * sizeof(float3)));

	const int blockSideLength = 8;
	const dim3 blockSize(blockSideLength, blockSideLength);
	const dim3 blocksPerGrid(
		(cam.projection.width + blockSize.x - 1) / blockSize.x,
		(cam.projection.height + blockSize.y - 1) / blockSize.y);

	generatePrimaryRays << <blocksPerGrid, blockSize >> >(scene.camera, dv_prim_rays, dv_pix_midpoints);
	cudaDeviceSynchronize();
	checkCudaError("generatePrimaryRays<<<>>>()");
}

__host__
void cleanUpPathTracing()
{
	cudaOk(cudaFree(dv_prim_rays));
	cudaOk(cudaFree(dv_pix_midpoints));
}


// Init primary rays
__global__
void generatePrimaryRays(Camera cam, Ray* rays, float3* pix_midpoints)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.projection.width && y < cam.projection.height) {
		int index = x + (y * cam.projection.width);

		float pix_side = cam.projection.pixel_size.x;

		float3 screenDistanceVec = cam.projection.screen_dist * normalize(cam.direction);
		float3 pixVector = (cam.projection.screen_halfsize.x - x * pix_side - pix_side / 2.0f) * normalize(cam.right) +
			(cam.projection.screen_halfsize.y - y * pix_side - pix_side / 2.0f) * normalize(cam.up);

		// init primary ray
		init_ray(&(rays[index]), cam.position, normalize(screenDistanceVec + pixVector));

		// store middle pixel point
		pix_midpoints[index] = cam.position + screenDistanceVec + pixVector;
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

// Jitter rays radomly shooting them through the pixel to get anti-aliasing effect
__global__
void jitterPrimaryRays(Scene scene, float3* pix_midpoints, Ray* out_rays, int seed) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	Camera& cam = scene.camera;

	if (x < cam.projection.width && y < cam.projection.height) {
		int index = x + (y * cam.projection.width);

		curandState curand_state;
		initRandomCuda(seed, curand_state);

		// calculate jitter (jitter bounds are <-pix_side, pix_side>)
		float2 jitter = (cam.projection.pixel_size.x) * cam.projection.aa_jitter * 
			make_float2(curand_uniform(&curand_state) - 0.5f, curand_uniform(&curand_state) - 0.5f);
		// change primary ray direction adding the jitter
		float3 jitter_dir = normalize(pix_midpoints[index] + jitter.x * cam.right + jitter.y * cam.up - cam.position);
		//float3 jitter_orig = cam.position + jitter.x * cam.right + jitter.y * cam.up;
		change_ray_dir(out_rays[index], jitter_dir);
		//out_rays[index].originPoint = jitter_orig;
		/*if (index == 0) {
			printf("pix side: %f\n", cam.projection.pixel_size.x);
			printf(" mid point [0] = [%f, %f, %f]\n", pix_midpoints[0].x, pix_midpoints[0].y, pix_midpoints[0].z);
			printf("        jitter = [%f, %f]\n", jitter.x, jitter.y);
			printf("jitter ray [0] = [%f, %f, %f]\n\n", out_rays[0].direction.x, out_rays[0].direction.y, out_rays[0].direction.z);
		}*/
	}
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

		// accumulate image
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
void runPathTracing(int iterHash, int jitterHash)
{
	Camera& cam = scene.camera;

	const int blockSideLength = 8;
	const dim3 blockSize(blockSideLength, blockSideLength);
	const dim3 blocksPerGrid(
		(cam.projection.width + blockSize.x - 1) / blockSize.x,
		(cam.projection.height + blockSize.y - 1) / blockSize.y);

	if (scene.camera.changed) {
		generatePrimaryRays << <blocksPerGrid, blockSize >> >(scene.camera, dv_prim_rays, dv_pix_midpoints);
		cudaDeviceSynchronize();
		checkCudaError("generatePrimaryRays<<<>>>()");

		/*debugTexture<<<1,1>>>(scene);
		checkCudaError("debugTexture<<<1,1>>>(scene)");*/

		scene.camera.changed = false;
	}

	// jitter sampling primary rays to get antialiasing effect
	jitterPrimaryRays << <blocksPerGrid, blockSize >> > (scene, dv_pix_midpoints, dv_prim_rays, jitterHash);
	cudaDeviceSynchronize();
	checkCudaError("jitterPrimaryRays<<<>>>()");

	tracePaths << <blocksPerGrid, blockSize >> >(iterHash, scene, dv_prim_rays, device_accum_image);
	cudaDeviceSynchronize();
	checkCudaError("traceRays<<<>>>()");
}
