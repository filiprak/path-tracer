#include "pathtracing.cuh"
#include "kernel.h"
#include "stdio.h"
#include "radiance.cuh"
#include "cudaUtility.h"
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
inline void init_ray(Ray* r, const float3& orig, const float3& dir) {
	r->direction = dir;
	r->originPoint = orig;
	r->inv_direction = make_float3(1 / dir.x, 1 / dir.y, 1 / dir.z);
	r->sign[0] = (r->inv_direction.x < 0);
	r->sign[1] = (r->inv_direction.y < 0);
	r->sign[2] = (r->inv_direction.z < 0);
}

__device__
inline void change_ray_dir(Ray& r, const float3& new_dir) {
	r.direction = new_dir;
	r.inv_direction = make_float3(1 / new_dir.x, 1 / new_dir.y, 1 / new_dir.z);
	r.sign[0] = (r.inv_direction.x < 0);
	r.sign[1] = (r.inv_direction.y < 0);
	r.sign[2] = (r.inv_direction.z < 0);
}

// Init pathtracing
__host__
void initPathTracing(const Scene& scene) {
	const Camera& cam = scene.camera;
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
void generatePrimaryRays(const Camera cam, Ray* rays, float3* pix_midpoints)
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

	const Camera& cam = scene.camera;

	if (x < cam.projection.width && y < cam.projection.height) {
		int index = x + (y * cam.projection.width);

		curandState curand_state;
		initRandomCuda(seed, curand_state);

		// calculate jitter (jitter bounds are <-pix_side, pix_side>)
		float2 jitter = (cam.projection.pixel_size.x) * cam.projection.aa_jitter * 
			make_float2(curand_uniform(&curand_state) - 0.5f, curand_uniform(&curand_state) - 0.5f);

		// change primary ray direction adding the jitter
		float3 jitter_dir = normalize(pix_midpoints[index] + jitter.x * cam.right + jitter.y * cam.up - cam.position);
		change_ray_dir(out_rays[index], jitter_dir);

	}
}

// Trace rays
__global__
void tracePaths(int iterHash, Scene scene, Ray* primary_rays, float4* acc_image)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	const Camera& cam = scene.camera;
	
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

__host__
void runPathTracing(Scene& scene, int iterHash, int jitterHash)
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
