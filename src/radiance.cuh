#pragma once

#include "cuda.h"
#include "cuda_runtime.h"
#include "cutil_math.h"
#include "world.h"
#include "curand_kernel.h"
#include "intersect.cuh"
#include "constants.h"
#include "surfaces.cuh"
#include "config.h"

#include <string>

__host__ __device__
void print_4path(float3* path) {
	printf("{Wektor((%.2f,%.2f,%.2f),(%.2f,%.2f,%.2f)),\nWektor((%.2f,%.2f,%.2f),(%.2f,%.2f,%.2f)),\nWektor((%.2f,%.2f,%.2f),(%.2f,%.2f,%.2f)),\nWektor((%.2f,%.2f,%.2f),(%.2f,%.2f,%.2f))}\n\n",
		path[0].x, path[0].y, path[0].z, path[1].x, path[1].y, path[1].z, path[2].x, path[2].y, path[2].z, path[3].x, path[3].y, path[3].z,
		path[4].x, path[4].y, path[4].z, path[5].x, path[5].y, path[5].z, path[6].x, path[6].y, path[6].z, path[7].x, path[7].y, path[7].z);
}

__device__
float3 gatherRadiance(Ray& prim_ray, Scene& scene, curandState* curand_s)
{
	// result radiance for primary ray
	float3 radiance = make_float3(0.0f);

	// color mask
	float3 mask = make_float3(1.0f, 1.0f, 1.0f);

	// copy primary ray instance
	Ray ray = prim_ray;

	/*float3 debug_path[64];
	debug_path[0] = ray.originPoint;
	bool hit_refractive = false;*/

	// bounce ray
	for (int bounce = 0; bounce < MAX_NUM_RAY_BOUNCES; ++bounce) {
		int inters_obj_idx;
		float3 inters_point, surf_normal;
		

		if (!rayIntersectsScene(ray, scene, inters_obj_idx, inters_point, surf_normal))
			return make_float3(0.0);
		
		Material mat = scene.dv_wobjects_ptr[inters_obj_idx].material;
		// take hit object reference
		/*if (obj.type == TriangleMeshObj) {
			MeshGeometryData* md = (MeshGeometryData*)obj.geometry_data;
			mat = md->meshes[mesh_idx].material;
			printf("mesh: %d\n    Ka: [%.1f,%.1f,%.1f]\n    Ke: [%.1f,%.1f,%.1f]\n    d: %f, Ni: %f\n",
				mesh_idx,
				mat.norm_color.x, mat.norm_color.y, mat.norm_color.z,
				mat.emittance.x, mat.emittance.y, mat.emittance.z,
				mat.reflect_factor, mat.refract_index);
		}*/
		
		/*if (!hit_refractive && bounce == 0)
			hit_refractive = obj.material.type == Refractive;
		if (hit_refractive)
			debug_path[2 * bounce] = ray.originPoint;*/
		// shading pixels -------------------------------------------------
		switch (mat.type) {

			// material cases
			case Luminescent: radiance += mask * mat.emittance; return radiance;
			case Diffusing: Diffuse_BRDF(ray, ray, surf_normal, inters_point, curand_s); break;
			case Reflective: ReflectiveDiffuse_BRDF(ray, ray, mat.reflect_factor, surf_normal, inters_point, curand_s); break;
			case Refractive: Refractive_BRDF(ray, ray, mat.refract_index,
				mat.reflect_factor, mask, surf_normal, inters_point, curand_s); break;

		}
		// mask light with current object colour
		mask *= mat.norm_color;
		/*if (hit_refractive)
			debug_path[2 * bounce + 1] = inters_point;*/
	}

	//print_4path(debug_path);
	return radiance;//mask * make_float3(255, 255, 255);
}