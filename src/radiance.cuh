#pragma once

#include "cuda.h"
#include "cuda_runtime.h"
#include "cutil_math.h"
#include "world.h"
#include "curand_kernel.h"
#include "intersect.cuh"

__device__
float3 gatherRadiance(Ray& prim_ray, Scene& scene, curandState* curand_s)
{
	float3 radiance = make_float3(0.0f);
	float3 mask = make_float3(1.0f, 1.0f, 1.0f);
	Ray ray = prim_ray;

	// bounce ray
	for (int bounce = 0; bounce < 4; ++bounce) {
		int inters_obj_idx;
		float3 inters_point, surf_normal;

		if (!rayIntersectsScene(ray, scene, inters_obj_idx, inters_point, surf_normal))
			return make_float3(0.0);

		// take hit object reference
		WorldObject& obj = scene.dv_wobjects_ptr[inters_obj_idx];
		//return obj.material.color * abs(dot(surf_normal, ray.direction));
		if (obj.material.type == Luminescent) {
			// add light object emmited energy
			radiance += mask * obj.material.emittance;
			break;
		}
			

		if (obj.material.type == Diffusing) {

			float r1 = 2 * 3.141592 * curand_uniform(curand_s);
			float r2 = curand_uniform(curand_s);
			float r2s = sqrtf(r2);

			float3 w = surf_normal;
			float3 u = normalize(cross((fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));
			float3 v = cross(w, u);

			ray.direction = normalize(u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrtf(1 - r2));

			ray.originPoint = surf_normal * 0.03f + inters_point;

			// mask color object
			mask *= (obj.material.color * abs(dot(ray.direction, surf_normal)) / 255.0f);
		}

	}

	return radiance;
}