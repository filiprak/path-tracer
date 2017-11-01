#pragma once

#include "cuda.h"
#include "cuda_runtime.h"
#include "cutil_math.h"
#include "world.h"
#include "curand_kernel.h"
#include "intersect.cuh"
#include "constants.h"

__device__
float3 gatherRadiance(Ray& prim_ray, Scene& scene, curandState* curand_s)
{
	// result radiance for primary ray
	float3 radiance = make_float3(0.0f);

	// color mask
	float3 mask = make_float3(1.0f, 1.0f, 1.0f);

	// copy primary ray instance
	Ray ray = prim_ray;

	// bounce ray
	for (int bounce = 0; bounce < 4; ++bounce) {
		int inters_obj_idx;
		float3 inters_point, surf_normal;

		if (!rayIntersectsScene(ray, scene, inters_obj_idx, inters_point, surf_normal))
			return make_float3(0.0);

		// take hit object reference
		WorldObject& obj = scene.dv_wobjects_ptr[inters_obj_idx];

		// shading pixels -------------------------------------------------
		//return obj.material.color * abs(dot(surf_normal, ray.direction));
		if (obj.material.type == Luminescent) {
			// add light object emmited energy
			radiance += mask * obj.material.emittance;
			break;
		}
			
		if (obj.material.type == Diffusing) {

			float r1 = PI_X2_f * curand_uniform(curand_s);
			float r2 = curand_uniform(curand_s);
			float r2s = __fsqrt_rn(r2);

			float3 u = normalize(cross((fabsf(surf_normal.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), surf_normal));
			float3 v = cross(surf_normal, u);

			float sinr1, cosr1;
			__sincosf(r1, &sinr1, &cosr1);

			ray.direction = normalize(u * cosr1 * r2s + v * sinr1*r2s + surf_normal * __fsqrt_rn(1 - r2));
			ray.originPoint = inters_point;

			// mask color object
			mask *= (obj.material.norm_color);
		}

		if (obj.material.type == Reflective){
			float r2 = curand_uniform(curand_s);
			if (r2 > 0.8) {
				float r1 = PI_X2_f * curand_uniform(curand_s);
				float r2s = __fsqrt_rn(r2);

				float3 u = normalize(cross((fabsf(surf_normal.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), surf_normal));
				float3 v = cross(surf_normal, u);

				float sinr1, cosr1;
				__sincosf(r1, &sinr1, &cosr1);

				ray.direction = normalize(u * cosr1 * r2s + v * sinr1*r2s + surf_normal * __fsqrt_rn(1 - r2));
				ray.originPoint = inters_point;
			}
			else {
				ray.direction = normalize(reflect(ray.direction, surf_normal));
				ray.originPoint = inters_point;
			}

			// mask color object
			mask *= (obj.material.norm_color);
		}

	}

	return radiance;
}