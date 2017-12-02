#pragma once

#include "world.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "cutil_math.h"
#include "curand_kernel.h"
#include "constants.h"
#include "config.h"


//DIFFUSE SURFACE // cosine weighed Lambertian model
__device__
inline void Diffuse_BRDF(	Ray* in_ray,
							 float3& surf_normal,
							 float3& inters_point,
							curandState* curand_s) {

	float r1 = PI_X2_f * curand_uniform(curand_s);
	float r2 = curand_uniform(curand_s);
	float r2s = __fsqrt_rn(r2);

	float3 u = normalize(cross((fabsf(surf_normal.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), surf_normal));
	float3 v = cross(surf_normal, u);

	float sinr1, cosr1;
	__sincosf(r1, &sinr1, &cosr1);

	init_ray(*in_ray, inters_point + surf_normal * 0.00005f,
		normalize(u * cosr1 * r2s + v * sinr1*r2s + surf_normal * __fsqrt_rn(1 - r2)));
}

// DIFFUSE-REFLECTIVE SURFACE // perfect reflective + cosine weighed Lambertian model
__device__
inline void ReflectiveDiffuse_BRDF(	Ray* in_ray,
									float refl_factor,
									 float3& surf_normal,
									 float3& inters_point,
									curandState* curand_s) {

	float3 new_orig = inters_point + surf_normal * 0.0001f;
	float r2 = curand_uniform(curand_s);
	if (r2 > refl_factor) {
		// ray scattering
		float r1 = PI_X2_f * curand_uniform(curand_s);
		float r2s = __fsqrt_rn(r2);

		float3 u = normalize(cross((fabsf(surf_normal.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), surf_normal));
		float3 v = cross(surf_normal, u);

		float sinr1, cosr1;
		__sincosf(r1, &sinr1, &cosr1);

		init_ray(*in_ray, new_orig, normalize(u * cosr1 * r2s + v * sinr1*r2s + surf_normal * __fsqrt_rn(1 - r2)));
	}
	else {
		// perfect reflection
		init_ray(*in_ray, new_orig, normalize(reflect(in_ray->direction, surf_normal)));
	}
	
}

// REFRACTIVE SURFACE // perfect reflective + fresnel effects by shortened Schlick Approximation
#define REFL_BIAS		0.05f
#define REFL_BIAS_LOW	0.0005f
__device__
inline void Refractive_BRDF(Ray* in_ray,
							float refract_index,
							float refl_factor,
							float3& mask,
							 float3& surf_normal,
							 float3& inters_point,
							curandState* curand_s) {

	// check if ray is inside refrected object
	float3 ray_oriented_norm = dot(surf_normal, in_ray->direction) < 0 ? surf_normal : -(float3)surf_normal;
	bool ray_outside = dot(surf_normal, ray_oriented_norm) > 0;

	// refract indices
	float n_obj = refract_index;

	float nn_ratio = ray_outside ? (SCENE_REFRACTION_INDEX / n_obj) : (n_obj / SCENE_REFRACTION_INDEX); // refract indices ratio

	float cos_ray = dot(in_ray->direction, ray_oriented_norm); // cosine of ray and oriented normal angle
	float cos2refr = 1.0f - nn_ratio * nn_ratio * (1.0f - cos_ray * cos_ray);

	if (cos2refr < 0.0f) // total internal reflection 
	{
		init_ray(*in_ray, inters_point + ray_oriented_norm * REFL_BIAS,
			reflect(in_ray->direction, ray_oriented_norm));
	}
	else // refraction of light ray
	{
		float cosrefr = sqrtf(cos2refr);

		// direction of transmission ray
		float3 trans_ray_dir = normalize(in_ray->direction * nn_ratio - surf_normal *
			((ray_outside ? 1 : -1) * (cos_ray*nn_ratio + cosrefr)));

		// Fresnel effects - using Schlick Approximation
		float ndiff = n_obj - SCENE_REFRACTION_INDEX;
		float nsum = n_obj + SCENE_REFRACTION_INDEX;
		float Refl0 = (ndiff*ndiff)/(nsum*nsum);
			
		float pow = 1.0f - (ray_outside ? -cos_ray : dot(trans_ray_dir, surf_normal));
		// shortened Schlick Approximation
		float Reflectance = Refl0 + (1.0f - Refl0)*pow*pow*pow;
		// scale energy -
		// due to splitting ray energy on two rays and losing one of them
		// it had to be scaled to keep conservation of energy law
		float EnergyScale = refl_factor + (1.0f - refl_factor - refl_factor) * Reflectance;

		// random ray split
		if (curand_uniform(curand_s) < refl_factor)
		{//reflection
			mask *= (Reflectance / EnergyScale);
			init_ray(*in_ray, inters_point + ray_oriented_norm * REFL_BIAS,
				reflect(in_ray->direction, surf_normal));
		}
		else
		{//transmision
			mask *= ((1.0f - Reflectance) / (1.0f - EnergyScale));
			init_ray(*in_ray, inters_point - ray_oriented_norm * REFL_BIAS_LOW,
				trans_ray_dir);
		}
	}
	
}
