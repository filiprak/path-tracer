#pragma once

#include "world.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "cutil_math.h"
#include "curand_kernel.h"
#include "constants.h"
#include "config.h"

// Helper function to calculate random ray direction inside specified cone of directions
// Uses random cosine weighed cone sampling
__device__
inline void rand_cone_Dir(	float3* result,
							const float3& cone_axis,
							float ang_max_dev,
							curandState* curand_s) {
#define EPS		0.0001
	float3 u = normalize(cone_axis.x != 0.0f || cone_axis.y != 0.0f ?
		(make_float3(cone_axis.y, -cone_axis.x, 0.0f)) : 
		(make_float3(0.0f, cone_axis.z, -cone_axis.y)));
	float3 v = normalize(cross(u, cone_axis));

	const float r2 = __sinf(ang_max_dev) * curand_uniform(curand_s);
	const float r = __fsqrt_rn(r2);
	const float theta = PI_X2_f * curand_uniform(curand_s);

	float sinth, costh;
	__sincosf(theta, &sinth, &costh);

	*result = normalize(sinth * r * u + costh * r * v + __fsqrt_rn(1.0f - r2 + EPS) * cone_axis);
}


//DIFFUSE SURFACE // cosine weighed Lambertian model
__device__
inline void Diffuse_BRDF(	Ray* in_ray,
							const float3& surf_normal,
							const float3& inters_point,
							curandState* curand_s) {
	float3 rand_dir;
	rand_cone_Dir(&rand_dir, surf_normal, PI_D2_f, curand_s);
	init_ray(in_ray, inters_point + surf_normal * 0.00005f, rand_dir);
}

// DIFFUSE-REFLECTIVE SURFACE // perfect reflective + cosine weighed Lambertian model
__device__
inline void Specular_Diffuse_BRDF(	Ray* in_ray,
									float refl_factor,
									float sharpness,
									const float3& surf_normal,
									const float3& inters_point,
									curandState* curand_s) {

	float3 new_orig = inters_point + surf_normal * 0.0001f;
	float r2 = curand_uniform(curand_s);
	if (r2 > refl_factor) {
		// diffuse rayy scatter
		Diffuse_BRDF(in_ray, surf_normal, inters_point, curand_s);
	}
	else {
		// glossy reflection with sharpness
		const float max_cone_ang = PI_D2_f * (1.0f - sharpness);
		const float scatt_ang = PI_D2_f - acosf(fabs(dot(in_ray->direction, surf_normal)));
		float3 rand_dir;
		rand_cone_Dir(&rand_dir, reflect(in_ray->direction, surf_normal), fmin(max_cone_ang, scatt_ang), curand_s);
		init_ray(in_ray, new_orig, rand_dir);
	}
	
}

// REFRACTIVE SURFACE // perfect reflective + fresnel effects by Schlick Approximation
#define REFL_BIAS		0.05f
#define REFL_BIAS_LOW	0.0005f
__device__
inline void Refractive_BRDF(	Ray* in_ray,
								float refract_index,
								float refl_factor,
								float3& mask,
								const float3& surf_normal,
								const float3& inters_point,
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
		init_ray(in_ray, inters_point + ray_oriented_norm * REFL_BIAS,
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
		float Reflectance = Refl0 + (1.0f - Refl0)*pow*pow*pow*pow*pow;
		// scale energy -
		// due to splitting ray energy on two rays and losing one of them
		// it had to be scaled to keep conservation of energy law
		float EnergyScale = refl_factor + (1.0f - refl_factor - refl_factor) * Reflectance;

		// random ray split
		if (curand_uniform(curand_s) < refl_factor)
		{//reflection
			mask *= (Reflectance / EnergyScale);
			init_ray(in_ray, inters_point + ray_oriented_norm * REFL_BIAS,
				reflect(in_ray->direction, surf_normal));
		}
		else
		{//transmision
			mask *= ((1.0f - Reflectance) / (1.0f - EnergyScale));
			init_ray(in_ray, inters_point - ray_oriented_norm * REFL_BIAS_LOW,
				trans_ray_dir);
		}
	}
	
}
