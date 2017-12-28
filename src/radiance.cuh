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

	// make rays path: bounce one ray from each intersection point
	for (int bounce = 0; bounce < scene.camera.max_ray_bounces; ++bounce) {
		IntersectInfo ii;

		if (!rayIntersectsScene(&ray, scene, ii)) {
			if (scene.camera.aabb_mode)
				return (make_float3(1.0f) - ii.debug_mask);
			return make_float3(0);
		}
		
		if (scene.camera.aabb_mode)
			return 0.8 * (make_float3(1.0f) - ii.debug_mask) + 0.2 * ii.imat->color * fabs(dot(ray.direction, ii.normal));

		const Material* mat = ii.imat;

		// preview mode radiance computing
		if (scene.camera.preview_mode) {
			float3 tex_color = make_float3(0.0f);
			float tex_blend = 0.0f;
			float alpha = 0.0f;
			if (scene.camera.texture_enabled && mat->cuda_texture_obj > -1) {
				float2 texuv = ii.bary_coords.x * ii.itrg.tx_a + ii.bary_coords.y * ii.itrg.tx_b + ii.bary_coords.z * ii.itrg.tx_c;
				float4 texel = tex2D<float4>((cudaTextureObject_t)mat->cuda_texture_obj, texuv.x, texuv.y);
				alpha = texel.w;
				tex_color = make_float3(texel.x, texel.y, texel.z);
				tex_blend = 0.5f;
			}

			float rdot = dot(ii.normal, ray.direction);
			if (rdot < 0)
				return -rdot * (alpha*tex_blend*tex_color + (1.0f - tex_blend) * make_float3(1.0f));
			else return rdot * (tex_blend*tex_color + (1.0f - tex_blend) * make_float3(1.0f, 0, 0));
		}


		// shading pixels in normal mode-------------------------------------------------
		switch (mat->type) {
			// material cases
			case Diffuse:
				Diffuse_BRDF(&ray, ii.normal, ii.ipoint, curand_s); break;
			case Specular:
				Specular_Diffuse_BRDF(&ray, mat->reflect_factor, mat->sharpness, ii.normal, ii.ipoint, curand_s); break;
			case Transparent:
				Refractive_BRDF(&ray, mat->refract_index, mat->reflect_factor, mask, ii.normal, ii.ipoint, curand_s); break;
		}
		
		// if material has texture then blend it with material color
		if (scene.camera.texture_enabled && mat->cuda_texture_obj > -1) {
			float2 texuv = ii.bary_coords.x * ii.itrg.tx_a + ii.bary_coords.y * ii.itrg.tx_b + ii.bary_coords.z * ii.itrg.tx_c;
			float4 texel = tex2D<float4>((cudaTextureObject_t)mat->cuda_texture_obj, texuv.x, texuv.y);
			mask *= 1.0f * make_float3(texel.x, texel.y, texel.z) + 0.0f * mat->color;

		} else // mask light with current object colour
			mask *= mat->color;

		// if ray hit light then terminate path
		if (mat->type == Luminescent)
			return mask * mat->emittance;
	}

	return radiance;//mask * make_float3(255, 255, 255);
}