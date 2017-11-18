#pragma once

#include "world.h"
#include "cuda.h"
#include "config.h"
#include "cuda_runtime.h"
#include "cutil_math.h"
#include "cudaUtility.h"

#define EPSILON		0.0000001

/* Möller–Trumbore ray-triangle intersection algorithm */
__device__
bool rayIntersectsTriangle(Ray& ray, float3& va, float3& vb, float3& vc, float& res_dist)
{
	float3 edge1, edge2, h;
	float a;
	edge1 = vb - va;
	edge2 = vc - va;

	h = cross(ray.direction, edge2);
	a = dot(edge1, h);

	if (a > -EPSILON && a < EPSILON)
		return false;

	float f = __fdividef(1.0, a);
	float3 s = ray.originPoint - va;
	float u = f * (dot(s, h));
	if (u < 0.0 || u > 1.0)
		return false;

	float3 q = cross(s, edge1);
	float v = f * (dot(ray.direction, q));
	if (v < 0.0 || u + v > 1.0)
		return false;

	// At this stage we can compute t to find out where the intersection point is on the line.
	float t = f * dot(edge2, q);
	if (t > EPSILON) // ray intersection
	{
		res_dist = t;
		return true;
	}
	// This means that there is a line intersection but not a ray intersection.
	return false;
}

__device__
bool testSphereIntersection(Ray& ray, WorldObject& obj, float3& hit_point, float3& hit_norm) {
	SphereGeometryData* gdata = (SphereGeometryData*)obj.geometry_data;
	float rad = gdata->radius;

	float3 op = gdata->position - ray.originPoint;
	float t, epsilon = 0.0001f;
	float b = dot(op, ray.direction);
	float disc = b*b - dot(op, op) + rad*rad; // discriminant

	if (disc < 0)
		return false;
	else disc = sqrtf(disc);

	t = b - disc;
	if (t < epsilon)
		t = b + disc;
	if (t < epsilon)
		return false;

	/* calculate hit point and normal to surface */
	hit_point = ray.originPoint + ray.direction * t;
	hit_norm = normalize(hit_point - gdata->position);

	return true;
}

/*__device__
bool testTriangleMeshIntersection(Ray& ray, WorldObject& obj, TriangleMesh& imesh, float3& hit_point, float3& hit_norm) {
	MeshGeometryData* gdata = (MeshGeometryData*)obj.geometry_data;

	int inters_mesh_idx;
	float closest_dist = HUGE_VALF;
	bool intersects = false;
	bool culling = false;

	for (int t = 0; t < gdata->num_triangles; ++t) {
		Triangle& trg = gdata->triangles[t];
		if (culling && dot(ray.direction, trg.norm_a) > 0) // skip triangles turned back to ray
			continue;

		float inters_dist;
		bool triangle_intersects = rayIntersectsTriangle(ray,
			trg.a,
			trg.b,
			trg.c,
			inters_dist);

		if (triangle_intersects && !intersects) {
			intersects = true;
		}
		// check if point is closest to viewer
		if (triangle_intersects && inters_dist < closest_dist) {
			closest_dist = inters_dist;
			hit_norm = trg.norm_a;
		}
	}
	if (intersects) {
		hit_point = ray.originPoint + closest_dist * ray.direction;
	}
	return intersects;
}*/

/* based on: https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection */
__device__
bool testBBoxIntersection(BBox& bbox, Ray& ray) {
	float tmin, tmax, tymin, tymax, tzmin, tzmax;

	tmin = (bbox.bounds[ray.sign[0]].x - ray.originPoint.x) * ray.inv_direction.x;
	tmax = (bbox.bounds[1 - ray.sign[0]].x - ray.originPoint.x) * ray.inv_direction.x;
	tymin = (bbox.bounds[ray.sign[1]].y - ray.originPoint.y) * ray.inv_direction.y;
	tymax = (bbox.bounds[1 - ray.sign[1]].y - ray.originPoint.y) * ray.inv_direction.y;

	if ((tmin > tymax) || (tymin > tmax))
		return false;
	if (tymin > tmin)
		tmin = tymin;
	if (tymax < tmax)
		tmax = tymax;

	tzmin = (bbox.bounds[ray.sign[2]].z - ray.originPoint.z) * ray.inv_direction.z;
	tzmax = (bbox.bounds[1 - ray.sign[2]].z - ray.originPoint.z) * ray.inv_direction.z;

	if ((tmin > tzmax) || (tzmin > tmax))
		return false;
	if (tzmin > tmin)
		tmin = tzmin;
	if (tzmax < tmax)
		tmax = tzmax;

	return true;
}

// Iterate version of recursive ray KDNode intersection
__device__
bool rayIntersectsKDNodeLOOP(Ray& ray, Triangle* trs, KDNode* node, float3& norm, float& tmin, float3& debug_mask) {
	if (node == NULL)
		return false;
	if (!testBBoxIntersection(node->bbox, ray))
		return false;
#ifdef DEBUG_BBOXES
	debug_mask *= 0.95f;
#endif // DEBUG_BBOXES

	bool int_left = false, int_right = false;
	bool left = node->left != NULL, right = node->right != NULL;
	if (left || right) {
		if (node->left != NULL)
			int_left = rayIntersectsKDNodeLOOP(ray, trs, node->left, norm, tmin, debug_mask);
		if (node->right != NULL)
			int_right = rayIntersectsKDNodeLOOP(ray, trs, node->right, norm, tmin, debug_mask);
		return int_left || int_right;
	}
	else { // leaf node of kd tree
		float t;
		bool intersects = false;
		for (int i = 0; i < node->num_trgs; i++)
		{
			Triangle& trg = trs[node->trg_idxs[i]];
			//if (dot(ray.direction, trg.norm_a) >= 0) continue;
			if (rayIntersectsTriangle(ray, trg.a, trg.b, trg.c, t)) {
				intersects = true;
				if (t < tmin) {
					tmin = t;
					norm = trg.norm_a;
				}
			}
		}
		return intersects;
	}
}

__device__
bool rayIntersectsKDNode(Ray& ray, Triangle* trs, KDNode* node, float3& norm, float& tmin, float3& debug_mask) {
	if (node == NULL)
		return false;
	if (!testBBoxIntersection(node->bbox, ray))
		return false;
#ifdef DEBUG_BBOXES
	debug_mask *= 0.95f;
#endif // DEBUG_BBOXES

	bool int_left = false, int_right = false;
	bool left = node->left != NULL, right = node->right != NULL;
	if (left || right) {
		if (node->left != NULL)
			int_left = rayIntersectsKDNode(ray, trs, node->left, norm, tmin, debug_mask);
		if (node->right != NULL)
			int_right = rayIntersectsKDNode(ray, trs, node->right, norm, tmin, debug_mask);
		return int_left || int_right;
	}
	else { // leaf node of kd tree
		float t;
		bool intersects = false;
		for (int i = 0; i < node->num_trgs; i++)
		{
			Triangle& trg = trs[node->trg_idxs[i]];
			//if (dot(ray.direction, trg.norm_a) >= 0) continue;
			if (rayIntersectsTriangle(ray, trg.a, trg.b, trg.c, t)) {
				intersects = true;
				if (t < tmin) {
					tmin = t;
					norm = trg.norm_a;
				}
			}
		}
		return intersects;
	}
}

__device__
bool rayIntersectsObject(Ray& ray, WorldObject& obj, float3& hit_point, float3& hit_norm, float3& debug_mask) {
	bool intersects = false;

	if (obj.type == SphereObj) {
		intersects = testSphereIntersection(ray, obj, hit_point, hit_norm);
	}
	else if (obj.type == TriangleMeshObj) {
		MeshGeometryData* gdata = (MeshGeometryData*)obj.geometry_data;
		float tmin = HUGE_VALF;
#ifdef USE_KD_TREES
		intersects = rayIntersectsKDNode(ray, gdata->triangles, gdata->kdroot, hit_norm, tmin, debug_mask);
#else
		float t;
		int n_idx; //normal index
		for (int i = 0; i < gdata->num_triangles; i++)
		{
			
	#ifdef USE_TRIANGLE_TEXTURE_MEM
			int trg_i = 6 * i;
			float4 a = tex1Dfetch<float4>(gdata->triangles_tex, trg_i);
			float4 b = tex1Dfetch<float4>(gdata->triangles_tex, trg_i + 1);
			float4 c = tex1Dfetch<float4>(gdata->triangles_tex, trg_i + 2);
	#else
				//printf("[%f,%f,%f,%f]\n", a.x, a.y, a.z, a.w);
			Triangle& trg = gdata->triangles[i];
	#endif
				//if (dot(ray.direction, trg.norm_a) >= 0) continue;
	#ifdef USE_TRIANGLE_TEXTURE_MEM
			if (rayIntersectsTriangle(ray, make_float3(a.x, a.y, a.z),
				make_float3(b.x, b.y, b.z),
				make_float3(c.x, c.y, c.z), t)) {
				intersects = true;
				if (t < tmin) {
					tmin = t;
					n_idx = trg_i;
				}
			}
	#else
			if (rayIntersectsTriangle(ray, trg.a, trg.b, trg.c, t)) {
				intersects = true;
				if (t < tmin) {
					tmin = t;
					n_idx = i;
				}
			}
		
	#endif
		}
#endif
		if (intersects)
		{
			hit_point = ray.originPoint + tmin * ray.direction;
			//float4 n = tex1Dfetch<float4>(gdata->triangles_tex, n_idx);
#ifndef USE_KD_TREES
	#ifdef USE_TRIANGLE_TEXTURE_MEM
			float4 n = tex1Dfetch<float4>(gdata->triangles_tex, n_idx + 3);
			hit_norm = make_float3(n.x, n.y, n.z);
	#else
			hit_norm = gdata->triangles[n_idx].norm_a;
	#endif
#endif
		}
	}
	return intersects;
}

__device__
bool rayIntersectsScene(Ray& ray, Scene& scene, int& res_obj_idx, float3& hit_point, float3& hit_norm, float3& debug_mask) {
	bool intersects = false;
	float closest_dist = HUGE_VALF;
	float3 inters_point, inters_norm;

	for (int i = 0; i < scene.num_wobjects; ++i) {
		WorldObject& obj = scene.dv_wobjects_ptr[i];

		if (rayIntersectsObject(ray, obj, inters_point, inters_norm, debug_mask)) {
			intersects = true;
			float inters_dist = length(inters_point - ray.originPoint);
			if (inters_dist < closest_dist) {
				closest_dist = inters_dist;
				res_obj_idx = i;
				hit_point = inters_point;
				hit_norm = inters_norm;
			}
		};
	}
	return intersects;
}
