#pragma once

#include "world.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "cutil_math.h"

#define EPSILON		0.0000001

/* Möller–Trumbore ray-triangle intersection algorithm */
__device__
bool rayIntersectsTriangle(Ray& ray, float3& va, float3& vb, float3& vc, float3& result)
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
		result = ray.originPoint + normalize(ray.direction) * (t * (length(ray.direction)));
		return true;
	}
	// This means that there is a line intersection but not a ray intersection.
	return false;
}

__device__
bool rayIntersectsObject(Ray& ray, WorldObject& obj, float3& result, Triangle& tres) {
	int indices_len = sizeof(obj);

	float3 closest_inters_point;
	float closest_dist = HUGE_VALF;

	bool intersects = false;
	Triangle inters_tr;
	//printf("Testing intersection with meshes: %d\n", obj.num_meshes);

	for (int m = 0; m < obj.num_meshes; ++m) {
		TriangleMesh& mesh = obj.meshes[m];
		Triangle* triangles = mesh.triangles;

		for (int t = 0; t < mesh.num_triangles; ++t) {
			Triangle& trg = triangles[t];
			/*printf("Testing intersection: triangle:[%f,%f,%f] X ray[%f, %f, %f]\n",
				trg.a.x, trg.b.y, trg.c.z, ray.direction.x, ray.direction.y, ray.direction.z);*/
			
			if (dot(normalize(ray.direction), normalize(trg.norm_a)) > -0.00001)
				continue;

			float3 inters_point;
			bool triangle_intersects = rayIntersectsTriangle(ray,
				trg.a,
				trg.b,
				trg.c,
				inters_point);

			if (triangle_intersects && !intersects) {
				/*printf("Ray collision: triangle:[%d] X ray[%f, %f, %f]\n",
					t, ray.direction.x, ray.direction.y, ray.direction.z);*/
				intersects = true;
			}
			// check if point is closest to viewer
			float dist = length(inters_point - ray.originPoint);
			if (triangle_intersects && dist < closest_dist) {
				closest_dist = dist;
				closest_inters_point = inters_point;
				inters_tr = trg;
			}
		}
	}

	result = closest_inters_point;
	tres = inters_tr;
	return intersects;
}