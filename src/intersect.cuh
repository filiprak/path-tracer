#pragma once

#include "world.h"
#include "cuda.h"
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
	float t, epsilon = 0.01f;
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

__device__
bool testTriangleMeshIntersection(Ray& ray, WorldObject& obj, float3& hit_point, float3& hit_norm) {
	MeshGeometryData* gdata = (MeshGeometryData*)obj.geometry_data;

	float closest_dist = HUGE_VALF;
	float res_hit_angle_cos;
	bool intersects = false;

	for (int m = 0; m < gdata->num_meshes; ++m) {
		TriangleMesh& mesh = gdata->meshes[m];
		Triangle* triangles = mesh.triangles;

		for (int t = 0; t < mesh.num_triangles; ++t) {
			Triangle& trg = triangles[t];
			res_hit_angle_cos = dot(normalize(ray.direction), trg.norm_a);
			if (res_hit_angle_cos > -0.00001) // skip triangles turned back from camera
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
	}
	if (intersects) {
		hit_point = ray.originPoint + closest_dist * ray.direction;
	}
	return intersects;
}

__device__
bool rayIntersectsObject(Ray& ray, WorldObject& obj, float3& hit_point, float3& hit_norm) {
	bool intersects = false;

	if (obj.type == SphereObj) {
		intersects = testSphereIntersection(ray, obj, hit_point, hit_norm);
	}
	else if (obj.type == TriangleMeshObj) {
		intersects = testTriangleMeshIntersection(ray, obj, hit_point, hit_norm);
	}

	return intersects;
}

__device__
bool rayIntersectsScene(Ray& ray, Scene& scene, int& res_obj_idx, float3& hit_point, float3& hit_norm) {
	bool intersects = false;
	float closest_dist = HUGE_VALF;
	float3 inters_point, inters_norm;

	for (int i = 0; i < scene.num_wobjects; ++i) {
		WorldObject& obj = scene.dv_wobjects_ptr[i];

		if (rayIntersectsObject(ray, obj, inters_point, inters_norm)) {
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
