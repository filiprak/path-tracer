#pragma once

#include "world.h"
#include "cuda_runtime.h"

/* Möller–Trumbore ray-triangle intersection algorithm */
__device__ __host__
bool rayIntersectsTriangle(Ray ray, Triangle triangle, glm::vec3& result)
{
	const float EPSILON = 0.0000001;
	glm::vec3 vertex0 = triangle.a;
	glm::vec3 vertex1 = triangle.b;
	glm::vec3 vertex2 = triangle.c;
	glm::vec3 edge1, edge2, h, s, q;

	float a, f, u, v;
	edge1 = vertex1 - vertex0;
	edge2 = vertex2 - vertex0;

	h = glm::cross(ray.direction, edge2);
	a = glm::dot(edge1, h);

	if (a > -EPSILON && a < EPSILON)
		return false;
	f = 1 / a;
	s = ray.originPoint - vertex0;
	u = f * (glm::dot(s, h));
	if (u < 0.0 || u > 1.0)
		return false;
	q = glm::cross(s, edge1);
	v = f * glm::dot(ray.direction, q);
	if (v < 0.0 || u + v > 1.0)
		return false;
	// At this stage we can compute t to find out where the intersection point is on the line.
	float t = f * glm::dot(edge2, q);
	if (t > EPSILON) // ray intersection
	{
		result = ray.direction + (glm::normalize(ray.direction) * (t * glm::length(ray.direction) ));
		return true;
	}
	else // This means that there is a line intersection but not a ray intersection.
		return false;
}

__device__ __host__
bool rayIntersectsObject(Ray ray, WorldObject obj, glm::vec3& result) {
	int indices_len = sizeof(obj.indices);

	glm::vec3 closest_inters_point;
	float closest_dist = HUGE_VALF;

	bool intersects = false;

	for (int i = 0; i < 36; i = i + 3) {
		Triangle t;
		t.a = obj.vertices[obj.indices[i]];
		t.b = obj.vertices[obj.indices[i + 1]];
		t.c = obj.vertices[obj.indices[i + 2]];
		
		/*printf("Testing intersection: \ntriangle:\n[%f, %f, %f\n%f, %f, %f\n%f, %f, %f]\n x \nray[%f, %f, %f]",
			t.a.x, t.a.y, t.a.z, t.b.x, t.b.y, t.b.z, t.c.x, t.c.y, t.c.z, ray.originPoint.x, ray.originPoint.x, ray.originPoint.x);*/

		glm::vec3 inters_point;
		bool triangle_intersects = rayIntersectsTriangle(ray, t, inters_point);
		if (triangle_intersects)
			intersects = true;
		// check if point is closest to viewer
		if (triangle_intersects && glm::length(inters_point - ray.originPoint) < closest_dist) {
			closest_dist = (inters_point - ray.originPoint).length();
			closest_inters_point = inters_point;
			//printf("Intersects triangle: [%d, %d, %d]\n", obj.indices[i], obj.indices[i + 1], obj.indices[i + 2]);
		}
	}

	result = closest_inters_point;
	return intersects;
}