#pragma once

#include "world.h"

/* Möller–Trumbore ray-triangle intersection algorithm */
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