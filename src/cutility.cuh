#include "stdio.h"
#include "world.h"
#include "cutil_math.h"


inline unsigned int utilhash(unsigned int seed) {
	seed = (seed + 0x7ed55d16) + (seed << 12);
	seed = (seed ^ 0xc761c23c) ^ (seed >> 19);
	seed = (seed + 0x165667b1) + (seed << 5);
	seed = (seed + 0xd3a2646c) ^ (seed << 9);
	seed = (seed + 0xfd7046c5) + (seed << 3);
	seed = (seed ^ 0xb55a4f09) ^ (seed >> 16);
	return seed;
}

inline unsigned int wang_hash(unsigned int seed)
{
	seed = (seed ^ 61) ^ (seed >> 16);
	seed *= 9;
	seed = seed ^ (seed >> 4);
	seed *= 0x27d4eb2d;
	seed = seed ^ (seed >> 15);
	return seed;
}

__device__
void print_float3(float3 v)
{
	printf("Wektor((%.2f,%.2f,%.2f)),", v.x, v.y, v.z);
}

__device__
void print_float3(float3 vs, float3 ve)
{
	printf("Wektor((%.2f,%.2f,%.2f),(%.2f,%.2f,%.2f)),", vs.x, vs.y, vs.z, ve.x, ve.y, ve.z);
}

__device__
void print_ray(Ray ray)
{
	print_float3(ray.originPoint, ray.originPoint + ray.direction);
}