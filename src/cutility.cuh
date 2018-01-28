#include "stdio.h"
#include "world.h"
#include "cutil_math.h"


/* origin: http://www.burtleburtle.net/bob/hash/integer.html */
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