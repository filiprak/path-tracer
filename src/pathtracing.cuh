#pragma once

#include "world.h"
#include "camera.h"
#include "cuda_runtime.h"


__global__
void generatePrimaryRays(Camera, Ray*, float3*);

__global__
void traceRays(Camera, WorldObject*, Ray*, float3*);

__host__
void initPathTracing(const Scene&);

__host__
void cleanUpPathTracing();

__host__
void runPathTracing(Scene&, int, int);

__device__
inline void init_ray(Ray*, const float3&, const float3&);