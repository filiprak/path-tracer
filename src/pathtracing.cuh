#pragma once

#include "world.h"
#include "camera.h"
#include "cuda_runtime.h"


__global__
void generatePrimaryRays(Camera, Ray*, float3*);

__global__
void traceRays(Camera, WorldObject*, Ray*, float3*);

__host__
void initPathTracing();

__host__
void cleanUpPathTracing();

__host__
void runPathTracing(int, int);

__device__
inline void init_ray(Ray&, float3&, float3&);