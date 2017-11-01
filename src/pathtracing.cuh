#pragma once

#include "world.h"
#include "camera.h"
#include "cuda_runtime.h"


__global__
void generatePrimaryRays(Camera cam, Ray* rays);

__global__
void traceRays(Camera cam, WorldObject* wobjs, Ray* primary_rays, float3* image);

__host__
void initPathTracing();

__host__
void cleanUpPathTracing();

__host__
void runPathTracing(int iterHash);