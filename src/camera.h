#pragma once

#include <glm\glm.hpp>
#include "vector_types.h"

/* Camera structure */
typedef struct {
	float3 position;
	float3 direction;
	float3 up;
	float3 right;

	struct {
		int width, height;
		float viewer_dist;

	} projection;

	bool changed;

} Camera;


void initCamera();

void moveCamera(float3 diff);

void rotateVCamera(float degrees);
void rotateHCamera(float degrees);
