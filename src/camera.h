#pragma once

#include <glm\glm.hpp>
#include "vector_types.h"

/* Camera structure */
typedef struct {
	// init params
	float3 init_dir, init_up, init_right, init_pos;

	// current params
	float3 position;
	float3 direction;
	float3 up;
	float3 right;

	float h_ang, v_ang;

	struct {
		int width, height;
		float viewer_dist;

	} projection;

	bool changed;

} Camera;


void initCamera();
void resetCamera();

void moveCamera(float3);

void rotateVCamera(float);
void rotateHCamera(float);

void printCamInfo();
