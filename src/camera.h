#pragma once

#include <glm/glm.hpp>
#include "vector_types.h"
#include "main.h"

/* Camera structure */
typedef struct {
	// init params
	float3 init_dir, init_up, init_right, init_pos;

	// current params
	float3 position;
	float3 direction;
	float3 up;
	float3 right;

	// angles of view
	float h_ang, v_ang;

	// projection of image
	struct {
		int width, height;
		int num_pixels;
		float2 pixel_size;
		float2 screen_halfsize;
		float screen_dist;

		float aa_jitter;

	} projection;

	// set true when params change
	bool changed;

	// mesh preview mode
	bool preview_mode;
	bool texture_enabled;

	// max ray bounces number
	int max_ray_bounces;

} Camera;


void initCamera(const Json::Value&);
void resetCamera();

void moveCamera(float3);

void rotateVCamera(float);
void rotateHCamera(float);

void updateMaxRayBnc(int);
void updateAajitter(float);
void togglePrevMode();
void toggleTextures();

void printCamInfo();
