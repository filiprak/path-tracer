#pragma once

#include <glm\glm.hpp>


/* Camera structure */
typedef struct {
	glm::vec3 position;
	glm::vec3 direction;
	glm::vec3 up;

	glm::mat4 viewMat;

	struct {
		int width, height;
		float viewer_dist;

	} projection;

	bool changed;

} Camera;


void initCamera();

void moveCamera(glm::vec3 diff);

void rotateVCamera(float degrees);
void rotateHCamera(float degrees);
