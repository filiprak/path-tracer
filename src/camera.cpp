#include "camera.h"
#include "world.h"
#include "view.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/rotate_vector.hpp>


void initCamera() {
	Camera& c = scene.camera;

	c.direction = glm::normalize(glm::vec3(0.0, 0.0, -1.0));
	c.position = glm::vec3(0.0, 0.0, 7.0);
	c.up = glm::normalize(glm::vec3(0.0, 1.0, 0.0));

	c.projection.width = viewWidth;
	c.projection.height = viewHeight;
	c.projection.viewer_dist = 3.0f;

	c.viewMat = glm::lookAt(c.position, c.position + c.direction, c.up);
	c.changed = false;
}

void moveCamera(glm::vec3 diff) {
	scene.camera.position += diff;
	scene.camera.changed = true;
}

void rotateVCamera(float degrees) {
	Camera& c = scene.camera;
	glm::vec3 rotaxis = glm::normalize(glm::cross(c.direction, c.up));
	c.direction = glm::normalize(glm::rotate(c.direction, glm::radians(degrees), rotaxis));
	c.up = glm::normalize(glm::rotate(c.up, glm::radians(degrees), rotaxis));
	c.changed = true;
}

void rotateHCamera(float degrees) {
	Camera& c = scene.camera;
	glm::vec3 rotaxis = glm::normalize(c.up);
	c.direction = glm::normalize(glm::rotate(c.direction, glm::radians(degrees), rotaxis));
	c.changed = true;
}