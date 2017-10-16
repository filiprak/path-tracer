#include "world.h"
#include "view.h"
#include <glm/gtc/matrix_transform.hpp>


Camera createCamera() {
	Camera cam;

	cam.direction = glm::normalize(glm::vec3(0.0, 0.0, -1.0));
	cam.position = glm::vec3(0.0, 0.0, 10.0);
	cam.up = glm::vec3(0.0, 1.0, 0.0);

	cam.projection.width = viewWidth;
	cam.projection.height = viewHeight;
	cam.projection.viewer_dist = 5.0f;

	cam.viewMat = glm::lookAt(cam.position, cam.position + cam.direction, cam.up);

	return cam;
}