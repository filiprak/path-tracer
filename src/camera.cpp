#include "camera.h"
#include "world.h"
#include "view.h"

#include "cutil_math.h"
#include "glm/glm.hpp"
#include <glm/gtx/rotate_vector.hpp>

#include "constants.h"
#include "stdio.h"


void initCamera() {
	Camera& c = scene.camera;

	c.direction = normalize(make_float3(0.0, 0.0, -1.0));
	c.position = make_float3(1.0, 3.0, 12.0);
	c.up = normalize(make_float3(0.0, 1.0, 0.0));
	c.right = normalize(cross(c.up, c.direction));

	c.projection.width = viewWidth;
	c.projection.height = viewHeight;
	c.projection.viewer_dist = 3.0f;

	c.changed = false;
}

float3 rotate_float3(float3 vec, float radians, float3 normal) {
	return cosf(radians) * vec + sinf(radians) * cross(normalize(normal), vec);
}

float degToRad(float degrees) {
	return (degrees / 180.0) * PI_f;
}

void moveCamera(float3 diff) {
	scene.camera.position += diff;
	scene.camera.changed = true;
}

void rotateVCamera(float degrees) {
	Camera& c = scene.camera;
	c.direction = normalize(rotate_float3(c.direction, degToRad(degrees), c.right));
	c.up = normalize(cross(c.direction, c.right));
	c.changed = true;
}

void rotateHCamera(float degrees) {
	Camera& c = scene.camera;
	c.direction = normalize(rotate_float3(c.direction, degToRad(degrees), c.up));
	c.right = normalize(cross(c.up, c.direction));
	c.changed = true;
}