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

	c.init_dir = normalize(make_float3(0.0, 0.0, -1.0));
	c.init_up = normalize(make_float3(0.0, 1.0, 0.0));
	c.init_pos = make_float3(0.0, 3.0, 15.0);
	c.init_right = normalize(cross(c.init_up, c.init_dir));

	c.position = c.init_pos;
	c.direction = c.init_dir;
	c.up = c.init_up;
	c.right = c.init_right;

	c.h_ang = c.v_ang = 0.0f;

	c.projection.width = viewWidth;
	c.projection.height = viewHeight;
	c.projection.viewer_dist = 3.0f;

	c.changed = false;
}

void resetCamera() {
	Camera& c = scene.camera;
	c.position = c.init_pos;
	c.direction = c.init_dir;
	c.up = c.init_up;
	c.right = c.init_right;
	c.h_ang = c.v_ang = 0.0f;
	c.changed = true;
}

float3 rotate_float3(float3 vec, float radians, float3 normal) {
	if (fabs(dot(vec, normal)) == 1.0f)
		return vec;
	glm::mat4 rotmat;
	rotmat = glm::rotate(rotmat, radians, glm::vec3(normal.x, normal.y, normal.z));
	glm::vec4 res = rotmat * glm::vec4(vec.x, vec.y, vec.z, 1.0f);
	return make_float3(res.x, res.y, res.z);
}

float degToRad(float degrees) {
	return (float)(degrees / 180.0) * PI_f;
}

void refreshCamera() {
	Camera& c = scene.camera;
	c.direction = normalize(rotate_float3(c.init_dir, degToRad(c.v_ang), c.init_right));
	c.direction = normalize(rotate_float3(c.direction, degToRad(c.h_ang), c.init_up));
	c.up = normalize(rotate_float3(c.init_up, degToRad(c.v_ang), c.init_right));
	c.up = normalize(rotate_float3(c.up, degToRad(c.h_ang), c.init_up));
	c.right = normalize(rotate_float3(c.init_right, degToRad(c.v_ang), c.init_right));
	c.right = normalize(rotate_float3(c.right, degToRad(c.h_ang), c.init_up));
	c.changed = true;
}

void moveCamera(float3 diff) {
	scene.camera.position += diff;
	scene.camera.changed = true;
}

void rotateVCamera(float degrees) {
	Camera& c = scene.camera;
	c.v_ang += degrees;
	refreshCamera();
}

void rotateHCamera(float degrees) {
	Camera& c = scene.camera;
	c.h_ang += degrees;
	refreshCamera();
}



void printCamInfo() {
	Camera& c = scene.camera;
	printf("\nCamera Info -------------------\n");
	printf("> position:  [%.3f,%.3f,%.3f]\n", c.position.x, c.position.y, c.position.z);
	printf("> direction: [%.3f,%.3f,%.3f]\n", c.direction.x, c.direction.y, c.direction.z);
	printf("> up:        [%.3f,%.3f,%.3f]\n", c.up.x, c.up.y, c.up.z);
	printf("> right:     [%.3f,%.3f,%.3f]\n", c.right.x, c.right.y, c.right.z);
	printf("> v_ang:     %f\n", c.v_ang);
	printf("> h_ang:     %f\n", c.h_ang);
	/* debug
	printf("> dot(dir,up):        %f\n", dot(c.direction, c.up));
	printf("> dot(dir,right):     %f\n", dot(c.direction, c.right));
	printf("> dot(right,up):      %f\n", dot(c.right, c.up));*/
}