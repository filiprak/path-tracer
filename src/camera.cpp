#include "camera.h"
#include "world.h"
#include "view.h"
#include "config.h"

#include "cutil_math.h"
#include "glm/glm.hpp"
#include <glm/gtx/rotate_vector.hpp>

#include "constants.h"
#include "stdio.h"
#include "jsonResolve.h"


void initCamera(const Json::Value& jcam) {
	Camera& c = scene.camera;

	c.init_dir = normalize(resolveFloat3(jcam["direction"]));
	c.init_up = normalize(resolveFloat3(jcam["up"]));
	c.init_pos = resolveFloat3(jcam["position"]);
	c.init_right = normalize(cross(c.init_dir, c.init_up));

	c.position = c.init_pos;
	c.direction = c.init_dir;
	c.up = c.init_up;
	c.right = c.init_right;

	c.h_ang = c.v_ang = 0.0f;

	int viewWidth = jcam["pixelWidth"].asInt();
	int viewHeight = jcam["pixelHeight"].asInt();

	c.projection.width = viewWidth;
	c.projection.height = viewHeight;
	c.projection.num_pixels = viewWidth * viewHeight;
	c.projection.screen_dist = resolveFloat(jcam["screenDist"]);

	float screenHeight = resolveFloat(jcam["screenHeight"]);
	c.projection.pixel_size = make_float2(screenHeight / (float)viewHeight);
	c.projection.screen_halfsize.y = screenHeight / 2.0f;
	c.projection.screen_halfsize.x = c.projection.screen_halfsize.y * (float)viewWidth / (float)viewHeight;
	c.projection.aa_jitter = 2.0f;

	c.preview_mode = true;
	c.texture_enabled = false;
	c.changed = false;
	c.max_ray_bounces = MAX_NUM_RAY_BOUNCES;
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

void updateMaxRayBnc(int delta) {
	Camera& c = scene.camera;
	int newv = c.max_ray_bounces + delta;
	if (newv < 0)
		return;
	c.max_ray_bounces = newv;
	c.changed = true;
}

void updateAajitter(float delta) {
	Camera& c = scene.camera;
	float aaj = c.projection.aa_jitter + delta;
	if (aaj < 0.0)
		return;
	c.projection.aa_jitter = aaj;
	c.changed = true;
}

void togglePrevMode() {
	scene.camera.preview_mode = !scene.camera.preview_mode;
	scene.camera.changed = true;
}

void toggleTextures() {
	scene.camera.texture_enabled = !scene.camera.texture_enabled;
	scene.camera.changed = true;
}

void printCamInfo() {
	Camera& c = scene.camera;
	printf("\nCamera Info -------------------\n");
	printf("> position:  [%.3f,%.3f,%.3f]\n", c.position.x, c.position.y, c.position.z);
	printf("> direction: [%.3f,%.3f,%.3f]\n", c.direction.x, c.direction.y, c.direction.z);
	printf("> up:        [%.3f,%.3f,%.3f]\n", c.up.x, c.up.y, c.up.z);
	printf("> right:     [%.3f,%.3f,%.3f]\n", c.right.x, c.right.y, c.right.z);
	printf("> v_ang:            %.3f\n", c.v_ang);
	printf("> h_ang:            %.3f\n", c.h_ang);
	printf("> max_ray_bounces:  %d\n", c.max_ray_bounces);
	printf("> aa ray jitter:    %.3f\n", c.projection.aa_jitter);
	/* debug
	printf("> dot(dir,up):        %f\n", dot(c.direction, c.up));
	printf("> dot(dir,right):     %f\n", dot(c.direction, c.right));
	printf("> dot(right,up):      %f\n", dot(c.right, c.up));*/
}