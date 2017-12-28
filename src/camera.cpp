#include "camera.h"
#include "world.h"
#include "config.h"

#include "cutil_math.h"
#include "glm/glm.hpp"
#include <glm/gtx/rotate_vector.hpp>

#include "constants.h"
#include "stdio.h"
#include "jsonResolve.h"



void Camera::init(const Json::Value& jcam) {
	init_dir = normalize(resolveFloat3(jcam["direction"]));
	init_up = normalize(resolveFloat3(jcam["up"]));
	init_pos = resolveFloat3(jcam["position"]);
	init_right = normalize(cross(init_dir, init_up));

	position = init_pos;
	direction = init_dir;
	up = init_up;
	right = init_right;

	h_ang = v_ang = 0.0f;

	int viewWidth = jcam["pixelWidth"].asInt();
	int viewHeight = jcam["pixelHeight"].asInt();

	projection.width = viewWidth;
	projection.height = viewHeight;
	projection.num_pixels = viewWidth * viewHeight;
	projection.screen_dist = resolveFloat(jcam["screenDist"]);

	float screenHeight = resolveFloat(jcam["screenHeight"]);
	projection.pixel_size = make_float2(screenHeight / (float)viewHeight);
	projection.screen_halfsize.y = screenHeight / 2.0f;
	projection.screen_halfsize.x = projection.screen_halfsize.y * (float)viewWidth / (float)viewHeight;
	projection.aa_jitter = 2.0f;
	projection.gamma_corr = 0.5f;

	preview_mode = true;
	aabb_mode = false;
	texture_enabled = false;
	changed = false;
	max_ray_bounces = MAX_NUM_RAY_BOUNCES;
}

void Camera::reset() {
	position = init_pos;
	direction = init_dir;
	up = init_up;
	right = init_right;
	h_ang = v_ang = 0.0f;
	changed = true;
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

void Camera::refresh() {
	direction = normalize(rotate_float3(init_dir, degToRad(v_ang), init_right));
	direction = normalize(rotate_float3(direction, degToRad(h_ang), init_up));
	up = normalize(rotate_float3(init_up, degToRad(v_ang), init_right));
	up = normalize(rotate_float3(up, degToRad(h_ang), init_up));
	right = normalize(rotate_float3(init_right, degToRad(v_ang), init_right));
	right = normalize(rotate_float3(right, degToRad(h_ang), init_up));
	changed = true;
}

void Camera::move(float3 diff) {
	position += diff;
	changed = true;
}

void Camera::rotateV(float degrees) {
	v_ang += degrees;
	refresh();
}

void Camera::rotateH(float degrees) {
	h_ang += degrees;
	refresh();
}

void Camera::updateMaxRayBnc(int delta) {
	int newv = max_ray_bounces + delta;
	if (newv < 0)
		return;
	max_ray_bounces = newv;
	changed = true;
}

void Camera::updateAajitter(float delta) {
	float aaj = projection.aa_jitter + delta;
	if (aaj < 0.0)
		return;
	projection.aa_jitter = aaj;
	changed = true;
}

void Camera::updateGamma(float delta) {
	float g = projection.gamma_corr + delta;
	if (g <= 0.0)
		return;
	projection.gamma_corr = g;
}

void Camera::togglePrevMode() {
	preview_mode = !preview_mode;
	changed = true;
}

void Camera::toggleTextures() {
	texture_enabled = !texture_enabled;
	changed = true;
}

void Camera::printInfo() {
	printf("\nCamera Info -------------------\n");
	printf("> position:  [%.3f,%.3f,%.3f]\n", position.x, position.y, position.z);
	printf("> direction: [%.3f,%.3f,%.3f]\n", direction.x, direction.y, direction.z);
	printf("> up:        [%.3f,%.3f,%.3f]\n", up.x, up.y, up.z);
	printf("> right:     [%.3f,%.3f,%.3f]\n", right.x, right.y, right.z);
	printf("> v_ang:            %.3f\n", v_ang);
	printf("> h_ang:            %.3f\n", h_ang);
	printf("> max_ray_bounces:  %d\n", max_ray_bounces);
	printf("> aa ray jitter:    %.3f\n", projection.aa_jitter);
	printf("> image gamma:      %.3f\n", projection.gamma_corr);
	/* debug
	printf("> dot(dir,up):        %f\n", dot(direction, up));
	printf("> dot(dir,right):     %f\n", dot(direction, right));
	printf("> dot(right,up):      %f\n", dot(right, up));*/
}