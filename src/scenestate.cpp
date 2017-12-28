#include "scenestate.h"
#include "world_load.h"
#include "cudaUtility.h"

// scene state global object - wrapper for scene data handling
SceneState sceneState;


SceneState::SceneState() : loaded(false)
{

}


SceneState::~SceneState()
{
	this->clean();
}

bool SceneState::load(const Json::Value& jscene, DialogLogger* logger) {
	mutex.lock();
	freeWorldObjects();
	this->loaded = loadSceneWorldObjects(this->scene, jscene, logger);
	mutex.unlock();
	return this->loaded;
}

bool SceneState::isLoaded() {
	return this->loaded;
}
bool SceneState::clean() {
	mutex.lock();
	freeWorldObjects();
	this->loaded = false;
	mutex.unlock();
	return !checkCudaError("freeWorldObjects()");
}

Scene SceneState::clone() {
	mutex.lock();
	Scene cloned = this->scene;
	mutex.unlock();
	return cloned;
}

void SceneState::cameraChanged(bool flag, int* iter) {
	mutex.lock();
	if (iter && scene.camera.changed)
		*iter = 1;
	scene.camera.changed = flag;
	mutex.unlock();
}


void SceneState::resetCamera() {
	mutex.lock();
	scene.camera.reset();
	mutex.unlock();
}
void SceneState::setCamPositionX(float x) {
	mutex.lock();
	scene.camera.position.x = x;
	scene.camera.changed = true;
	mutex.unlock();
}
void SceneState::setCamPositionY(float y) {
	mutex.lock();
	scene.camera.position.y = y;
	scene.camera.changed = true;
	mutex.unlock();
}
void SceneState::setCamPositionZ(float z) {
	mutex.lock();
	scene.camera.position.z = z;
	scene.camera.changed = true;
	mutex.unlock();
}
void SceneState::setCamVang(float ang) {
	mutex.lock();
	scene.camera.v_ang = ang;
	scene.camera.refresh();
	mutex.unlock();
}
void SceneState::setCamHang(float ang) {
	mutex.lock();
	scene.camera.h_ang = ang;
	scene.camera.refresh();
	mutex.unlock();
}
void SceneState::updateMaxRayBnc(int val) {
	mutex.lock();
	scene.camera.max_ray_bounces = val;
	scene.camera.changed = true;
	mutex.unlock();
}
void SceneState::updateAajitter(float val) {
	if (val < 0.0f)
		return;
	mutex.lock();
	scene.camera.projection.aa_jitter = val;
	scene.camera.changed = true;
	mutex.unlock();
}
void SceneState::updateGamma(float gamma) {
	if (gamma < 0.0f)
		return;
	mutex.lock();
	scene.camera.projection.gamma_corr = gamma;
	mutex.unlock();
}
void SceneState::togglePrevMode(bool val) {
	mutex.lock();
	scene.camera.preview_mode = val;
	scene.camera.changed = true;
	mutex.unlock();
}
void SceneState::toggleAABBmode(bool val) {
	mutex.lock();
	scene.camera.aabb_mode = val;
	scene.camera.changed = true;
	mutex.unlock();
}
void SceneState::toggleTextures(bool val) {
	mutex.lock();
	scene.camera.texture_enabled = val;
	scene.camera.changed = true;
	mutex.unlock();
}