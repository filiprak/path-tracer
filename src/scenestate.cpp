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
	loadSceneWorldObjects(this->scene, jscene, logger);
	this->loaded = true;
	mutex.unlock();
	return true;
}

bool SceneState::isLoaded() {
	mutex.lock();
	bool res = this->loaded;
	mutex.unlock();
	return res;
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