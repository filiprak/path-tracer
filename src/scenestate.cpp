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