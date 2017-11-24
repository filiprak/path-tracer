#include "world.h"
#include "world_load.h"
#include "view.h"
#include "camera.h"

// 3D rendering scene global object
Scene scene;


void sceneInit(const Json::Value& jcam) {
	initCamera(jcam);
}

void worldInit(const Json::Value& jscene) {
	sceneInit(jscene["camera"]);
	loadSceneWorldObjects(jscene);
}

void worldCleanUp() {
	freeWorldObjects();
}