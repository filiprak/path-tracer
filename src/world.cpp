#include "world.h"
#include "view.h"
#include "camera.h"

// 3D rendering scene global object
Scene scene;


void sceneInit() {
	initCamera();
}

void worldInit() {
	sceneInit();
}