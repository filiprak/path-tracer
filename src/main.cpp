#include "main.h"
#include "view.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

int main() {
	printf("Starting path-tracer application.\n");

	Assimp::Importer imp;

	const aiScene* scene = imp.ReadFile("teapot.obj",
		aiProcess_CalcTangentSpace |
		aiProcess_Triangulate |
		aiProcess_JoinIdenticalVertices |
		aiProcess_SortByPType);
	
	if (scene)
		printf("Loaded scene from file using assimp.");

	viewInit(300, 300);

	viewLoop();

	exit(0);
}