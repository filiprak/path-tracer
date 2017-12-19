#pragma once
#include "world.h"
#include "dialoglogger.h"
#include <string>

/* glm */
#include <glm/glm.hpp>


#define MAX_OBJECTS_NUM		32
#define OBJECTS_TYPES_NUM	2

void loadSceneWorldObjects(Scene&, const Json::Value&, DialogLogger*);
bool parseJsonScene(std::string, Json::Value&, std::string&);

void freeWorldObjects();

bool loadSphereObj(void*, WorldObject&);

bool loadTriangleMeshObj(void*, WorldObject&);

// kd tree functions
KDNode* copyKDTreeToCuda(KDNode*);
void freeKDTree(KDNode*);
void freeCudaKDTree(KDNode*);
KDNode* buildKDTree(Triangle*, int*, int, int, int&, int&);
void flatenKDTree(KDNode*, KDNode*);

typedef struct {
	WorldObjType type;
	void* worldObjectInfo;

} WorldObjectDescription;


typedef struct {
	WorldObjectDescription sources[MAX_OBJECTS_NUM];
	int num_objects;

	// dont reorder values, because they correspond enum types
	bool(*loadFuncMapping[OBJECTS_TYPES_NUM])(void*, WorldObject&);

} WorldObjectsSources;

/* object infos */
typedef struct {
	float radius;
	float3 position;

	Material material;

} SphereObjInfo;

typedef struct {
	char src_filename[256];

	glm::mat4 transform;

} TriangleMeshObjInfo;


extern WorldObjectsSources world_obj_sources;