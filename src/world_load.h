#pragma once
#include "cuda_runtime.h"
#include "world.h"
#include <string>

/* glm */
#include <glm/glm.hpp>


#define MAX_OBJECTS_NUM		32
#define OBJECTS_TYPES_NUM	2

__host__
void loadSceneWorldObjects();

__host__
void freeWorldObjects();

__host__
bool loadSphereObj(void*, WorldObject&);

__host__
bool loadTriangleMeshObj(void*, WorldObject&);

// kd tree functions
KDNode* copyKDTreeToCuda(KDNode*);
void freeKDTree(KDNode*);
void freeCudaKDTree(KDNode*);
KDNode* buildKDTree(Triangle*, int*&, int, int);


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
	Material material;
	char src_filename[128];

	glm::mat4 transform;

} TriangleMeshObjInfo;


extern WorldObjectsSources world_obj_sources;