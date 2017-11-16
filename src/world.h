#pragma once

#include "main.h"
#include "camera.h"
#include "KDTree.h"
#include <glm\glm.hpp>


/* World Object types, important: dont reorder values */
typedef enum {
	SphereObj = 0,
	TriangleMeshObj

} WorldObjType;

/* Material types */
typedef enum {
	Luminescent = 0,
	Reflective,
	Refractive,
	Diffusing

} MaterialType;



typedef struct alignMem(16) {
	MaterialType type;
	float3 color, norm_color;  // mtl: Ka

	// for light materials
	float3 emittance; // mtl: Ke

	// reflective materials
	float reflect_factor; // mtl: d

	// refractive materials
	float refract_index; // mtl: Ni

} Material;


typedef struct alignMem(16) {
	float3 a,b,c;
	float3 norm_a, norm_b, norm_c;

} Triangle;


typedef struct alignMem(16) {
	Triangle* triangles;
	int num_triangles;

	Material material;

} TriangleMesh;

// Axis aligned bounding boxes AABB
typedef struct alignMem(16) {
	float3 bounds[2]; // minimal and maximal volume point
	// bounds[0] - min, bounds[1] - max
} BBox;


typedef alignMem(16) struct KDNode {
	// Bounding box of the tree node
	BBox bbox;

	// Triangle indexes of stored in current node
	int* trg_idxs;
	int num_trgs;

	// child pointers
	KDNode* left;
	KDNode*	right;

} KDNode;


typedef struct alignMem(16) {
	Triangle* triangles;
	KDNode* kdroot;
	int num_triangles;

} MeshGeometryData;


typedef struct alignMem(16) {
	float radius;
	float3 position;

} SphereGeometryData;


/* WorldObject structure */
typedef struct alignMem(16) {
	WorldObjType type;
	Material material;
	void* geometry_data;

} WorldObject;


/* Scene structure */
typedef struct {
	WorldObject* dv_wobjects_ptr;
	int num_wobjects;
	Camera camera;
} Scene;


/* Ray structure */
typedef struct {
	float3 originPoint;
	float3 direction;

	// precomputed inverse of direction for faster AABB boxes intersecting
	float3 inv_direction;
	int sign[3]; // inverse rays coordinates signs
} Ray;


void worldInit();

// global scene object
extern Scene scene;