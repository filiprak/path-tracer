#pragma once

#include "main.h"
#include "camera.h"
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
	Diffusing

} MaterialType;

typedef struct alignMem(16) {
	MaterialType type;
	float3 color, norm_color;
	float3 emittance;

} Material;


typedef struct alignMem(16) {
	float3 a,b,c;
	float3 norm_a, norm_b, norm_c;

} Triangle;

typedef struct alignMem(16) {
	Triangle* triangles;
	int num_triangles;

} TriangleMesh;

typedef struct alignMem(16) {
	TriangleMesh* meshes;
	int num_meshes;

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
typedef struct alignMem(16) {
	float3 originPoint;
	float3 direction;

	bool active;

} Ray;


void worldInit();


// global scene object
extern Scene scene;