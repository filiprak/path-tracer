#pragma once

#include "camera.h"
#include <glm\glm.hpp>


/* Material types */
typedef enum {
	Light,
	Reflecting,
	Diffuse

} Material;

typedef struct {
	float3 a,b,c;
	float3 norm_a, norm_b, norm_c;

} Triangle;

typedef struct {
	Triangle* triangles;
	int num_triangles;

} TriangleMesh;

/* WorldObject structure */
typedef struct {
	Material materialType;

	
	glm::mat4 transformMat;
	glm::mat4 inversedTransMat;

	TriangleMesh* meshes;
	int num_meshes;

} WorldObject;

/* Scene structure */
typedef struct {
	Camera camera;
	WorldObject* dv_wobjects_ptr;
	int num_wobjects;

} Scene;


/* Ray structure */
typedef struct {
	float3 originPoint;
	float3 direction;

	bool active;
	float3 color;

} Ray;


void worldInit();


// global scene object
extern Scene scene;