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
	glm::vec3 a, b, c;
} Triangle;

/* Mesh structure - default cube mesh */
typedef struct {
	Material materialType;

	glm::mat4 transformMat;
	glm::mat4 inversedTransMat;

	glm::vec3 vertices[8];
	int indices[36];
	Triangle* triangles;

} WorldObject;

/* Scene structure */
typedef struct {
	Camera camera;
	WorldObject* objects;

} Scene;


/* Ray structure */
typedef struct {
	glm::vec3 originPoint;
	glm::vec3 direction;

} Ray;


void worldInit();


// global scene object
extern Scene scene;