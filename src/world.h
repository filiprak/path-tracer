#pragma once

#include <glm\glm.hpp>


/* Camera structure */
typedef struct {
	glm::vec3 position;
	glm::vec3 direction;
	glm::vec3 up;

	glm::mat4 viewMat;

	struct {
		int width, height;
		float viewer_dist;

	} projection;

} Camera;

/* Material types */
typedef enum {
	Light,
	Reflecting,
	Diffuse

} Material;

/* Mesh structure - default cube mesh */
typedef struct {
	Material materialType;

	glm::mat4 transformMat;
	glm::mat4 inversedTransMat;

	/*glm::vec3 vertices[8] = {
		glm::vec3(-0.5, -0.5, 0.5),
		glm::vec3(0.5, -0.5, 0.5),
		glm::vec3(0.5, 0.5, 0.5),
		glm::vec3(-0.5, 0.5, 0.5),
		glm::vec3(-0.5, -0.5, -0.5),
		glm::vec3(0.5, -0.5, -0.5),
		glm::vec3(0.5, 0.5, -0.5),
		glm::vec3(-0.5, 0.5, -0.5)
	};

	int indices[36] = {
		// front
		0, 1, 2,
		2, 3, 0,
		// top
		1, 5, 6,
		6, 2, 1,
		// back
		7, 6, 5,
		5, 4, 7,
		// bottom
		4, 0, 3,
		3, 7, 4,
		// left
		4, 5, 1,
		1, 0, 4,
		// right
		3, 2, 6,
		6, 7, 3,
	};*/

} Mesh;

/* Scene structure */
typedef struct {
	struct {
		Camera camera;
		Mesh mesh;

	} elements;

} Scene;


/* Ray structure */
typedef struct {
	glm::vec3 originPoint;
	glm::vec3 direction;

} Ray;


typedef struct {
	glm::vec3 a, b, c;
} Triangle;


Camera createCamera();