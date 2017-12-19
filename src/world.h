#pragma once

#include "main.h"
#include "camera.h"
#include "cuda_runtime.h"
#include <glm/glm.hpp>


/* World Object types, important: dont reorder values */
typedef enum {
	SphereObj = 0,
	TriangleMeshObj

} WorldObjType;

/* Material types */
typedef enum {
	Luminescent = 0,
	Diffuse,
	Specular,
	Transparent,

} MaterialType;


#define GLOB_MEM_ALIGNMENT		16

typedef struct alignMem(GLOB_MEM_ALIGNMENT) {
	MaterialType type;
	float3 color, norm_color;  // mtl: Ka
	int cuda_texture_obj;

	// for light materials
	float3 emittance; // mtl: Ke

	// reflective materials
	float reflect_factor; // mtl: d

	// refractive materials
	float refract_index; // mtl: Ni
	float sharpness; // mtl: Ns

} Material;


typedef struct alignMem(GLOB_MEM_ALIGNMENT) {
	float3 a,b,c,e1,e2; // triangle vertices A B C and edges E1 = a - b,  E2 = c - b
	float3 norm_a, norm_b, norm_c; // normals of triangle vertices
	float2 tx_a, tx_b, tx_c; // texture coords of triangle vertices
	int material_idx; // material index of the triangle
} Triangle;


// Axis aligned bounding boxes AABB
typedef struct alignMem(GLOB_MEM_ALIGNMENT) {
	float3 bounds[2]; // minimal and maximal volume point
	// bounds[0] - min, bounds[1] - max
} BBox;


typedef alignMem(GLOB_MEM_ALIGNMENT) struct KDNode {
	// Bounding box of the tree node
	BBox bbox;

	// Triangle indexes of stored in current node
	int* trg_idxs;
	int num_trgs;

	// child pointers
	KDNode* left;
	KDNode*	right;

	// for iterating purposes
	int idx;
	int left_idx;
	int right_idx;

} KDNode;


typedef struct alignMem(GLOB_MEM_ALIGNMENT) {
	// Pointer to tringles data in global memory
	Triangle* triangles;
	int num_triangles;

	// Id of texture where triangles data is stored
	cudaTextureObject_t triangles_tex;
	float4* triangle_tex_data;

	KDNode* kdroot;
	KDNode* flat_kd_root;

} MeshGeometryData;


typedef struct alignMem(GLOB_MEM_ALIGNMENT) {
	float radius;
	float3 position;

} SphereGeometryData;


/* WorldObject structure */
typedef struct alignMem(GLOB_MEM_ALIGNMENT) {
	WorldObjType type;
	Material* materials;
	int num_materials;

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