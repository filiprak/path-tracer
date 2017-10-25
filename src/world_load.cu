#include "world_load.h"
#include "world.h"
#include "main.h"
#include <vector>

/* assimp include files */
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

typedef void* dv_ptr;
static std::vector<dv_ptr> dv_mem_ptrs;


__host__
Triangle* loadTriangles(const aiMesh* mesh) {
	if (mesh == NULL)
		return NULL;

	Triangle* dv_tr_ptr = NULL;

	int num_faces = mesh->mNumFaces;
	Triangle* tr_ptr = (Triangle*)malloc(num_faces * sizeof(Triangle));
	if (tr_ptr == NULL)
		return NULL;
	
	for (int i = 0; i < num_faces; ++i) {
		const aiFace& face = mesh->mFaces[i];
		aiVector3D& va = mesh->mVertices[face.mIndices[0]];
		aiVector3D& vb = mesh->mVertices[face.mIndices[1]];
		aiVector3D& vc = mesh->mVertices[face.mIndices[2]];
		tr_ptr[i].a = make_float3(va.x, va.y, va.z);
		tr_ptr[i].b = make_float3(vb.x, vb.y, vb.z);
		tr_ptr[i].c = make_float3(vc.x, vc.y, vc.z);
		printf("triangles[%d]=[%f,%f,%f]\n", i, tr_ptr[i].a.x, tr_ptr[i].b.x, tr_ptr[i].c.x);
		if (mesh->HasNormals()) {
			aiVector3D& normal0 = mesh->mNormals[face.mIndices[0]];
			aiVector3D& normal1 = mesh->mNormals[face.mIndices[1]];
			aiVector3D& normal2 = mesh->mNormals[face.mIndices[2]];
			tr_ptr[i].norm_a = make_float3(normal0.x, normal0.y, normal0.z);
			tr_ptr[i].norm_b = make_float3(normal1.x, normal1.y, normal1.z);
			tr_ptr[i].norm_c = make_float3(normal2.x, normal2.y, normal2.z);
			printf("normal[%d]=[%f,%f,%f]\n", i, normal0.x, normal0.y, normal0.z);
		}

	}
	cudaMalloc(&dv_tr_ptr, num_faces * sizeof(Triangle));
	cudaMemcpy(dv_tr_ptr, tr_ptr, num_faces * sizeof(Triangle), cudaMemcpyHostToDevice);

	free(tr_ptr);
	dv_mem_ptrs.push_back(dv_tr_ptr);
	return dv_tr_ptr;
}

__host__
float3* loadVertices(const aiMesh* mesh) {
	if (mesh == NULL)
		return NULL;

	float3* dv_vrt_ptr = NULL;

	int num_vertices = mesh->mNumVertices;
	float3* vrt_ptr = (float3*)malloc(num_vertices * sizeof(float3));
	if (vrt_ptr == NULL)
		return NULL;

	for (int i = 0; i < num_vertices; ++i) {
		aiVector3D& v = mesh->mVertices[i];
		vrt_ptr[i] = make_float3(v.x, v.y, v.z);
		printf("vertices[%d]=[%f,%f,%f]\n", i, v.x, v.y, v.z);
	}

	cudaMalloc(&dv_vrt_ptr, num_vertices * sizeof(float3));
	cudaMemcpy(dv_vrt_ptr, vrt_ptr, num_vertices * sizeof(float3), cudaMemcpyHostToDevice);

	free(vrt_ptr);
	dv_mem_ptrs.push_back(dv_vrt_ptr);
	return dv_vrt_ptr;
}


__host__
TriangleMesh* loadTriangleMeshes(const aiScene* scene) {
	if (scene == NULL)
		return NULL;

	TriangleMesh* dv_tm_ptr = NULL;
	int num_meshes = scene->mNumMeshes;

	TriangleMesh* tm_ptr = (TriangleMesh*)malloc(num_meshes * sizeof(TriangleMesh));
	if (tm_ptr == NULL)
		return NULL;

	for (int i = 0; i < num_meshes; ++i) {
		aiMesh* mesh = scene->mMeshes[i];
		tm_ptr[i].num_triangles = mesh->mNumFaces;
		tm_ptr[i].triangles = loadTriangles(mesh);
	}

	cudaMalloc(&dv_tm_ptr, num_meshes * sizeof(TriangleMesh));
	cudaMemcpy(dv_tm_ptr, tm_ptr, num_meshes * sizeof(TriangleMesh), cudaMemcpyHostToDevice);

	dv_mem_ptrs.push_back(dv_tm_ptr);
	return dv_tm_ptr;
}

__host__
WorldObject* loadWorldObjects(std::string* filenames, int num) {
	
	Assimp::Importer importer;

	WorldObject* wo_ptr;
	WorldObject* dv_wo_ptr = NULL;

	wo_ptr = (WorldObject*)malloc(num * sizeof(WorldObject));

	if (wo_ptr == NULL)
		return NULL;

	for (int i = 0; i < num; ++i) {
		printf("Loading World Object: %s\n", scenefile.c_str());
		const aiScene* aiscene = importer.ReadFile(scenefile,
			aiProcess_CalcTangentSpace |
			aiProcess_Triangulate |
			aiProcess_JoinIdenticalVertices |
			aiProcess_SortByPType
			);

		int num_meshes = aiscene != NULL ? aiscene->mNumMeshes : 0;
		wo_ptr[i].num_meshes = num_meshes;
		wo_ptr[i].meshes = loadTriangleMeshes(aiscene);
		wo_ptr[i].materialType = Light;

		if (aiscene == NULL) {
			printf("  Failed to load: %s.\n", scenefile.c_str());
		}
		else {
			printf("  Success, loaded: %s. \n", scenefile.c_str());
		}
	}

	cudaMalloc(&dv_wo_ptr, num * sizeof(WorldObject));
	cudaMemcpy(dv_wo_ptr, wo_ptr, num * sizeof(WorldObject), cudaMemcpyHostToDevice);
	free(wo_ptr);
	dv_mem_ptrs.push_back(dv_wo_ptr);
	return dv_wo_ptr;
}

__host__
void loadSceneWorldObjects() {
	scene.dv_wobjects_ptr = loadWorldObjects(NULL, 1);
}

__host__
void freeWorldObjects() {
	for (int i = 0; i < dv_mem_ptrs.size(); ++i) {
		cudaFree(dv_mem_ptrs[i]);
	}
	dv_mem_ptrs.clear();
}

