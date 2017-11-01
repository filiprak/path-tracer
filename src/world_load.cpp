#include "world_load.h"
#include "world.h"
#include "main.h"
#include <vector>

/* assimp include files */
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "cudaUtility.h"
#include "cutil_math.h"

WorldObjectsSources world_obj_sources;


typedef void* dv_ptr;
static std::vector<dv_ptr> dv_mem_ptrs;

void initWorldObjSources() {
	// init load handlers
	world_obj_sources.loadFuncMapping[SphereObj] = loadSphereObj;
	world_obj_sources.loadFuncMapping[TriangleMeshObj] = loadTriangleMeshObj;
	
	// create world objects - compose our scene
	scene.num_wobjects = world_obj_sources.num_objects = 4;
	
	// init light sphere
	world_obj_sources.sources[0].type = SphereObj;
	Material lightmat;
	lightmat.color = make_float3(255.0, 145.0, 50.0);
	lightmat.norm_color = lightmat.color / 255.0f;
	lightmat.type = Reflective;
	lightmat.emittance = make_float3(2550.0f);
	SphereObjInfo* sphere = (SphereObjInfo*)malloc(sizeof(SphereObjInfo));
	sphere->material = lightmat;
	sphere->position = make_float3(3.0, 1.6, 1.0);
	sphere->radius = 1.3f;
	world_obj_sources.sources[0].worldObjectInfo = sphere;

	// init light sphere
	world_obj_sources.sources[1].type = SphereObj;
	Material lightmat1;
	lightmat1.color = make_float3(250.0, 230.0, 20.0);
	lightmat1.norm_color = lightmat1.color / 255.0f;
	lightmat1.type = Luminescent;
	lightmat1.emittance = make_float3(1000.0f);
	SphereObjInfo* sphere1 = (SphereObjInfo*)malloc(sizeof(SphereObjInfo));
	sphere1->material = lightmat1;
	sphere1->position = make_float3(0.0, 15.0, 0.0);
	sphere1->radius = 10.0f;
	world_obj_sources.sources[1].worldObjectInfo = sphere1;

	// init cube mesh
	world_obj_sources.sources[2].type = TriangleMeshObj;
	Material cubemat;
	cubemat.color = make_float3(10.0, 128.0, 255.0);
	cubemat.norm_color = cubemat.color / 255.0f;
	cubemat.type = Diffusing;
	TriangleMeshObjInfo* cube = (TriangleMeshObjInfo*)malloc(sizeof(TriangleMeshObjInfo));
	cube->material = cubemat;

	glm::mat4 trans;
	trans = glm::scale(trans, glm::vec3(20, 0.2, 20));
	trans = glm::translate(trans, glm::vec3(-0.5, -0.5, -0.5));
	
	cube->transform = trans;
	strcpy(cube->src_filename, "C:/Users/raqu/git/path-tracer/scenes/cube.obj");
	world_obj_sources.sources[2].worldObjectInfo = cube;

	// init cube mesh 2
	world_obj_sources.sources[3].type = TriangleMeshObj;
	Material cubemat2;
	cubemat2.color = make_float3(204.0, 9.0, 102.0);
	cubemat2.norm_color = cubemat2.color / 255.0f;
	cubemat2.type = Reflective;
	TriangleMeshObjInfo* cube2 = (TriangleMeshObjInfo*)malloc(sizeof(TriangleMeshObjInfo));
	cube2->material = cubemat2;

	glm::mat4 trans2;
	trans2 = glm::translate(trans2, glm::vec3(0, 2, -3));
	trans2 = glm::scale(trans2, glm::vec3(3));
	trans2 = glm::translate(trans2, glm::vec3(-0.5, -0.5, -0.5));

	cube2->transform = trans2;
	strcpy(cube2->src_filename, "C:/Users/raqu/git/path-tracer/scenes/cube.obj");
	world_obj_sources.sources[3].worldObjectInfo = cube2;
}

void freeWorldObjSources() {
	// create world objects - compose our scene
	for (int i = 0; i < MAX_OBJECTS_NUM; ++i) {
		if (world_obj_sources.sources[i].worldObjectInfo != NULL)
			free(world_obj_sources.sources[i].worldObjectInfo);
	}
	world_obj_sources.num_objects = 0;
}

__host__
Triangle* loadTriangles(const aiMesh* mesh, glm::mat4 transform) {
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

		// transform using glm
		glm::vec4 ga = transform * glm::vec4(va.x, va.y, va.z, 1.0f);
		glm::vec4 gb = transform * glm::vec4(vb.x, vb.y, vb.z, 1.0f);
		glm::vec4 gc = transform * glm::vec4(vc.x, vc.y, vc.z, 1.0f);

		tr_ptr[i].a = make_float3(ga.x, ga.y, ga.z);
		tr_ptr[i].b = make_float3(gb.x, gb.y, gb.z);
		tr_ptr[i].c = make_float3(gc.x, gc.y, gc.z);
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
	cudaOk(cudaMalloc(&dv_tr_ptr, num_faces * sizeof(Triangle)));
	cudaOk(cudaMemcpy(dv_tr_ptr, tr_ptr, num_faces * sizeof(Triangle), cudaMemcpyHostToDevice));
	checkCudaError("loadTriangles");

	free(tr_ptr);
	dv_mem_ptrs.push_back(dv_tr_ptr);
	return dv_tr_ptr;
}

/*__host__
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
}*/

TriangleMesh* loadTriangleMeshes(const aiScene* scene, glm::mat4 transform) {
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
		tm_ptr[i].triangles = loadTriangles(mesh, transform);
	}

	cudaOk(cudaMalloc(&dv_tm_ptr, num_meshes * sizeof(TriangleMesh)));
	cudaOk(cudaMemcpy(dv_tm_ptr, tm_ptr, num_meshes * sizeof(TriangleMesh), cudaMemcpyHostToDevice));
	dv_mem_ptrs.push_back(dv_tm_ptr);
	return dv_tm_ptr;
}

bool loadSphereObj(void* objInfo, WorldObject& result) {
	SphereObjInfo* info = (SphereObjInfo*)objInfo;

	printf("Loading Sphere Object: r=%f, pos=[%f,%f,%f]\n", info->radius, info->position.x, info->position.y, info->position.z);
	SphereGeometryData* gptr = (SphereGeometryData*)malloc(sizeof(SphereGeometryData));
	
	gptr->position = info->position;
	gptr->radius = info->radius;

	SphereGeometryData* dv_gptr = NULL;
	cudaOk(cudaMalloc(&dv_gptr, sizeof(SphereGeometryData)));
	result.type = SphereObj;
	result.material = info->material;
	result.geometry_data = dv_gptr;
	cudaOk(cudaMemcpy(dv_gptr, gptr, sizeof(SphereGeometryData), cudaMemcpyHostToDevice));
	printf("  SUCCESS: Loaded sphere.\n");
	free(gptr);
	dv_mem_ptrs.push_back(dv_gptr);
	return true;
}

bool loadTriangleMeshObj(void* objInfo, WorldObject& result) {
	TriangleMeshObjInfo* info = (TriangleMeshObjInfo*)objInfo;

	Assimp::Importer importer;
	printf("Loading Triangle Mesh Object: %s\n", info->src_filename);
	const aiScene* aiscene = importer.ReadFile(info->src_filename,
		aiProcess_CalcTangentSpace |
		aiProcess_Triangulate |
		aiProcess_JoinIdenticalVertices |
		aiProcess_SortByPType
		);
	if (aiscene == NULL) {
		printf("  ERROR: Failed to load Triangle Mesh Object: %s\n", info->src_filename);
		return false;
	}

	MeshGeometryData* gptr = (MeshGeometryData*)malloc(sizeof(MeshGeometryData));
	MeshGeometryData* dv_gptr = NULL;
	gptr->num_meshes = aiscene->mNumMeshes;
	gptr->meshes = loadTriangleMeshes(aiscene, info->transform);

	cudaOk(cudaMalloc(&dv_gptr, sizeof(MeshGeometryData)));
	result.type = TriangleMeshObj;
	result.material = info->material;
	result.geometry_data = dv_gptr;
	cudaOk(cudaMemcpy(dv_gptr, gptr, sizeof(MeshGeometryData), cudaMemcpyHostToDevice));
	
	if (!checkCudaError("Loading Triangle Mesh Object"))
		printf("  SUCCESS: Loaded Triangle Mesh Object: %s\n", info->src_filename);
	dv_mem_ptrs.push_back(dv_gptr);
	free(gptr);
	return true;
}

WorldObject* loadWorldObjects() {
	
	WorldObject* wo_ptr;
	WorldObject* dv_wo_ptr = NULL;

	int num_objects = world_obj_sources.num_objects;
	wo_ptr = (WorldObject*)malloc(num_objects * sizeof(WorldObject));

	for (int i = 0; i < num_objects; ++i) {
		WorldObjType type = world_obj_sources.sources[i].type;
		void* objInfo = world_obj_sources.sources[i].worldObjectInfo;
		if (!(*world_obj_sources.loadFuncMapping[type])(objInfo, wo_ptr[i]))
			return NULL;
	}

	cudaOk(cudaMalloc(&dv_wo_ptr, num_objects * sizeof(WorldObject)));
	cudaOk(cudaMemcpy(dv_wo_ptr, wo_ptr, num_objects * sizeof(WorldObject), cudaMemcpyHostToDevice));

	free(wo_ptr);
	dv_mem_ptrs.push_back(dv_wo_ptr);
	return dv_wo_ptr;
}

void loadSceneWorldObjects() {
	printf("Sizeof: %s = %d\n\n", "WorldObject", sizeof(WorldObject));
	printf("Sizeof: %s = %d\n", "MeshGeometryData", sizeof(MeshGeometryData));
	printf("Sizeof: %s = %d\n", "SphereGeometryData", sizeof(SphereGeometryData));

	printf("Sizeof: %s = %d\n", "TriangleMesh", sizeof(TriangleMesh));
	printf("Sizeof: %s = %d\n", "Triangle", sizeof(Triangle));

	printf("Sizeof: %s = %d\n", "Material", sizeof(Material));
	printf("Sizeof: %s = %d\n", "WorldObjType", sizeof(WorldObjType));

	printf("Sizeof: %s = %d\n\n", "void*", sizeof(void*));
	initWorldObjSources();
	scene.dv_wobjects_ptr = loadWorldObjects();
}

void freeWorldObjects() {
	for (int i = 0; i < dv_mem_ptrs.size(); ++i) {
		cudaOk(cudaFree(dv_mem_ptrs[i]));
	}
	dv_mem_ptrs.clear();
	freeWorldObjSources();
	checkCudaError("freeWorldObjects");
}
