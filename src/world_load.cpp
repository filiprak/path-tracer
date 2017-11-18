#include "KDTree.h"
#include "world_load.h"
#include "world.h"
#include "main.h"
#include "cuda_runtime.h"
#include <vector>

/* assimp include files */
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "cudaUtility.h"
#include "cutil_math.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

WorldObjectsSources world_obj_sources;

textureReference triangle_texture;

static std::vector<dv_ptr> dv_mem_ptrs;
static std::vector<cudaTextureObject_t> dv_textures;
static std::vector<dv_ptr> dv_mem_kdtrees_ptrs;


void initWorldObjSources() {
	// init load handlers
	world_obj_sources.loadFuncMapping[SphereObj] = loadSphereObj;
	world_obj_sources.loadFuncMapping[TriangleMeshObj] = loadTriangleMeshObj;
	
	// create world objects - compose our scene
	scene.num_wobjects = world_obj_sources.num_objects = 3;
	
	// init light sphere
	/*world_obj_sources.sources[2].type = SphereObj;
	Material lightmat;
	lightmat.color = make_float3(50.0, 255.0, 255.0);
	lightmat.norm_color = lightmat.color / 255.0f;
	lightmat.type = Diffusing;
	lightmat.reflect_factor = 0.2;
	lightmat.refract_index = 1.5f;
	SphereObjInfo* sphere = (SphereObjInfo*)malloc(sizeof(SphereObjInfo));
	sphere->material = lightmat;
	sphere->position = make_float3(0, -50.0, 0);
	sphere->radius = 50.0f;
	world_obj_sources.sources[2].worldObjectInfo = sphere;*/

	// init light sphere
	/*world_obj_sources.sources[4].type = SphereObj;
	Material lightmat3;
	lightmat3.color = make_float3(255.0, 0.0, 255.0);
	lightmat3.norm_color = lightmat3.color / 255.0f;
	lightmat3.emittance = make_float3(255.0f);
	lightmat3.type = Refractive;
	lightmat3.reflect_factor = 0.5;
	lightmat3.refract_index = 1.5f;
	SphereObjInfo* sphere3 = (SphereObjInfo*)malloc(sizeof(SphereObjInfo));
	sphere3->material = lightmat3;
	sphere3->position = make_float3(2, 1.3, 1);
	sphere3->radius = 1.3f;
	world_obj_sources.sources[4].worldObjectInfo = sphere3;*/

	// init light sphere
	world_obj_sources.sources[1].type = SphereObj;
	Material lightmat1;
	lightmat1.color = make_float3(250.0, 230.0, 20.0);
	lightmat1.norm_color = lightmat1.color / 255.0f;
	lightmat1.type = Luminescent;
	lightmat1.emittance = make_float3(7 * 255.0f);
	SphereObjInfo* sphere1 = (SphereObjInfo*)malloc(sizeof(SphereObjInfo));
	sphere1->material = lightmat1;
	sphere1->position = make_float3(0.0, 16.6f, 0.0);
	sphere1->radius = 7.0f;
	world_obj_sources.sources[1].worldObjectInfo = sphere1;

	// init cube mesh 2
	world_obj_sources.sources[2].type = TriangleMeshObj;
	TriangleMeshObjInfo* glasscube = (TriangleMeshObjInfo*)malloc(sizeof(TriangleMeshObjInfo));
	Material glasscubemat;
	glasscubemat.color = make_float3(0, 255.0f, 0);
	glasscubemat.norm_color = glasscubemat.color / 255.0f;
	glasscubemat.reflect_factor = 0.2f;
	glasscubemat.refract_index = 1.3f;
	glasscubemat.type = Diffusing;

	glm::mat4 transg;//f
	transg = glm::translate(transg, glm::vec3(0, 0, 0));
	transg = glm::scale(transg, glm::vec3(0.03));
	transg = glm::rotate(transg, glm::radians(90.0f), glm::vec3(0, 1, 0));
	transg = glm::rotate(transg, glm::radians(-90.0f), glm::vec3(1,0,0));
	//transg = glm::translate(transg, glm::vec3(-0.5, -0.5, -0.5));

	glasscube->transform = transg;
	glasscube->material = glasscubemat;
	strcpy(glasscube->src_filename, "C:/Users/raqu/git/path-tracer/scenes/porsche_911.obj");
	world_obj_sources.sources[2].worldObjectInfo = glasscube;

	// init cube mesh 2
	world_obj_sources.sources[0].type = TriangleMeshObj;
	Material cubemat2;
	cubemat2.color = make_float3(240, 125, 128);
	cubemat2.norm_color = cubemat2.color / 255.0f;
	cubemat2.type = Diffusing;
	cubemat2.reflect_factor = 0.2f;
	cubemat2.refract_index = 1.3f;
	TriangleMeshObjInfo* cube2 = (TriangleMeshObjInfo*)malloc(sizeof(TriangleMeshObjInfo));
	cube2->material = cubemat2;

	glm::mat4 trans2;//f
	//trans2 = glm::translate(trans2, glm::vec3(-4.4, 1, 0));
	//trans2 = glm::scale(trans2, glm::vec3(3));
	//trans2 = glm::rotate(trans2, glm::radians(90.0f), glm::vec3(1,0,0));
	//trans2 = glm::translate(trans2, glm::vec3(-0.5, -0.5, -0.5));

	cube2->transform = trans2;
	strcpy(cube2->src_filename, "C:/Users/raqu/git/path-tracer/scenes/cornell_box.obj");
	world_obj_sources.sources[0].worldObjectInfo = cube2;
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
Material* loadMaterialsToHost(const aiScene* aiscene) {
	if (!aiscene->HasMaterials())
		return NULL;

	int num_materials = aiscene->mNumMaterials;
	Material* mat_ptr = (Material*)malloc(num_materials * sizeof(Material));
	for (int i = 0; i < num_materials; ++i)
	{
		aiMaterial* material = aiscene->mMaterials[i];

		// parse name, convention name_of_material.type_of_material, example: metal.reflective, sun.luminescent
		// if no dot provided in name or not valid name specified, then DefaultMaterial is used
		aiString name;
		material->Get(AI_MATKEY_NAME, name);
		std::string sname = std::string(name.C_Str());
		int dotpos = sname.find_first_of('.');
		std::string mat_type = "default";
		if (dotpos > -1 && dotpos + 1 < sname.length())
			mat_type = sname.substr(dotpos + 1, sname.length() - 1);

		// load material properties
		/* mapping mtl values on Material attributes:
		 * Ka = color
		 * Ke = emitance
		 * d = reflect_factor
		 * Ni = refract_index
		 */
		aiColor3D Ka, Kd, Ke;
		float d, Ni;
		material->Get(AI_MATKEY_COLOR_AMBIENT, Ka);
		material->Get(AI_MATKEY_COLOR_DIFFUSE, Kd);
		material->Get(AI_MATKEY_COLOR_EMISSIVE, Ke);
		material->Get(AI_MATKEY_REFRACTI, Ni);
		material->Get(AI_MATKEY_OPACITY, d);
		
		mat_ptr[i].norm_color = make_float3(Ka.r, Ka.g, Ka.b);
		mat_ptr[i].color = 255.0f * make_float3(Ka.r, Ka.g, Ka.b);
		mat_ptr[i].emittance = make_float3(Ke.r, Ke.g, Ke.b);
		mat_ptr[i].reflect_factor = d;
		mat_ptr[i].refract_index = Ni;
		
		// switch material type
		if (!mat_type.compare("diffuse")) {
			mat_ptr[i].type = Diffusing;
		}
		else if (!mat_type.compare("reflective")) {
			mat_ptr[i].type = Reflective;
		}
		else if (!mat_type.compare("refractive")) {
			mat_ptr[i].type = Refractive;
		}
		else if (!mat_type.compare("luminescent")) {
			mat_ptr[i].type = Luminescent;
		}
		else {
			mat_ptr[i].type = Diffusing;
			mat_ptr[i].norm_color = make_float3(0.75f);
			mat_ptr[i].color = 255.0f * mat_ptr[i].norm_color;
		}
		printf("  Loaded material[%d]: name: %s, type: %s\n", i, name.C_Str(), mat_type.c_str());
		/*printf("    Ka: [%.1f,%.1f,%.1f]\n    Ke: [%.1f,%.1f,%.1f]\n    d: %f, Ni: %f\n",
			mat_ptr[i].norm_color.x, mat_ptr[i].norm_color.y, mat_ptr[i].norm_color.z,
			mat_ptr[i].emittance.x, mat_ptr[i].emittance.y, mat_ptr[i].emittance.z,
			mat_ptr[i].reflect_factor, mat_ptr[i].refract_index);*/
	}
	return mat_ptr;
}

cudaTextureObject_t loadTrianglesToCudaTexture(float4 *dev_triangles_ptr, unsigned int num_triangles)
{
	// create cuda resource object
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeLinear;
	resDesc.res.linear.devPtr = dev_triangles_ptr;
	resDesc.res.linear.desc = cudaCreateChannelDesc<float4>();
	/*resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
	resDesc.res.linear.desc.x = 8 * sizeof(float4); // bits per channel*/
	resDesc.res.linear.sizeInBytes = num_triangles * 6 * sizeof(float4);

	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.readMode = cudaReadModeElementType;
	texDesc.filterMode = cudaFilterModePoint;
	texDesc.addressMode[0] = cudaAddressModeWrap;
	texDesc.normalizedCoords = false;

	// create texture object
	cudaTextureObject_t tex = 0;
	cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);
	dv_textures.push_back(tex);
	checkCudaError("loadTrianglesToCudaTexture()");
	return tex;
}

__host__
Triangle* loadTriangles(const aiScene* aiscene,
						glm::mat4 transform,
						glm::mat4 inv_transform,
						int& all_trs,
						Triangle*& tr_host_ptr,
						float4*& dv_tr_tex_data) {

	// get all scene triangles count
	int all_triangles = 0;
	for (int i = 0; i < aiscene->mNumMeshes; i++)
		all_triangles += aiscene->mMeshes[i]->mNumFaces;

	all_trs = all_triangles;
	tr_host_ptr = (Triangle*)malloc(all_triangles * sizeof(Triangle));
	float4* tr_host_tex_data = (float4*)malloc(all_triangles * 6 * sizeof(float4));
	Triangle* dv_tr_ptr = NULL;

	int loaded_trgs = 0;
	for (int m = 0; m < aiscene->mNumMeshes; m++)
	{
		aiMesh* mesh = aiscene->mMeshes[m];

		for (int i = loaded_trgs; i < loaded_trgs + mesh->mNumFaces; ++i) {
			const aiFace& face = mesh->mFaces[i - loaded_trgs];
			aiVector3D& va = mesh->mVertices[face.mIndices[0]];
			aiVector3D& vb = mesh->mVertices[face.mIndices[1]];
			aiVector3D& vc = mesh->mVertices[face.mIndices[2]];

			// transform triangles using glm
			glm::vec4 ga = transform * glm::vec4(va.x, va.y, va.z, 1.0f);
			glm::vec4 gb = transform * glm::vec4(vb.x, vb.y, vb.z, 1.0f);
			glm::vec4 gc = transform * glm::vec4(vc.x, vc.y, vc.z, 1.0f);

			tr_host_ptr[i].a = make_float3(ga.x, ga.y, ga.z);
			tr_host_ptr[i].b = make_float3(gb.x, gb.y, gb.z);
			tr_host_ptr[i].c = make_float3(gc.x, gc.y, gc.z);

			int trg_tex_idx = 6 * i;
			tr_host_tex_data[trg_tex_idx] = make_float4(ga.x, ga.y, ga.z, 0.0f);
			tr_host_tex_data[trg_tex_idx + 1] = make_float4(gb.x, gb.y, gb.z, 0.0f);
			tr_host_tex_data[trg_tex_idx + 2] = make_float4(gc.x, gc.y, gc.z, 0.0f);

			if (mesh->HasNormals()) {
				aiVector3D& normal0 = mesh->mNormals[face.mIndices[0]];
				aiVector3D& normal1 = mesh->mNormals[face.mIndices[1]];
				aiVector3D& normal2 = mesh->mNormals[face.mIndices[2]];

				// transform normals using glm
				glm::vec4 na = inv_transform * glm::vec4(normal0.x, normal0.y, normal0.z, 1.0f);
				glm::vec4 nb = inv_transform * glm::vec4(normal1.x, normal1.y, normal1.z, 1.0f);
				glm::vec4 nc = inv_transform * glm::vec4(normal2.x, normal2.y, normal2.z, 1.0f);

				tr_host_ptr[i].norm_a = normalize(make_float3(na.x, na.y, na.z));
				tr_host_ptr[i].norm_b = normalize(make_float3(nb.x, nb.y, nb.z));
				tr_host_ptr[i].norm_c = normalize(make_float3(nc.x, nc.y, nc.z));

				tr_host_tex_data[trg_tex_idx + 3] = normalize(make_float4(na.x, na.y, na.z, 0.0f));
				tr_host_tex_data[trg_tex_idx + 4] = normalize(make_float4(na.x, na.y, na.z, 0.0f));
				tr_host_tex_data[trg_tex_idx + 5] = normalize(make_float4(na.x, na.y, na.z, 0.0f));
			}

		}
		loaded_trgs += mesh->mNumFaces;
	}
	cudaOk(cudaMalloc(&dv_tr_ptr, all_triangles * sizeof(Triangle)));
	cudaOk(cudaMemcpy(dv_tr_ptr, tr_host_ptr, all_triangles * sizeof(Triangle), cudaMemcpyHostToDevice));

	cudaOk(cudaMalloc(&dv_tr_tex_data, all_triangles * 6 * sizeof(float4)));
	cudaOk(cudaMemcpy(dv_tr_tex_data, tr_host_tex_data, all_triangles * 6 * sizeof(float4), cudaMemcpyHostToDevice));
	checkCudaError("loadTriangles");

	dv_mem_ptrs.push_back(dv_tr_ptr);
	dv_mem_ptrs.push_back(dv_tr_tex_data);
	return dv_tr_ptr;
}


/*TriangleMesh* loadTriangleMeshes(const aiScene* scene, glm::mat4 transform, glm::mat4 inv_transform) {
	if (scene == NULL)
		return NULL;

	TriangleMesh* dv_tm_ptr = NULL;
	int num_meshes = scene->mNumMeshes;

	TriangleMesh* tm_ptr = (TriangleMesh*)malloc(num_meshes * sizeof(TriangleMesh));
	if (tm_ptr == NULL)
		return NULL;

	Material* materials = loadMaterialsToHost(scene);

	for (int i = 0; i < num_meshes; ++i) {
		aiMesh* mesh = scene->mMeshes[i];
		tm_ptr[i].num_triangles = mesh->mNumFaces;
		tm_ptr[i].triangles = loadTriangles(mesh, transform, inv_transform);
		tm_ptr[i].material = materials[mesh->mMaterialIndex];
		//memcpy(&(tm_ptr[i].material), &(materials[mesh->mMaterialIndex]), sizeof(Material));
		printf("  Loaded mesh[%d]: number of triangles: %d\n", i, mesh->mNumFaces);
	}

	cudaOk(cudaMalloc(&dv_tm_ptr, num_meshes * sizeof(TriangleMesh)));
	cudaOk(cudaMemcpy(dv_tm_ptr, tm_ptr, num_meshes * sizeof(TriangleMesh), cudaMemcpyHostToDevice));
	free(tm_ptr);
	free(materials);
	dv_mem_ptrs.push_back(dv_tm_ptr);
	return dv_tm_ptr;
}*/

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
	printf("  SUCCESS: Loaded sphere.\n\n");
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
	Triangle* trg_host_ptr = NULL;
	float4* dv_tex_triangle_data = NULL;
	
	// count transpose inversed matrix for normal vectors transforming
	glm::mat4 inv_transform = glm::transpose(glm::inverse(info->transform));
	// load all triangles
	int all_trgs_count;
	gptr->triangles = loadTriangles(aiscene, info->transform, inv_transform, all_trgs_count, trg_host_ptr, dv_tex_triangle_data);
	gptr->num_triangles = all_trgs_count;
	gptr->triangles_tex = loadTrianglesToCudaTexture(dv_tex_triangle_data, all_trgs_count);
	printf("  Loaded %d triangles in %d meshes\n", gptr->num_triangles, aiscene->mNumMeshes);
	
	// build kd tree
	printf("  Building KD-Tree for all scene triangles (%d).\n", all_trgs_count);
	int* trg_idxs = (int*)malloc(all_trgs_count * sizeof(int));
	int** trg_addr = &trg_idxs;
	for (int idx = 0; idx < all_trgs_count; idx++)
		trg_idxs[idx] = idx;
	KDNode* host_kdtree = buildKDTree(trg_host_ptr, trg_idxs, all_trgs_count, 0);
	KDNode* dv_kdtree = copyKDTreeToCuda(host_kdtree);
	gptr->kdroot = dv_kdtree;
	dv_mem_kdtrees_ptrs.push_back(dv_kdtree);
	freeKDTree(host_kdtree);

	cudaOk(cudaMalloc(&dv_gptr, sizeof(MeshGeometryData)));
	result.type = TriangleMeshObj;
	result.material = info->material;
	result.geometry_data = dv_gptr;
	cudaOk(cudaMemcpy(dv_gptr, gptr, sizeof(MeshGeometryData), cudaMemcpyHostToDevice));

	if (!checkCudaError("Loading Triangle Mesh Object"))
		printf("  SUCCESS: Loaded Triangle Mesh Object: %s\n\n", info->src_filename);
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

	printf("Sizeof: %s = %d\n", "Triangle", sizeof(Triangle));

	printf("Sizeof: %s = %d\n", "Material", sizeof(Material));
	printf("Sizeof: %s = %d\n", "WorldObjType", sizeof(WorldObjType));
	initWorldObjSources();
	scene.dv_wobjects_ptr = loadWorldObjects();
}

void freeWorldObjects() {
	// free kd trees
	for (int i = 0; i < dv_mem_kdtrees_ptrs.size(); ++i) {
		freeCudaKDTree((KDNode*)dv_mem_kdtrees_ptrs[i]);
	}
	// free allocated cuda memory
	for (int i = 0; i < dv_mem_ptrs.size(); ++i) {
		cudaOk(cudaFree(dv_mem_ptrs[i]));
	}
	// free cuda texture objects
	for (int i = 0; i < dv_textures.size(); i++)
	{
		cudaDestroyTextureObject(dv_textures[i]);
	}
	dv_mem_ptrs.clear();
	dv_mem_kdtrees_ptrs.clear();
	dv_textures.clear();
	freeWorldObjSources();
	checkCudaError("freeWorldObjects");
}
