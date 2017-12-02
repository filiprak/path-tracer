#include "KDTree.h"
#include "world_load.h"
#include "world.h"
#include "main.h"
#include "errors.h"
#include <vector>

/* stb image */
#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

/* assimp include files */
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

/* json includes */
#include "json/json.h"
#include "jsonResolve.h"

/* cuda includes */
#include "cuda_runtime.h"
#include "cudaUtility.h"
#include "cutil_math.h"

/* glm */
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>


WorldObjectsSources world_obj_sources;

// cuda device resources collections
static std::vector<dv_ptr> dv_mem_ptrs;
static std::vector<cudaTextureObject_t> dv_textures;
static std::vector<dv_ptr> dv_mem_kdtrees_ptrs;


void initWorldObjSources(const Json::Value& jscene) {
	// init load handlers
	world_obj_sources.loadFuncMapping[SphereObj] = loadSphereObj;
	world_obj_sources.loadFuncMapping[TriangleMeshObj] = loadTriangleMeshObj;
	
	// parse scene from json description
	int num_objects = min((jscene["objects"].isArray() ? jscene["objects"].size() : 0), MAX_OBJECTS_NUM);
	scene.num_wobjects = world_obj_sources.num_objects = num_objects;

	for (int i = 0; i < num_objects; i++)
	{
		const Json::Value& jobj = jscene["objects"][i];
		std::string type = jobj["type"].asString();

		if (!type.compare("sphere")) {
			world_obj_sources.sources[i].type = SphereObj;

			Material msphere;
			msphere.norm_color = resolveFloat3(jobj["material"]["Ka"]);
			msphere.color = 255.0f * msphere.norm_color;
			msphere.cuda_texture_obj = -1;
			msphere.type = resolveMatType(jobj["material"]["type"]);
			msphere.emittance = resolveFloat3(jobj["material"]["Ke"]);
			msphere.reflect_factor = resolveFloat(jobj["material"]["d"]);
			msphere.refract_index = resolveFloat(jobj["material"]["Ni"]);

			SphereObjInfo* sinfo = (SphereObjInfo*)malloc(sizeof(SphereObjInfo));
			sinfo->material = msphere;
			sinfo->position = resolveFloat3(jobj["position"]);
			sinfo->radius = resolveFloat(jobj["radius"]);
			world_obj_sources.sources[i].worldObjectInfo = sinfo;
		}
		else if (!type.compare("mesh")) {
			world_obj_sources.sources[i].type = TriangleMeshObj;
			TriangleMeshObjInfo* minfo = (TriangleMeshObjInfo*)malloc(sizeof(TriangleMeshObjInfo));
			const char* src = jobj["src"].asCString();
			strcpy_s(minfo->src_filename, src);
			
			glm::mat4 transform;
			// read transformations
			if (jobj["transform"].isArray()) {
				for (int t = 0; t < jobj["transform"].size(); t++)
				{
					const Json::Value& jtrans = jobj["transform"][t];
					if (!jtrans.isObject() || jtrans.size() != 1)
						continue;
					std::string trans_type = jtrans.getMemberNames()[0];
					if (!trans_type.compare("translate"))
						transform = glm::translate(transform, resolveGlmVec3(jtrans[trans_type]));
					else if (!trans_type.compare("rotate"))
						transform = glm::rotate(transform,
							glm::radians(resolveFloat(jtrans[trans_type][0])),
							resolveGlmVec3(jtrans[trans_type][1]));
					else if (!trans_type.compare("scale"))
						transform = glm::scale(transform, resolveGlmVec3(jtrans[trans_type]));
				}
			}
			minfo->transform = transform;
			world_obj_sources.sources[i].worldObjectInfo = minfo;
		}
		else {
			freeWorldObjects();
			throw scene_file_error("Unknown object type, supported types: mesh/sphere");
		}
	}
	
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
cudaTextureObject_t loadTexture(const std::string src_path)
{
	int width, height, channels;
	float* img = stbi_loadf(src_path.c_str(), &width, &height, &channels, 0);
	if (!img) // failed to load texture file
		return -1;

	float4* host_tex_data = (float4*)malloc(height * width * sizeof(float4));
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			int index = channels * (row * width + col);
			float rgba[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
			for (int channel = 0; channel < channels; channel++)
				rgba[channel] = img[index + channel];
			if (channels < 3)
				host_tex_data[row * width + col] = make_float4(rgba[0], rgba[0], rgba[0], rgba[1]);
			else
				host_tex_data[row * width + col] = make_float4(rgba[0], rgba[1], rgba[2], rgba[3]);
		}
	}
	stbi_image_free(img);
	// allocate and copy pitch2D on cuda device
	float4* dev_tex_data;
	size_t pitch;
	cudaOk(cudaMallocPitch(&dev_tex_data, &pitch, width * sizeof(float4), height));
	cudaOk(cudaMemcpy2D(dev_tex_data, pitch, host_tex_data,
		width * sizeof(float4), height * sizeof(float4),
		height, cudaMemcpyHostToDevice));
	dv_mem_ptrs.push_back(dev_tex_data);
	free(host_tex_data);

	// create cuda resource object
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypePitch2D;
	resDesc.res.pitch2D.width = width;
	resDesc.res.pitch2D.height = height;
	resDesc.res.pitch2D.pitchInBytes = pitch;
	resDesc.res.pitch2D.devPtr = dev_tex_data;
	resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float4>();

	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.readMode = cudaReadModeElementType;
	texDesc.filterMode = cudaFilterModePoint;
	texDesc.addressMode[0] = cudaAddressModeWrap;
	texDesc.addressMode[1] = cudaAddressModeWrap;
	texDesc.normalizedCoords = true;

	// create texture object
	cudaTextureObject_t tex = 0;
	cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);
	dv_textures.push_back(tex);
	checkCudaError("loadTexture()");
	return tex;
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
		aiString texture_path;
		unsigned int uvindex;

		material->Get(AI_MATKEY_COLOR_AMBIENT, Ka);
		material->Get(AI_MATKEY_COLOR_DIFFUSE, Kd);
		material->Get(AI_MATKEY_COLOR_EMISSIVE, Ke);
		material->Get(AI_MATKEY_REFRACTI, Ni);
		material->Get(AI_MATKEY_OPACITY, d);
		
		mat_ptr[i].cuda_texture_obj = -1;
		if (material->GetTextureCount(aiTextureType_AMBIENT) > 0) {
			material->GetTexture(aiTextureType_AMBIENT, 0, &texture_path, NULL, &uvindex);
			printf("    Loading ambient texture: %s\n", texture_path.C_Str());
			mat_ptr[i].cuda_texture_obj = loadTexture(std::string(texture_path.C_Str()));
		}
		else if (material->GetTextureCount(aiTextureType_DIFFUSE) > 0) {
			material->GetTexture(aiTextureType_DIFFUSE, 0, &texture_path, NULL, &uvindex);
			printf("    Loading diffuse texture: %s\n", texture_path.C_Str());
			mat_ptr[i].cuda_texture_obj = loadTexture(std::string(texture_path.C_Str()));
		}
		else if (material->GetTextureCount(aiTextureType_EMISSIVE) > 0) {
			material->GetTexture(aiTextureType_EMISSIVE, 0, &texture_path, NULL, &uvindex);
			printf("    Loading emissive texture: %s\n", texture_path.C_Str());
			mat_ptr[i].cuda_texture_obj = loadTexture(std::string(texture_path.C_Str()));
		}
		if (mat_ptr[i].cuda_texture_obj < 0)
			printf("    Warning: texture load failed\n");
		
		mat_ptr[i].norm_color = make_float3(Kd.r, Kd.g, Kd.b);
		mat_ptr[i].color = 255.0f * make_float3(Kd.r, Kd.g, Kd.b);
		mat_ptr[i].emittance = 255.0f * make_float3(Ke.r, Ke.g, Ke.b);
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

		// load triangles
		for (int i = loaded_trgs; i < loaded_trgs + mesh->mNumFaces; ++i) {
			const aiFace& face = mesh->mFaces[i - loaded_trgs];
			const aiVector3D& va = mesh->mVertices[face.mIndices[0]];
			const aiVector3D& vb = mesh->mVertices[face.mIndices[1]];
			const aiVector3D& vc = mesh->mVertices[face.mIndices[2]];

			// transform triangles using glm
			glm::vec4 ga = transform * glm::vec4(va.x, va.y, va.z, 1.0f);
			glm::vec4 gb = transform * glm::vec4(vb.x, vb.y, vb.z, 1.0f);
			glm::vec4 gc = transform * glm::vec4(vc.x, vc.y, vc.z, 1.0f);

			tr_host_ptr[i].a = make_float3(ga.x, ga.y, ga.z);
			tr_host_ptr[i].b = make_float3(gb.x, gb.y, gb.z);
			tr_host_ptr[i].c = make_float3(gc.x, gc.y, gc.z);
			tr_host_ptr[i].material_idx = mesh->mMaterialIndex;

			int trg_tex_idx = 6 * i;
			tr_host_tex_data[trg_tex_idx] = make_float4(ga.x, ga.y, ga.z, 0.0f);
			tr_host_tex_data[trg_tex_idx + 1] = make_float4(gb.x, gb.y, gb.z, 0.0f);
			tr_host_tex_data[trg_tex_idx + 2] = make_float4(gc.x, gc.y, gc.z, 0.0f);

		}
		// load normals
		if (mesh->HasNormals()) {
			for (int i = loaded_trgs; i < loaded_trgs + mesh->mNumFaces; ++i) {
				const aiFace& face = mesh->mFaces[i - loaded_trgs];
				const aiVector3D& normal0 = mesh->mNormals[face.mIndices[0]];
				const aiVector3D& normal1 = mesh->mNormals[face.mIndices[1]];
				const aiVector3D& normal2 = mesh->mNormals[face.mIndices[2]];

				// transform normals using glm
				glm::vec4 na = inv_transform * glm::vec4(normal0.x, normal0.y, normal0.z, 1.0f);
				glm::vec4 nb = inv_transform * glm::vec4(normal1.x, normal1.y, normal1.z, 1.0f);
				glm::vec4 nc = inv_transform * glm::vec4(normal2.x, normal2.y, normal2.z, 1.0f);

				tr_host_ptr[i].norm_a = normalize(make_float3(na.x, na.y, na.z));
				tr_host_ptr[i].norm_b = normalize(make_float3(nb.x, nb.y, nb.z));
				tr_host_ptr[i].norm_c = normalize(make_float3(nc.x, nc.y, nc.z));

				int trg_tex_idx = 6 * i;
				tr_host_tex_data[trg_tex_idx + 3] = normalize(make_float4(na.x, na.y, na.z, 0.0f));
				tr_host_tex_data[trg_tex_idx + 4] = normalize(make_float4(na.x, na.y, na.z, 0.0f));
				tr_host_tex_data[trg_tex_idx + 5] = normalize(make_float4(na.x, na.y, na.z, 0.0f));

			}
		}
		// load tex coords from 0 channel - currently only one channel supported
		if (mesh->HasTextureCoords(0)) {
			for (int i = loaded_trgs; i < loaded_trgs + mesh->mNumFaces; ++i) {
				const aiFace& face = mesh->mFaces[i - loaded_trgs];
				const aiVector3D& ta = mesh->mTextureCoords[0][face.mIndices[0]];
				const aiVector3D& tb = mesh->mTextureCoords[0][face.mIndices[1]];
				const aiVector3D& tc = mesh->mTextureCoords[0][face.mIndices[2]];

				tr_host_ptr[i].tx_a = make_float2(ta.x, ta.y);
				tr_host_ptr[i].tx_b = make_float2(tb.x, tb.y);
				tr_host_ptr[i].tx_c = make_float2(tc.x, tc.y);

				//printf("[%f,%f] [%f,%f] [%f,%f]\n", ta.x, ta.y, tb.x, tb.y, tc.x, tc.y);
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


bool loadSphereObj(void* objInfo, WorldObject& result) {
	SphereObjInfo* info = (SphereObjInfo*)objInfo;

	printf("Loading Sphere Object: r=%f, pos=[%f,%f,%f]\n", info->radius, info->position.x, info->position.y, info->position.z);
	SphereGeometryData* gptr = (SphereGeometryData*)malloc(sizeof(SphereGeometryData));
	
	gptr->position = info->position;
	gptr->radius = info->radius;

	Material* dv_mat = NULL;
	cudaOk(cudaMalloc(&dv_mat, sizeof(Material)));
	cudaOk(cudaMemcpy(dv_mat, &(info->material), sizeof(Material), cudaMemcpyHostToDevice));
	dv_mem_ptrs.push_back(dv_mat);

	SphereGeometryData* dv_gptr = NULL;
	cudaOk(cudaMalloc(&dv_gptr, sizeof(SphereGeometryData)));
	result.type = SphereObj;
	result.materials = dv_mat;
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
	printf("Loading Mesh Object: %s\n", info->src_filename);
	const aiScene* aiscene = importer.ReadFile(info->src_filename,
		aiProcess_CalcTangentSpace |
		aiProcess_Triangulate |
		aiProcess_JoinIdenticalVertices |
		aiProcess_SortByPType
		);
	if (aiscene == NULL) {
		std::string path(info->src_filename);
		printf("  ERROR: Failed to load Mesh Object: %s\n", info->src_filename);
		throw scene_file_error("Assimp failed to load file: " + path);
	}

	if (aiscene->HasTextures()) {
		for (int i = 0; i < aiscene->mNumTextures; i++)
		{
			const aiTexture* aitex = aiscene->mTextures[i];
			printf(" Loading texture[%d]: %s %dx%d\n", i, aitex->achFormatHint, aitex->mWidth, aitex->mHeight);
			for (int x = 0; x < aitex->mWidth; x++)
			{
				for (int y = 0; y < aitex->mHeight; y++)
				{
					const aiTexel& tex = aitex->pcData[(aitex->mHeight) * y + x];
				}
			}
		}
	}

	MeshGeometryData* gptr = (MeshGeometryData*)malloc(sizeof(MeshGeometryData));
	MeshGeometryData* dv_gptr = NULL;
	Triangle* trg_host_ptr = NULL;
	float4* dv_tex_triangle_data = NULL;

	// load materials
	Material* dv_materials = NULL;
	Material* host_materials = loadMaterialsToHost(aiscene);
	cudaOk(cudaMalloc(&dv_materials, aiscene->mNumMaterials * sizeof(Material)));
	cudaOk(cudaMemcpy(dv_materials, host_materials, aiscene->mNumMaterials * sizeof(Material), cudaMemcpyHostToDevice));
	result.num_materials = aiscene->mNumMaterials;
	result.materials = dv_materials;
	dv_mem_ptrs.push_back(dv_materials);
	
	// count transpose inversed matrix for normal vectors transforming
	glm::mat4 inv_transform = glm::transpose(glm::inverse(info->transform));

	// load all triangles
	int all_trgs_count;
	gptr->triangles = loadTriangles(aiscene, info->transform, inv_transform, all_trgs_count, trg_host_ptr, dv_tex_triangle_data);
	gptr->num_triangles = all_trgs_count;
	gptr->triangles_tex = loadTrianglesToCudaTexture(dv_tex_triangle_data, all_trgs_count);
	int trgs_mem_size = all_trgs_count * sizeof(Triangle) / 1024;
	printf("  Loaded %d triangles in %d meshes. (Glob. memory taken: %d KB)\n", gptr->num_triangles, aiscene->mNumMeshes, trgs_mem_size);
	
	// build kd tree
	printf("  Building KD-Tree for all triangles (%d).\n", all_trgs_count);
	int* trg_idxs = (int*)malloc(all_trgs_count * sizeof(int));
	for (int idx = 0; idx < all_trgs_count; idx++)
		trg_idxs[idx] = idx;
	int num_nodes = 0, max_depth = 0;
	KDNode* host_kdtree = buildKDTree(trg_host_ptr, trg_idxs, all_trgs_count, 0, num_nodes, max_depth);
	int kd_tree_size = num_nodes * sizeof(KDNode) / 1024;
	KDNode* host_flat_tree = (KDNode*)malloc(num_nodes * sizeof(KDNode));
	flatenKDTree(host_kdtree, host_flat_tree);
	printf("  Finished building KD-Tree (Num. nodes: %d, Max depth: %d, Mem. size: %d KB).\n", num_nodes, max_depth, kd_tree_size);
	KDNode* dv_flat_tree = NULL;
	cudaOk(cudaMalloc(&dv_flat_tree, num_nodes * sizeof(KDNode)));
	cudaOk(cudaMemcpy(dv_flat_tree, host_flat_tree, num_nodes * sizeof(KDNode), cudaMemcpyHostToDevice));
	dv_mem_kdtrees_ptrs.push_back(dv_flat_tree);
	gptr->flat_kd_root = dv_flat_tree;

	KDNode* dv_kdtree = copyKDTreeToCuda(host_kdtree);
	gptr->kdroot = dv_kdtree;
	dv_mem_kdtrees_ptrs.push_back(dv_kdtree);

	cudaOk(cudaMalloc(&dv_gptr, sizeof(MeshGeometryData)));
	result.type = TriangleMeshObj;
	result.materials = dv_materials;
	result.geometry_data = dv_gptr;
	cudaOk(cudaMemcpy(dv_gptr, gptr, sizeof(MeshGeometryData), cudaMemcpyHostToDevice));

	if (!checkCudaError("Loading Mesh Object"))
		printf("  SUCCESS: Loaded Mesh Object: %s\n\n", info->src_filename);
	dv_mem_ptrs.push_back(dv_gptr);
	free(gptr);
	free(host_flat_tree);
	free(host_materials);
	freeKDTree(host_kdtree);
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

void loadSceneWorldObjects(const Json::Value& jscene) {
	printSep();
	printf("Memory struct sizes:\n");
	printf("Sizeof: %s = %d\n", "WorldObject", sizeof(WorldObject));
	printf("Sizeof: %s = %d\n", "KDNode", sizeof(KDNode));
	printf("Sizeof: %s = %d\n", "BBox", sizeof(BBox));
	printf("Sizeof: %s = %d\n", "MeshGeometryData", sizeof(MeshGeometryData));
	printf("Sizeof: %s = %d\n", "SphereGeometryData", sizeof(SphereGeometryData));
	printf("Sizeof: %s = %d\n", "Triangle", sizeof(Triangle));
	printf("Sizeof: %s = %d\n", "Material", sizeof(Material));
	printSep();
	try {
		initWorldObjSources(jscene);
		scene.dv_wobjects_ptr = loadWorldObjects();
	}
	catch (const scene_file_error& e) {
		freeWorldObjects();
		throw scene_file_error(e.what());
	}
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
