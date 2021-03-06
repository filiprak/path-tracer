#include "world_load.h"
#include "world.h"
#include "main.h"
#include "errors.h"
#include <fstream>
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
#include "cudaUtility.h"
#include "cutil_math.h"

/* glm */
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>


static DialogLogger* reporter = NULL;
#define REPORT(t) if (reporter) reporter->t;

WorldObjectsSources world_obj_sources;

// cuda device resources collections
static std::vector<dv_ptr> dv_mem_ptrs;
static std::vector<cudaTextureObject_t> dv_textures;
static std::vector<dv_ptr> dv_mem_kdtrees_ptrs;


void initWorldObjSources(Scene& scene, const Json::Value& jscene) {
	// init load handlers
	world_obj_sources.loadFuncMapping[SphereObj] = loadSphereObj;
	world_obj_sources.loadFuncMapping[TriangleMeshObj] = loadTriangleMeshObj;
	
	// parse scene from json description
	int num_jobjects = min((jscene["objects"].isArray() ? jscene["objects"].size() : 0), MAX_OBJECTS_NUM);
	
	int obj_idx = 0;
	for (int i = 0; i < num_jobjects; i++)
	{
		const Json::Value& jobj = jscene["objects"][i];
		bool render = jobj["render"].asBool();
		if (!render) {
			continue;
		}
		std::string type = jobj["type"].asString();

		if (!type.compare("sphere")) {
			world_obj_sources.sources[obj_idx].type = SphereObj;

			Material msphere;
			msphere.color = resolveFloat3(jobj["material"]["Kd"]);
			msphere.cuda_texture_obj = -1;
			msphere.type = resolveMatType(jobj["material"]["type"]);
			msphere.emittance = resolveFloat3(jobj["material"]["Ke"]);
			msphere.reflect_factor = resolveFloat(jobj["material"]["d"]);
			msphere.refract_index = resolveFloat(jobj["material"]["Ni"]);
			msphere.sharpness = clamp(resolveFloat(jobj["material"]["Ns"]) / 1000.0f, 0.0f, 1.0f);

			SphereObjInfo* sinfo = (SphereObjInfo*)malloc(sizeof(SphereObjInfo));
			sinfo->material = msphere;
			sinfo->position = resolveFloat3(jobj["position"]);
			sinfo->radius = resolveFloat(jobj["radius"]);
			world_obj_sources.sources[obj_idx].worldObjectInfo = sinfo;
			obj_idx++;
		}
		else if (!type.compare("mesh")) {
			world_obj_sources.sources[obj_idx].type = TriangleMeshObj;
			TriangleMeshObjInfo* minfo = (TriangleMeshObjInfo*)malloc(sizeof(TriangleMeshObjInfo));
			const char* src = jobj["src"].asCString();
			strcpy(minfo->src_filename, src);
			
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
			world_obj_sources.sources[obj_idx].worldObjectInfo = minfo;
			obj_idx++;
		}
		else {
			freeWorldObjects();
			throw scene_file_error("Unknown object type, supported types: mesh/sphere");
		}
	}// for

	// set scene object number
	scene.num_wobjects = world_obj_sources.num_objects = obj_idx;
}

void freeWorldObjSources() {
	// create world objects - compose our scene
	for (int i = 0; i < world_obj_sources.num_objects; ++i) {
		if (world_obj_sources.sources[i].worldObjectInfo != NULL)
			free(world_obj_sources.sources[i].worldObjectInfo);
	}
	for (int i = 0; i < MAX_OBJECTS_NUM; i++)
	{
		world_obj_sources.sources[i].worldObjectInfo = NULL;
	}
	world_obj_sources.num_objects = 0;
}

__host__
cudaTextureObject_t loadTexture(const std::string src_path)
{
	int width = 0, height = 0, channels = 0;
	float* img = stbi_loadf(src_path.c_str(), &width, &height, &channels, 0);
	if (!img) // failed to load texture file
		return -1;

	float4* host_tex_data = (float4*)malloc(height * width * sizeof(float4));
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			int index = channels * (row * width + col);
			float rgba[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
			for (int channel = 0; channel < channels; channel++)
				rgba[channel] = img[index + channel];

			int host_index = (height - row - 1) * width + (col);
			if (channels < 2)
				host_tex_data[host_index] = make_float4(rgba[0], rgba[0], rgba[0], rgba[3]);
			else if (channels < 3)
				host_tex_data[host_index] = make_float4(rgba[0], rgba[0], rgba[0], rgba[1]);
			else
				host_tex_data[host_index] = make_float4(rgba[0], rgba[1], rgba[2], rgba[3]);
		}
	}
	stbi_image_free(img);
	// allocate and copy pitch2D on cuda device
	float4* dev_tex_data = NULL;
	size_t pitch;
	cudaOk(cudaMallocPitch(&dev_tex_data, &pitch, width * sizeof(float4), height));
	cudaOk(cudaMemcpy2D(dev_tex_data, pitch, host_tex_data,
		width * sizeof(float4), width * sizeof(float4), height, cudaMemcpyHostToDevice));
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
	if (checkCudaError("loadTexture()"))
		return -1;
	return tex;
}

Material* loadMaterialsToHost(const aiScene* aiscene) {
	if (!aiscene->HasMaterials())
		return NULL;

	int num_materials = aiscene->mNumMaterials;
	Material* mat_ptr = (Material*)malloc(num_materials * sizeof(Material));
	for (int i = 0; i < num_materials; ++i)
	{
		aiMaterial* material = aiscene->mMaterials[i];

		// parse name, convention name_of_material.type_of_material, example: metal.spec, sun.lumi
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
		 * Kd = color
		 * Ke = emitance
		 * d = reflect_factor
		 * Ni = refract_index
		 * Ns = sharpness
		 */
		aiColor3D Kd, Ke;
		float d, Ni, Ns;
		aiString texture_path;

		printf("  Loading material[%d]: name: %s, type: %s\n", i, name.C_Str(), mat_type.c_str());
		REPORT(info("  Loading material[%d]: name: %s, type: %s", i, name.C_Str(), mat_type.c_str()));

		material->Get(AI_MATKEY_COLOR_DIFFUSE, Kd);
		material->Get(AI_MATKEY_COLOR_EMISSIVE, Ke);
		material->Get(AI_MATKEY_REFRACTI, Ni);
		material->Get(AI_MATKEY_SHININESS, Ns);
		material->Get(AI_MATKEY_OPACITY, d);
		// assimp error fix with Ns value
		Ns /= 4.0f;
		
		mat_ptr[i].cuda_texture_obj = -1;
		int num_tex_ambient = material->GetTextureCount(aiTextureType_AMBIENT);
		int num_tex_diffuse = material->GetTextureCount(aiTextureType_DIFFUSE);
		int num_tex_emissive = material->GetTextureCount(aiTextureType_EMISSIVE);
		if (num_tex_ambient > 0) {
			material->GetTexture(aiTextureType_AMBIENT, 0, &texture_path);
			printf("    Loading ambient texture: %s\n", texture_path.C_Str());
			REPORT(info("    Loading ambient texture: %s", texture_path.C_Str()));
			mat_ptr[i].cuda_texture_obj = loadTexture(std::string(texture_path.C_Str()));
		}
		else if (num_tex_diffuse > 0) {
			material->GetTexture(aiTextureType_DIFFUSE, 0, &texture_path);
			printf("    Loading diffuse texture: %s\n", texture_path.C_Str());
			REPORT(info("    Loading diffuse texture: %s", texture_path.C_Str()));
			mat_ptr[i].cuda_texture_obj = loadTexture(std::string(texture_path.C_Str()));
		}
		else if (num_tex_emissive > 0) {
			material->GetTexture(aiTextureType_EMISSIVE, 0, &texture_path);
			printf("    Loading emissive texture: %s\n", texture_path.C_Str());
			REPORT(info("    Loading emissive texture: %s", texture_path.C_Str()));
			mat_ptr[i].cuda_texture_obj = loadTexture(std::string(texture_path.C_Str()));
		}
		if (mat_ptr[i].cuda_texture_obj < 0 && num_tex_ambient + num_tex_diffuse + num_tex_emissive > 0) {
			printf("    Warning: texture load failed\n");
			REPORT(warning("    Warning: texture load failed: %s", texture_path.C_Str()));
		}
		
		mat_ptr[i].type = resolveMatType(mat_type);
		mat_ptr[i].color = make_float3(Kd.r, Kd.g, Kd.b);
		mat_ptr[i].emittance = make_float3(Ke.r, Ke.g, Ke.b);
		mat_ptr[i].reflect_factor = d;
		mat_ptr[i].refract_index = Ni;
		mat_ptr[i].sharpness = clamp(Ns / 1000.0f, 0.0f, 1.0f);
		
		printf("    enum-type: %d\n    color: [%.1f,%.1f,%.1f]\n    emit: [%.1f,%.1f,%.1f]\n    reflect: %f, refract: %f, sharp: %f\n",
			mat_ptr[i].type,
			mat_ptr[i].color.x, mat_ptr[i].color.y, mat_ptr[i].color.z,
			mat_ptr[i].emittance.x, mat_ptr[i].emittance.y, mat_ptr[i].emittance.z,
			mat_ptr[i].reflect_factor, mat_ptr[i].refract_index, mat_ptr[i].sharpness);
	}
	return mat_ptr;
}

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
	for (unsigned m = 0; m < aiscene->mNumMeshes; m++)
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

			float3 a = make_float3(ga.x, ga.y, ga.z);
			float3 b = make_float3(gb.x, gb.y, gb.z);
			float3 c = make_float3(gc.x, gc.y, gc.z);
			tr_host_ptr[i].a = a;
			tr_host_ptr[i].b = b;
			tr_host_ptr[i].c = c;
			tr_host_ptr[i].e1 = b - a;
			tr_host_ptr[i].e2 = c - a;
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
			for (unsigned i = loaded_trgs; i < loaded_trgs + mesh->mNumFaces; ++i) {
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
	REPORT(info("Loading Sphere Object: r=%f, pos=[%f,%f,%f]", info->radius, info->position.x, info->position.y, info->position.z));
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
	REPORT(info("Loading Mesh Object: %s", info->src_filename));
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
		for (unsigned i = 0; i < aiscene->mNumTextures; i++)
		{
			const aiTexture* aitex = aiscene->mTextures[i];
			printf(" Loading texture[%d]: %s %dx%d\n", i, aitex->achFormatHint, aitex->mWidth, aitex->mHeight);
			REPORT(info(" Loading texture[%d]: %s %dx%d", i, aitex->achFormatHint, aitex->mWidth, aitex->mHeight));
			for (unsigned x = 0; x < aitex->mWidth; x++)
			{
				for (unsigned y = 0; y < aitex->mHeight; y++)
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
	int trgs_mem_size = all_trgs_count * sizeof(Triangle) / 1024;
	printf("  Loaded %d triangles in %d meshes. (Glob. memory taken: %d KB)\n", gptr->num_triangles, aiscene->mNumMeshes, trgs_mem_size);
	REPORT(info("  Loaded %d triangles in %d meshes. (Glob. memory taken: %d KB)", gptr->num_triangles, aiscene->mNumMeshes, trgs_mem_size));

	// build kd tree
	printf("  Building KD-Tree for all triangles (%d).\n", all_trgs_count);
	REPORT(info("  Building KD-Tree for all triangles (%d).", all_trgs_count));
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
		REPORT(progress((int) (100.0f * (float)i / (float)num_objects)));
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

bool loadSceneWorldObjects(Scene& scene, const Json::Value& jscene, DialogLogger* logger) {
	printSep();

	try {
		// setup logger
		reporter = logger;
		initWorldObjSources(scene, jscene);
		scene.camera.init(jscene["camera"]);
		scene.dv_wobjects_ptr = loadWorldObjects();
		REPORT(progress(100));
		reporter = NULL;
		return true;
	}
	catch (const scene_file_error& e) {
		freeWorldObjects();
		REPORT(error("Scene file error: %s", e.what()));
		reporter = NULL;
		return false;
	}
}

void freeWorldObjects() {
	// free kd trees
	for (int i = 0; i < dv_mem_kdtrees_ptrs.size(); ++i) {
		if (dv_mem_kdtrees_ptrs[i])
			freeCudaKDTree((KDNode*)dv_mem_kdtrees_ptrs[i]);
	}
	// free allocated cuda memory
	for (int i = 0; i < dv_mem_ptrs.size(); ++i) {
		if (dv_mem_ptrs[i])
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

bool parseJsonScene(std::string fpath, Json::Value& root, std::string& errmsg) {
	Json::Reader reader;
	std::ifstream jstream(fpath, std::ifstream::binary);
	if (!jstream.is_open()) {
		errmsg = "Unable to open file: " + fpath;
		return false;
	}
	bool parsingSuccessful = reader.parse(jstream, root, false);
	if (!parsingSuccessful)
		errmsg = reader.getFormattedErrorMessages();

	jstream.close();
	return parsingSuccessful;
}
