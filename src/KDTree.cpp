#include "KDTree.h"
#include "cutil_math.h"
#include "utility.h"
#include "cudaUtility.h"
#include "world.h"
#include "world_load.h"
#include <vector>


static std::vector<dv_ptr> dv_tree_mem_ptrs;


bool triangleInBBox(Triangle& tr, BBox& bbox) {
	return true;
}


float3 triangleMiddle(Triangle& tr) {
	return (tr.a + tr.b + tr.c) / 3.0f;
}


float trs_median(Triangle* trs_src, int* trs_idxs, int num_trs, int axis) {
	float* coords = (float*)malloc(num_trs * sizeof(float));

	switch (axis) {
		case 0: for (int i = 0; i < num_trs; i++)
				{
					float3 midp = triangleMiddle(trs_src[trs_idxs[i]]);
					coords[i] = midp.x;
				}; break;

		case 1: for (int i = 0; i < num_trs; i++)
				{
					float3 midp = triangleMiddle(trs_src[trs_idxs[i]]);
					coords[i] = midp.y;
				}; break;

		case 2: for (int i = 0; i < num_trs; i++)
				{
					float3 midp = triangleMiddle(trs_src[trs_idxs[i]]);
					coords[i] = midp.z;
				}; break;
	}
	

	float coords_median = fmedian(coords, num_trs);
	free(coords);
	return coords_median;
}

BBox makeTrgBBox(Triangle& trg) {
	BBox bbox;
	bbox.bounds[0].x = fmin(trg.a.x, fmin(trg.b.x, trg.c.x));
	bbox.bounds[0].y = fmin(trg.a.y, fmin(trg.b.y, trg.c.y));
	bbox.bounds[0].z = fmin(trg.a.z, fmin(trg.b.z, trg.c.z));

	bbox.bounds[1].x = fmax(trg.a.x, fmax(trg.b.x, trg.c.x));
	bbox.bounds[1].y = fmax(trg.a.y, fmax(trg.b.y, trg.c.y));
	bbox.bounds[1].z = fmax(trg.a.z, fmax(trg.b.z, trg.c.z));
	return bbox;
}

void expandBBox(BBox& bbox, Triangle& tr) {
	BBox trbbox = makeTrgBBox(tr);
	float3& trvmin = trbbox.bounds[0];
	float3& trvmax = trbbox.bounds[1];
	float3& vmin = bbox.bounds[0];
	float3& vmax = bbox.bounds[1];

	vmin.x = fmin(vmin.x, trvmin.x);
	vmin.y = fmin(vmin.y, trvmin.y);
	vmin.z = fmin(vmin.z, trvmin.z);

	vmax.x = fmax(vmax.x, trvmax.x);
	vmax.y = fmax(vmax.y, trvmax.y);
	vmax.z = fmax(vmax.z, trvmax.z);
}

KDNode* buildKDTree(Triangle* trg_src, int*& trg_idxs, int num_trs, int depth) {
	KDNode* node = (KDNode*)malloc(sizeof(KDNode));
	node->trg_idxs = trg_idxs;
	node->num_trgs = num_trs;
	node->left = node->right = NULL;
	/*printf("    Depth: %d, Node: [triangles: %d] ", depth, num_trs);
	printf("idx: [");
	for (int i = 0; i < num_trs; i++)
	{
		printf("%d,", trg_idxs[i]);
	}
	printf("]\n");*/
	// initial cases
	if (num_trs == 0)
		return node;

	if (num_trs == 1) {
		node->bbox = makeTrgBBox(trg_src[trg_idxs[0]]);
		node->left = NULL;// (KDNode*)malloc(sizeof(KDNode));
		node->right = NULL;//(KDNode*)malloc(sizeof(KDNode));
		//node->left->trg_idxs = node->left->trg_idxs = NULL;
		return node;
	}

	// construct bounding box from all triangles
	node->bbox = makeTrgBBox(trg_src[trg_idxs[0]]);
	for (int i = 0; i < num_trs; i++) {
		int debug_idx = trg_idxs[i];
		expandBBox(node->bbox, trg_src[trg_idxs[i]]);
	}

	// get axis, order is X -> Y -> Z -> X -> Y -> ...
	// X = 0, Y = 1, Z = 2
	int axis = depth % 3;
	float median = trs_median(trg_src, trg_idxs, num_trs, axis);

	// divide triangles into tree nodes
	std::vector<int> trg_left_idxs, trg_right_idxs;

	for (int i = 0; i < num_trs; i++)
	{
		int trg_idx = trg_idxs[i];
		if (axis == 0)
			(triangleMiddle(trg_src[trg_idx]).x > median) ? trg_left_idxs.push_back(trg_idx) : trg_right_idxs.push_back(trg_idx);
		else if (axis == 1)
			(triangleMiddle(trg_src[trg_idx]).y > median) ? trg_left_idxs.push_back(trg_idx) : trg_right_idxs.push_back(trg_idx);
		else if (axis == 2)
			(triangleMiddle(trg_src[trg_idx]).z > median) ? trg_left_idxs.push_back(trg_idx) : trg_right_idxs.push_back(trg_idx);
	}

	int left_size = trg_left_idxs.size();
	int right_size = trg_right_idxs.size();

	if (left_size > 6) {
		int* left_idxs = (int*)malloc(left_size * sizeof(int));
		memcpy(left_idxs, trg_left_idxs.data(), left_size * sizeof(int));
		node->left = buildKDTree(trg_src, left_idxs, left_size, depth + 1);

		int* right_idxs = (int*)malloc(right_size * sizeof(int));
		memcpy(right_idxs, trg_right_idxs.data(), right_size * sizeof(int));
		node->right = buildKDTree(trg_src, right_idxs, right_size, depth + 1);
	}

	return node;
}

KDNode* copyKDTreeToCuda(KDNode* host_root) {
	if (host_root == NULL)
		return NULL;
	KDNode copy_root;
	copy_root.left = copyKDTreeToCuda(host_root->left);
	copy_root.right = copyKDTreeToCuda(host_root->right);

	int* dv_trg_idxs = NULL;
	if (host_root->num_trgs > 0) {
		cudaOk(cudaMalloc(&dv_trg_idxs, host_root->num_trgs * sizeof(int)));
		dv_tree_mem_ptrs.push_back(dv_trg_idxs);
		cudaOk(cudaMemcpy(dv_trg_idxs, host_root->trg_idxs, host_root->num_trgs * sizeof(int), cudaMemcpyHostToDevice));
	}
	copy_root.trg_idxs = dv_trg_idxs;
	copy_root.bbox = host_root->bbox;
	copy_root.num_trgs = host_root->num_trgs;

	KDNode* dv_node_ptr = NULL;
	cudaOk(cudaMalloc(&dv_node_ptr, sizeof(KDNode)));
	dv_tree_mem_ptrs.push_back(dv_node_ptr);
	cudaOk(cudaMemcpy(dv_node_ptr, &copy_root, sizeof(KDNode), cudaMemcpyHostToDevice));
	checkCudaError("copyKDTreeToCuda");

	return dv_node_ptr;
}

void freeKDTree(KDNode* root) {
	if (root == NULL)
		return;
	if (root->left != NULL)
		freeKDTree(root->left);
	if (root->left != NULL)
		freeKDTree(root->right);
	if (root->trg_idxs != NULL)
		free(root->trg_idxs);
	free(root);
	root = NULL;
}

void freeCudaKDTree(KDNode* root) {
	for (int i = 0; i < dv_tree_mem_ptrs.size(); i++)
	{
		cudaFree(dv_tree_mem_ptrs[i]);
	}
	dv_tree_mem_ptrs.clear();
}
