#pragma once

#include "world.h"
#include "cuda.h"
#include "config.h"
#include "cuda_runtime.h"
#include "cutil_math.h"
#include "cudaUtility.h"

#define EPSILON		0.00000005f

typedef struct {
	float3 ipoint, normal, bary_coords;
	Triangle itrg;

	int iobj_index;
	const Material* imat;

	//debug
	float3 debug_mask;

} IntersectInfo;


/* Möller–Trumbore ray-triangle intersection algorithm */
__device__
bool rayIntersectsTriangle(Ray* ray, float3* va, float3* vb, float3* vc, float* res_dist, float* ru, float* rv)
{
	float3 edge1, edge2, h;
	float a;
	edge1 = *vb - *va;
	edge2 = *vc - *va;

	h = cross(ray->direction, edge2);
	a = dot(edge1, h);

	if (a > -EPSILON && a < EPSILON)
		return false;

	float f = __fdividef(1.0, a);
	float3 s = ray->originPoint - *va;
	float u = f * (dot(s, h));
	if (u < 0.0 || u > 1.0)
		return false;

	float3 q = cross(s, edge1);
	float v = f * (dot(ray->direction, q));
	if (v < 0.0 || u + v > 1.0)
		return false;

	// At this stage we can compute t to find out where the intersection point is on the line.
	float t = f * dot(edge2, q);
	if (t > EPSILON) // ray intersection
	{
		*res_dist = t;
		*ru = u;
		*rv = v;
		return true;
	}
	// This means that there is a line intersection but not a ray intersection.
	return false;
}

__device__
bool testSphereIntersection(Ray* ray, const WorldObject* obj, float3* hit_point, float3* hit_norm) {
	SphereGeometryData* gdata = (SphereGeometryData*)obj->geometry_data;
	float rad = gdata->radius;

	float3 op = gdata->position - ray->originPoint;
	float t, epsilon = 0.0001f;
	float b = dot(op, ray->direction);
	float disc = b*b - dot(op, op) + rad*rad; // discriminant

	if (disc < 0)
		return false;
	else disc = sqrtf(disc);

	t = b - disc;
	if (t < epsilon)
		t = b + disc;
	if (t < epsilon)
		return false;

	/* calculate hit point and normal to surface */
	*hit_point = ray->originPoint + ray->direction * t;
	*hit_norm = normalize(*hit_point - gdata->position);

	return true;
}

/*__device__
bool testTriangleMeshIntersection(Ray* ray, WorldObject* obj, TriangleMesh* imesh, float3* hit_point, float3* hit_norm) {
	MeshGeometryData* gdata = (MeshGeometryData*)obj.geometry_data;

	int inters_mesh_idx;
	float closest_dist = HUGE_VALF;
	bool intersects = false;
	bool culling = false;

	for (int t = 0; t < gdata->num_triangles; ++t) {
		Triangle& trg = gdata->triangles[t];
		if (culling && dot(ray->direction, trg.norm_a) > 0) // skip triangles turned back to ray
			continue;

		float inters_dist;
		bool triangle_intersects = rayIntersectsTriangle(ray,
			trg.a,
			trg.b,
			trg.c,
			inters_dist);

		if (triangle_intersects && !intersects) {
			intersects = true;
		}
		// check if point is closest to viewer
		if (triangle_intersects && inters_dist < closest_dist) {
			closest_dist = inters_dist;
			hit_norm = trg.norm_a;
		}
	}
	if (intersects) {
		hit_point = ray->originPoint + closest_dist * ray->direction;
	}
	return intersects;
}*/

/* based on: https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection */
__device__
bool testBBoxIntersection(BBox& bbox, Ray* ray, float* tmin) {
	float t[8];

	t[1] = (bbox.bounds[ray->sign[0]].x - ray->originPoint.x) * ray->inv_direction.x;
	t[2] = (bbox.bounds[1 - ray->sign[0]].x - ray->originPoint.x) * ray->inv_direction.x;
	t[3] = (bbox.bounds[ray->sign[1]].y - ray->originPoint.y) * ray->inv_direction.y;
	t[4] = (bbox.bounds[1 - ray->sign[1]].y - ray->originPoint.y) * ray->inv_direction.y;

	if ((t[1] > t[4]) || (t[3] > t[2]))
		return false;
	if (t[3] > t[1])
		t[1] = t[3];
	if (t[4] < t[2])
		t[2] = t[4];

	t[5] = (bbox.bounds[ray->sign[2]].z - ray->originPoint.z) * ray->inv_direction.z;
	t[6] = (bbox.bounds[1 - ray->sign[2]].z - ray->originPoint.z) * ray->inv_direction.z;

	if ((t[1] > t[6]) || (t[5] > t[2]))
		return false;
	if (t[5] > t[1])
		t[1] = t[5];
	if (t[6] < t[2])
		t[2] = t[6];

	*tmin = t[1] < 0 ? 0.0f : t[1];
	return true;
}


// Iteration loop version instead of recursive ray KDNode intersection
__device__
bool rayIntersectsKDNodeLOOP(Ray* ray,
							Triangle* trs,
							KDNode* flat_nodes,
							float3* bary_coords,
							Triangle* itrg,
							float* tmin,
							float3* debug_mask) {

	int curr_idx = 0;
	bool intersects = false;

	// check root node intersection
	float dist = HUGE_VALF;
	if (!testBBoxIntersection(flat_nodes[curr_idx].bbox, ray, &dist))
		return intersects;

	// init short stack
	int stack_pos = -1;
	int stack[64];
	memset(stack, -1, sizeof(stack));

	while (true) {
		// get current node
		KDNode& node = flat_nodes[curr_idx];

		if (node.left_idx == -1 && node.right_idx == -1) {
			// we are in leaf node: testing all triangles from node
			float t, u, v;
			for (int i = 0; i < node.num_trgs; i++)
			{
				Triangle& trg = trs[node.trg_idxs[i]];
				//if (dot(ray->direction, trg.norm_a) >= 0) continue;
				if (rayIntersectsTriangle(ray, &(trg.a), &(trg.b), &(trg.c), &t, &u, &v)) {
					intersects = true;
					if (t < *tmin) {
						*tmin = t;
						*itrg = trg;
						bary_coords->y = u;
						bary_coords->z = v;
					}
				}
			}
			// get next node from stack
			if (stack_pos < 0) 
				return intersects; // break if all elements from stack was traversed
			curr_idx = stack[stack_pos];
			stack[stack_pos] = -1;
			stack_pos--;
			continue;
		}

		float left_dist = HUGE_VALF, right_dist = HUGE_VALF;
		bool left_res = false, right_res = false;

		// test left and right node for intersection
		if (node.left_idx != -1)
			left_res = testBBoxIntersection(flat_nodes[node.left_idx].bbox, ray, &left_dist) && (left_dist < *tmin);
		if (node.right_idx != -1)
			right_res = testBBoxIntersection(flat_nodes[node.right_idx].bbox, ray, &right_dist) && (right_dist < *tmin);

		if (left_res && right_res) {
			// choose closer node to traverse, and push further node on stack for later visiting
			int further_idx = node.right_idx;
			curr_idx = node.right_idx;
			left_dist > right_dist ? (further_idx = node.left_idx) : (curr_idx = node.left_idx);
			stack_pos++;
			stack[stack_pos] = further_idx;
			continue;
		}
		else if (left_res) {
			// if one of nodes was hit then go to this node: left
			curr_idx = node.left_idx;
			continue;
		}
		else if (right_res) {
			// if one of nodes was hit then go to this node: right
			curr_idx = node.right_idx;
			continue;
		}
		else {
			// if both nodes was missed then pop next node from stack
			if (stack_pos < 0)
				return intersects;
			curr_idx = stack[stack_pos];
			stack[stack_pos] = -1;
			stack_pos--;
		}
	}
}

__device__
bool rayIntersectsKDNode(Ray* ray, Triangle* trs, int trg_tex, KDNode* node, float3* norm, int* mat_idx, float* tmin, float3* debug_mask) {
	if (node == NULL)
		return false;
	float t;
	if (!testBBoxIntersection(node->bbox, ray, &t))
		return false;
#ifdef DEBUG_BBOXES
	debug_mask *= 0.95f;
#endif // DEBUG_BBOXES

	bool int_left = false, int_right = false;
	bool left = node->left != NULL, right = node->right != NULL;
	if (left || right) {
		if (node->left != NULL)
			int_left = rayIntersectsKDNode(ray, trs, trg_tex, node->left, norm, mat_idx, tmin, debug_mask);
		if (node->right != NULL)
			int_right = rayIntersectsKDNode(ray, trs, trg_tex, node->right, norm, mat_idx, tmin, debug_mask);
		return int_left || int_right;
	}
	else { // leaf node of kd tree
		float t, u, v;
		bool intersects = false;
		for (int i = 0; i < node->num_trgs; i++)
		{
#ifdef USE_TRIANGLE_TEXTURE_MEM
			int trg_i = 6 * i;
			float4 a = tex1Dfetch<float4>(trg_tex, trg_i);
			float4 b = tex1Dfetch<float4>(trg_tex, trg_i + 1);
			float4 c = tex1Dfetch<float4>(trg_tex, trg_i + 2);
			if (rayIntersectsTriangle(ray, make_float3(a.x, a.y, a.z),
				make_float3(b.x, b.y, b.z),
				make_float3(c.x, c.y, c.z), t)) {
				intersects = true;
				if (t < tmin) {
					tmin = t;
					n_idx = trg_i;
				}
			}
#else
			Triangle& trg = trs[node->trg_idxs[i]];
			//if (dot(ray->direction, trg.norm_a) >= 0) continue;
			if (rayIntersectsTriangle(ray, &(trg.a), &(trg.b), &(trg.c), &t, &u, &v)) {
				intersects = true;
				if (t < *tmin) {
					*tmin = t;
					*norm = trg.norm_a;
					*mat_idx = trg.material_idx;
				}
			}
#endif
		}
#ifdef USE_TRIANGLE_TEXTURE_MEM
		if (intersects) {
			float4 n = tex1Dfetch<float4>(trg_tex, n_idx + 3);
			norm = make_float3(n.x, n.y, n.z);
		}
#endif
		return intersects;
	}
}

__device__
bool rayIntersectsObject(Ray* ray,
						const WorldObject* obj,
						const Material*& mat,
						Triangle* trg,
						float3* hit_point,
						float3* hit_norm,
						float3* bary_coords,
						float3* debug_mask) {

	bool intersects = false;

	if (obj->type == SphereObj) {
		intersects = testSphereIntersection(ray, obj, hit_point, hit_norm);
		mat = &(obj->materials[0]);
	}
	else if (obj->type == TriangleMeshObj) {
		MeshGeometryData* gdata = (MeshGeometryData*)obj->geometry_data;

		float tmin = HUGE_VALF;
#ifdef USE_KD_TREES
	#ifdef USE_SHORT_STACK_LOOP
		intersects = rayIntersectsKDNodeLOOP(ray, gdata->triangles, gdata->flat_kd_root, bary_coords, trg, &tmin, debug_mask);
	#else
		intersects = rayIntersectsKDNode(ray, gdata->triangles, gdata->triangles_tex, gdata->kdroot, hit_norm, res_mat_idx, tmin, debug_mask);
	#endif
#else
		float t, u, v;
		int n_idx; //normal index
		for (int i = 0; i < gdata->num_triangles; i++)
		{
			
	#ifdef USE_TRIANGLE_TEXTURE_MEM
			int trg_i = 6 * i;
			float4 a = tex1Dfetch<float4>(gdata->triangles_tex, trg_i);
			float4 b = tex1Dfetch<float4>(gdata->triangles_tex, trg_i + 1);
			float4 c = tex1Dfetch<float4>(gdata->triangles_tex, trg_i + 2);
	#else
				//printf("[%f,%f,%f,%f]\n", a.x, a.y, a.z, a.w);
			Triangle& trg = gdata->triangles[i];
	#endif
				//if (dot(ray->direction, trg.norm_a) >= 0) continue;
	#ifdef USE_TRIANGLE_TEXTURE_MEM
			if (rayIntersectsTriangle(ray, make_float3(a.x, a.y, a.z),
				make_float3(b.x, b.y, b.z),
				make_float3(c.x, c.y, c.z), t, u, v)) {
				intersects = true;
				if (t < *tmin) {
					*tmin = t;
					n_idx = trg_i;
				}
			}
	#else
			if (rayIntersectsTriangle(ray, &(trg.a), &(trg.b), &(trg.c), &t, &u, &v)) {
				intersects = true;
				if (t < *tmin) {
					*tmin = t;
					n_idx = i;
				}
			}
		
	#endif
		}
#endif
		if (intersects)
		{
			*hit_point = ray->originPoint + tmin * ray->direction;
			bary_coords->x = 1.0f - bary_coords->y - bary_coords->z;
			*hit_norm = normalize(bary_coords->x * trg->norm_a + bary_coords->y * trg->norm_b + bary_coords->z * trg->norm_c);
			mat = &(obj->materials[trg->material_idx]);

#ifndef USE_KD_TREES
	#ifdef USE_TRIANGLE_TEXTURE_MEM
			float4 n = tex1Dfetch<float4>(gdata->triangles_tex, n_idx + 3);
			*hit_norm = make_float3(n.x, n.y, n.z);
	#else
			*hit_norm = gdata->triangles[n_idx].norm_a;
			
	#endif
#endif
		}
	}
	return intersects;
}

__device__
bool rayIntersectsScene(Ray* ray, Scene& scene, IntersectInfo& ii) {
	bool intersects = false;
	float closest_dist = HUGE_VALF;
	float3 inters_point, inters_norm, bary_coords, debug_mask;

	for (int i = 0; i < scene.num_wobjects; ++i) {
		const WorldObject* obj = &(scene.dv_wobjects_ptr[i]);
		const Material* i_mat = &(obj->materials[0]);
		Triangle itrg;

		if (rayIntersectsObject(ray, obj, i_mat, &itrg, &inters_point, &inters_norm, &bary_coords, &debug_mask)) {
			intersects = true;
			float inters_dist = length(inters_point - ray->originPoint);
			if (inters_dist < closest_dist) {
				closest_dist = inters_dist;
				ii.iobj_index = i;
				ii.ipoint = inters_point;
				ii.normal = inters_norm;
				ii.imat = i_mat;
				ii.bary_coords = bary_coords;
				ii.debug_mask = debug_mask;
				ii.itrg = itrg;
			}
		};
	}
	return intersects;
}
