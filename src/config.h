#pragma once

// cam properities
#define CAM_ROTATE_ANGLE_DELTA			5.0f // degrees
#define CAM_MOVE_DISTANCE_DELTA			0.2f // units

// path tracing parameters
#define MAX_NUM_RAY_BOUNCES				3// maximum number of traced ray bounces in path tracing
#define SCENE_REFRACTION_INDEX			1.0f // refraction index of scene empty space, air = 1.0

// CUDA parameters
// Maximum stack size have to be adjusted manually due to recursive
// KD-tree traversing, default size is too small for large KD-trees
#define MAX_STACK_CUDA_SIZE					1024 * 50 // Bytes

// Renders AABB boxes of scene objects using simple shading
//#define DEBUG_BBOXES

// Use KD-tree for fast triangle mesh intersection detecting
#define USE_KD_TREES

// Use CUDA texture memory to store triangles data
#define USE_TRIANGLE_TEXTURE_MEM

// Precompute triangle edges, triangle is stored then as (a, b - a, c - a)
// when normally it is stored as (a, b, c)
#define PRECOMPUTE_TRI_EDGES