#pragma once

// cam properities
#define CAM_ROTATE_ANGLE_DELTA			5.0f // degrees
#define CAM_MOVE_DISTANCE_DELTA			0.2f // units

// path tracing parameters
#define MAX_NUM_RAY_BOUNCES				8// maximum number of traced ray bounces in path tracing
#define SCENE_REFRACTION_INDEX			1.0f // refraction index of scene empty space, air = 1.0

// CUDA parameters
// Maximum stack size have to be adjusted manually due to recursive
// KD-tree traversing, default size is too small for large KD-trees
#define MAX_STACK_SIZE					1024 * 50 // Bytes