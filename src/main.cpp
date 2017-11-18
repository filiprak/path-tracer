#include "main.h"
#include "view.h"
#include "world.h"
#include "cudaUtility.h"
#include "kernel.h"
#include "config.h"


void printProgramConfig() {
	printSep();
	printf("Running configuration:\n\n");
#ifdef DEBUG_BBOXES
	printf("Debug AABB boxes mode: enabled\n");
#else
	printf("Debug AABB boxes mode: disabled\n");
#endif
#ifdef USE_KD_TREES
	printf("Use KD-Trees mode: enabled\n");
#else
	printf("Use KD-Trees mode: disabled\n");
#endif
#ifdef USE_TRIANGLE_TEXTURE_MEM
	printf("Store triangles data in CUDA texture memory: enabled\n");
#else
	printf("Store triangles data in CUDA texture memory: disabled\n");
#endif
#ifdef PRECOMPUTE_TRI_EDGES
	printf("Precomputing triangle edges: enabled\n");
#else
	printf("Precomputing triangle edges: disabled\n");
#endif
	printSep();
}

int main(int argc, char* argv[]) {
	/*if (argc > 1)
		scenefile = std::string(argv[1]);
	else
		scenefile = std::string("C:/Users/raqu/git/path-tracer/scenes/teapot.obj");*/

	printf("Starting path-tracer application.\n");
	int dvNum = printCudaDevicesInfo();
	if (dvNum < 1) {
		printf("CUDA devices not found.\nPlease ensure you have it installed.");
		exit(EXIT_FAILURE);
	}

	// Choose which GPU to run on, change this on a multi-GPU system.
	int cuda_device_id = 0;
	cudaSetDevice(cuda_device_id);
	if (checkCudaError("Unable to set CUDA device"))
		exit(EXIT_FAILURE);

	cudaDeviceSetLimit(cudaLimitStackSize, (size_t)MAX_STACK_CUDA_SIZE);
	if (checkCudaError("Unable to set CUDA device stack size.\n"))
		exit(EXIT_FAILURE);

	printf("Initializing preview window...\n");
	viewInit();

	printf("Initializing world elements...\n");
	worldInit();



	printf("Starting preview loop...\n");
	printSep();
	printf("Starting preview loop...\n");
	viewLoop();
	printf("Exiting program...\n");
	exit(EXIT_SUCCESS);
}

void printSep() {
	 printf("------------------------------------------------------------\n");
};