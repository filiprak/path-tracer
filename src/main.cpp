#include "main.h"
#include "world.h"
#include "errors.h"
#include "cudaUtility.h"
#include "kernel.h"
#include "config.h"

#include "qt5gui.h"

std::string input_file;


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


int detectCUDAdevice() {
	// Choose which GPU to run on, change this on a multi-GPU system.
	int dvNum = printCudaDevicesInfo();
	if (dvNum < 1) {
		
		getchar();
		exit(EXIT_FAILURE);
	}

	int cuda_device_id = 0;
	cudaSetDevice(cuda_device_id);
	return dvNum;
}

int main(int argc, char* argv[]) {
	// run Qt
	return runQt5(argc, argv);
}

void printSep() {
	 printf("------------------------------------------------------------\n");
};