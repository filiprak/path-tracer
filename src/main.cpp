#include "main.h"
#include "world.h"
#include "errors.h"
#include "cudaUtility.h"
#include "kernel.h"
#include "config.h"

#include "qt5gui.h"

int cuda_device_id = -1;


void printProgramConfig() {
	printSep();
#ifdef USE_KD_TREES
	printf("Use KD-Trees mode enabled\n");
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

	cuda_device_id = 0;
	cudaSetDevice(cuda_device_id);
#ifdef MAX_STACK_CUDA_SIZE
	if (!checkCudaError("cudaSetDevice()")) {
		cudaDeviceSetLimit(cudaLimitStackSize, (size_t)MAX_STACK_CUDA_SIZE);
		if (checkCudaError("Unable to set CUDA device stack size.\n"))
			exit(EXIT_FAILURE);
	}
#endif
	return dvNum;
}

int main(int argc, char* argv[]) {
	printProgramConfig();
	// run Qt
	return runQt5(argc, argv);
}

void printSep() {
	 printf("------------------------------------------------------------\n");
};