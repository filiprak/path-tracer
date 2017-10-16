#include "main.h"
#include "view.h"
#include "cudaUtility.h"


int main() {
	printf("Starting path-tracer application.\n");
	int dvNum = printCudaDevicesInfo();
	if (dvNum < 1) {
		printf("CUDA devices not found.\nPlease ensure you have it installed.");
	}

	// Choose which GPU to run on, change this on a multi-GPU system.
	int cuda_device_id = 0;
	cudaSetDevice(cuda_device_id);
	if (checkCudaError("Unable to set CUDA device"))
		exit(EXIT_FAILURE);


	printf("Initializing preview window...\n");
	viewInit();

	printf("Starting preview loop...\n");
	viewLoop();

	exit(EXIT_SUCCESS);
}
