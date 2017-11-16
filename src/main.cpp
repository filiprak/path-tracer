#include "main.h"
#include "view.h"
#include "world.h"
#include "cudaUtility.h"
#include "kernel.h"

std::string scenefile;


int main(int argc, char* argv[]) {
	if (argc > 1)
		scenefile = std::string(argv[1]);
	else
		scenefile = std::string("C:/Users/raqu/git/path-tracer/scenes/teapot.obj");

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

	cudaDeviceSetLimit(cudaLimitStackSize, (size_t) 4096 * 10);
	if (checkCudaError("Unable to set CUDA device stack size to: %d B\n"))
		exit(EXIT_FAILURE);

	printf("Initializing preview window...\n");
	viewInit();

	printf("Initializing world elements...\n");
	worldInit();

	printSep();
	printf("Starting preview loop...\n");
	viewLoop();
	printf("Exiting program...\n");
	exit(EXIT_SUCCESS);
}

void printSep() {
	 printf("------------------------------------------------------------\n");
};