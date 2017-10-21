#include "main.h"
#include <cuda_runtime.h>


int printCudaDevicesInfo() {
	int nDevices;

	cudaGetDeviceCount(&nDevices);
	printf("Detected CUDA devices:\n");
	printSep();
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
		printf("  Clock Rate (KHz): %d\n", prop.clockRate);
		printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
		printf("  Total Global Memory (MB): %d\n", prop.totalGlobalMem / 1048576);
		printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
		printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %f\n\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8) / 1.0e6);
	}
	printSep();
	return nDevices;
}

bool checkCudaError(const char* context) {
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR (code %d): %s: message: '%s'\n", cudaStatus, context, cudaGetErrorString(cudaStatus));
		return true;
	}
	return false;
}
