#include "main.h"
#include "world.h"
#include "errors.h"
#include "cudaUtility.h"
#include "kernel.h"
#include "config.h"
#include "json/json.h"
#include <fstream>

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


bool parseJsonScene(std::string fpath, Json::Value& root, std::string& errmsg) {
	Json::Reader reader;
	std::ifstream jstream(fpath, std::ifstream::binary);
	if (!jstream.is_open()) {
		errmsg = "Unable to open file: " + input_file;
		return false;
	}
	bool parsingSuccessful = reader.parse(jstream, root, false);
	if (!parsingSuccessful)
		errmsg = reader.getFormattedErrorMessages();

	jstream.close();
	return parsingSuccessful;
}


int main(int argc, char* argv[]) {

	// Choose which GPU to run on, change this on a multi-GPU system.
	int cuda_device_id = 0;
	cudaSetDevice(cuda_device_id);
	if (checkCudaError("Unable to set CUDA device")) {
		getchar();
		exit(EXIT_FAILURE);
	}

	cudaDeviceSetLimit(cudaLimitStackSize, (size_t)MAX_STACK_CUDA_SIZE);
	if (checkCudaError("Unable to set CUDA device stack size.\n")) {
		getchar();
		exit(EXIT_FAILURE);
	}

	int dvNum = printCudaDevicesInfo();
	if (dvNum < 1) {
		printf("CUDA devices not found.\nPlease ensure you have it installed.");
		getchar();
		exit(EXIT_FAILURE);
	}

	// run Qt
	return runQt5(argc, argv);
}

void printSep() {
	 printf("------------------------------------------------------------\n");
};