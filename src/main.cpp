#include "main.h"
#include "view.h"
#include "world.h"
#include "errors.h"
#include "cudaUtility.h"
#include "kernel.h"
#include "config.h"
#include "json/json.h"
#include <fstream>

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


bool parseJsonScene(Json::Value& root, std::string& errmsg) {
	Json::Reader reader;
	std::ifstream jstream(input_file, std::ifstream::binary);
	bool parsingSuccessful = reader.parse(jstream, root, false);
	if (!parsingSuccessful)
		errmsg = reader.getFormattedErrorMessages();
	return parsingSuccessful;
}


int main(int argc, char* argv[]) {
	printf("Starting path-tracer application.\n");
	if (argc > 1)
		input_file = std::string(argv[1]);
	else {
		printf("Usage: command <scene_file.json>.");
		exit(EXIT_FAILURE);
	}

	// parse scene file
	Json::Value jscene;
	std::string errmsg;
	if (!parseJsonScene(jscene, errmsg)) {
		printf("Error in scene JSON: %s\n", errmsg.c_str());
		exit(EXIT_FAILURE);
	}
	
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

	try {
		printf("Initializing world elements...\n");
		worldInit(jscene);

		printf("Initializing preview window...\n");
		viewInit(jscene["camera"]);
	}
	catch (const Json::Exception& je) {
		printf("Error while parsing scene file: %s\n", je.what());
		exit(EXIT_FAILURE);
	}
	catch (const scene_file_error& err) {
		printf("Error while loading scene: %s\n", err.what());
		exit(EXIT_FAILURE);
	}

	printf("Starting preview loop...\n");
	viewLoop();
	printf("Exiting program...\n");
	exit(EXIT_SUCCESS);
}

void printSep() {
	 printf("------------------------------------------------------------\n");
};