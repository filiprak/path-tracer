#include "sceneloaderthread.h"
#include "world_load.h"
#include "scenestate.h"

SceneLoaderThread::SceneLoaderThread(QString src_file, DialogLogger* logger) :
logger(logger), src_file(src_file)
{

}

void SceneLoaderThread::run() {
	std::string errmsg;
	Json::Value jscene;

	logger->info("Parsing scene file: %s", (const char*)src_file.toLocal8Bit());
	if (parseJsonScene(std::string(src_file.toLocal8Bit()), jscene, errmsg)) {
		if(sceneState.load(jscene, logger))
			logger->green("Loaded scene: %s", (const char*)src_file.toLocal8Bit());
		else
			logger->error("Failed to load scene: %s", errmsg.c_str());
	} else
		logger->error("Scene file parse error: %s", errmsg.c_str());
}