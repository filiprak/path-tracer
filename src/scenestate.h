#pragma once
#include "dialoglogger.h"
#include "world.h"

#include <qobject.h>

// threadsafe scenestate object
class SceneState : public QObject
{
	Q_OBJECT
public:
	SceneState();
	~SceneState();

	bool load(const Json::Value& jscene, DialogLogger* logger);
	bool isLoaded();
	bool clean();

	Scene clone();

private:
	// 3D scene description
	Scene scene;

	QMutex mutex;
	bool loaded;

};

extern SceneState sceneState;

