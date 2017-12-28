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

	void cameraChanged(bool, int* = NULL);

	void resetCamera();
	void setCamPositionX(float);
	void setCamPositionY(float);
	void setCamPositionZ(float);
	void setCamVang(float);
	void setCamHang(float);
	void updateMaxRayBnc(int);
	void updateAajitter(float);
	void updateGamma(float);
	void togglePrevMode(bool);
	void toggleAABBmode(bool);
	void toggleTextures(bool);

private:
	// 3D scene description
	Scene scene;

	QMutex mutex;
	bool loaded;

};

extern SceneState sceneState;

