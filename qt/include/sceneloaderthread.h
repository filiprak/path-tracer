#ifndef SCENELOADERTHREAD_H
#define SCENELOADERTHREAD_H

#include "loaddialog.h"
#include <QThread>

class SceneLoaderThread : public QThread
{
	Q_OBJECT
public:
    SceneLoaderThread(QString src_file, DialogLogger* logger);

private:
    DialogLogger* logger;
	QString src_file;

	void run() override;
};

#endif // SCENELOADERTHREAD_H
