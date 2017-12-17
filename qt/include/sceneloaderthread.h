#ifndef SCENELOADERTHREAD_H
#define SCENELOADERTHREAD_H

#include "loaddialog.h"
#include <QThread>

class SceneLoaderThread : public QThread
{
    Q_OBJECT
    void run() override {
        for (int i = 0; i < 100; ++i) {
            logger->info("Progress: %d", i);
            logger->progress(i);
            msleep(20);
        }
        logger->error("Finished!");
        logger->progress(100);
    }
public:
    SceneLoaderThread(DialogLogger* logger);

private:
    DialogLogger* logger;
};

#endif // SCENELOADERTHREAD_H
