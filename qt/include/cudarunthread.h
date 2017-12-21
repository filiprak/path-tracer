#ifndef CUDARUNTHREAD_H
#define CUDARUNTHREAD_H

#include "previewglwidget.h"

#include <QThread>
#include <QMutex>
#include <QWaitCondition>


class CudaRunThread : public QThread
{
	Q_OBJECT
public:
	CudaRunThread();

	void cancel();
	void resetFlags();
	void pause();
	void resume();
	void step();

private:
	bool pause_requested;
	bool cancel_requested;

	QMutex mutex;
	QWaitCondition paused;

	void run() override;

private slots:

signals :
	void finishedIteration(int, double);
};

#endif // CUDARUNTHREAD_H
