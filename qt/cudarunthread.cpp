#include "cudarunthread.h"
#include "kernel.h"
#include "scenestate.h"
#include "cudaUtility.h"
#include <time.h>


CudaRunThread::CudaRunThread()
	: pause_requested(false), cancel_requested(false)
{
	
}

void CudaRunThread::run() {
	// init kernel settings
	kernelInit(sceneState.clone());

	// run path tracing iterations on CUDA
	int i = 1;
	while (i < 40001)
	{
		mutex.lock();
		
		Scene scene = sceneState.clone();
		sceneState.cameraChanged(false, &i);

		// run CUDA kernels
		clock_t begin = clock();
		runCUDA(pbo_dptr, scene, i);
		// blocked until kernels are finished
		clock_t end = clock();

		double time_perf = (double)(end - begin) / CLOCKS_PER_SEC;

		emit finishedIteration(i, time_perf);
		//printf("runCUDA(iteration: %d) time performance: %.3f s\n", i, time_perf);

		if (checkCudaError("runCUDA()")) {
			mutex.unlock();
			break;
		}

		if (pause_requested)
			paused.wait(&mutex);

		if (cancel_requested) {
			mutex.unlock();
			break;
		}
		mutex.unlock();

		++i;
	}

	kernelCleanUp();
}

void CudaRunThread::cancel() {
	cancel_requested = true;
	if (pause_requested)
		resume();
}

void CudaRunThread::resetFlags() {
	pause_requested = false;
	cancel_requested = false;
}

void CudaRunThread::pause() {
	pause_requested = true;
}

void CudaRunThread::resume() {
	pause_requested = false;
	paused.wakeAll();
}

void CudaRunThread::step() {
	pause_requested = true;
	if (!isRunning())
		start();
	else
		paused.wakeAll();
}