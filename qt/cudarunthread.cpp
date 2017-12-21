#include "cudarunthread.h"
#include "kernel.h"
#include "scenestate.h"
#include "cudaUtility.h"
#include <time.h>
#include <thread>


CudaRunThread::CudaRunThread()
	: pause_requested(false), cancel_requested(false)
{
	
}

void CudaRunThread::run() {
	// init kernel settings
	kernelInit(sceneState.clone());

	// run path tracing iterations on CUDA
	for (int i = 0; i < 5000; i++)
	{
		mutex.lock();
		clock_t begin = clock();

		// run CUDA kernels
		runCUDA(pbo_dptr, sceneState.clone(), i);
		// blocked until kernels are finished

		clock_t end = clock();
		double time_perf = (double)(end - begin) / CLOCKS_PER_SEC;

		emit finishedIteration(i + 1, time_perf);
		printf("runCUDA(iteration: %d) time performance: %.3f s\n", i, time_perf);

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