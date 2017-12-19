#include <qapplication.h>
#include <qmessagebox.h>
#include "mainwindow.h"
#include "main.h"


int runQt5(int argc, char* argv[]) {
	QApplication app(argc, argv);

#ifndef QT_NO_OPENGL
	MainWindow mwindow;

	if (detectCUDAdevice() < 1) {
		QMessageBox msg;
		msg.setText("CUDA devices not found.\nPlease ensure you have it installed.");
		msg.exec();
		return EXIT_FAILURE;
	}

	mwindow.show();
#else
	QMessageBox msg;
	msg.setText("OpenGL support required.");
	msg.show();
#endif

	return app.exec();
}