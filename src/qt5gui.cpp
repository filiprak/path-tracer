#include <qapplication.h>
#include <qmessagebox.h>
#include "mainwindow.h"


int runQt5(int argc, char* argv[]) {
	QApplication app(argc, argv);

	app.setApplicationVersion("0.1");
#ifndef QT_NO_OPENGL
	MainWindow mwindow;
	mwindow.show();
#else
	QMessageBox msg;
	msg.setText("OpenGL support required.");
	msg.show();
#endif

	return app.exec();
}