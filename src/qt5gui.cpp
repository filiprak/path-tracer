#include <qapplication.h>
#include "mainwindow.h"


int runQt5(int argc, char* argv[]) {
	QApplication app(argc, argv);
	MainWindow mwindow;
	mwindow.show();

	return app.exec();
}