#include "mainwindow.h"
#include "loaddialog.h"
#include "sceneloaderthread.h"
#include "scenestate.h"
#include "ui_mainwindow.h"

#include <QCloseEvent>
#include <QFileDialog>
#include <QProgressDialog>


MainWindow::MainWindow(QWidget *parent) :
	QMainWindow(parent), cuda_thread(NULL), ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    ui->settings_dockw->setWindowTitle("Settings");

	cuda_thread = new CudaRunThread();
	connect(cuda_thread, SIGNAL(finishedIteration(int, double)), ui->previewGLWidget, SLOT(refresh(int, double)),
		Qt::QueuedConnection);
	connect(cuda_thread, SIGNAL(finishedIteration(int, double)), this, SLOT(update_stats(int, double)),
		Qt::QueuedConnection);
	connect(ui->autosave_checkbox, SIGNAL(stateChanged(int)), this, SLOT(toggle_saveiters(int)));
	ui->save_iters->setEnabled(ui->autosave_checkbox->isChecked());
}

void MainWindow::toggle_saveiters(int state)
{
	ui->save_iters->setEnabled((bool)state);
}

void MainWindow::closeEvent(QCloseEvent *Event)
{
    if(!this->askForClosing())
    {
        Event->ignore();
        return;
    }
	// stop rendering if was in progress
	stop_rendering();
    Event->accept();
}

void MainWindow::showEvent(QShowEvent *Event)
{
	// default scene load
	LoadDialog ldial(this);
	SceneLoaderThread slthread("..\\..\\scenes\\scene_teapot.json", ldial.getLogger());
	ldial.setWorkerThread(&slthread);
	ldial.exec();
	if (!slthread.isFinished())
		slthread.wait();
	ldial.accept();
	Camera cam = sceneState.clone().camera;
	ui->previewGLWidget->reloadTexture(cam.projection.width, cam.projection.height);

	Event->accept();
}

MainWindow::~MainWindow()
{
	delete cuda_thread;
    delete ui;
}

void MainWindow::update_stats(int iter, double time)
{
	ui->lcdNumber->display(iter);
	ui->iter_label->setText(QString::asprintf("Iteration(%.3fs)", (float)time));
}

void MainWindow::on_actionSettings_triggered()
{
    ui->settings_dockw->show();
}

void MainWindow::on_actionClose_triggered()
{
    this->close();
}

bool MainWindow::askForClosing()
{
    QMessageBox msgBox(this);
    msgBox.setWindowTitle("Closing");
    msgBox.setText("Do you want to exit ?");
    msgBox.setStandardButtons(QMessageBox::Yes | QMessageBox::No);
    msgBox.setDefaultButton(QMessageBox::No);
    if (msgBox.exec() == QMessageBox::Yes) {
        return true;
    };
    return false;
}

void MainWindow::on_actionAbout_triggered()
{
    QMessageBox aboutBox(this);
    aboutBox.setWindowTitle("About");
    aboutBox.setIcon(QMessageBox::Icon::Information);
    aboutBox.setText("Path tracing GPU CUDA renderer\nby Filip Rak");
    aboutBox.setStandardButtons(QMessageBox::Close);
    aboutBox.setDefaultButton(QMessageBox::Close);
    aboutBox.exec();
}

void MainWindow::on_actionLoad_Scene_triggered()
{
	
	/*if (cuda_thread && cuda_thread->isRunning()) {
		QMessageBox aboutBox(this);
		aboutBox.setWindowTitle("Scene loading");
		aboutBox.setIcon(QMessageBox::Icon::Critical);
		aboutBox.setText("Rendering in progress.\nPlease stop rendering to load new scene.");
		aboutBox.setStandardButtons(QMessageBox::Close);
		aboutBox.setDefaultButton(QMessageBox::Close);
		aboutBox.exec();
		return;
	}*/

    QString sceneFilename = QFileDialog::getOpenFileName(this,
        "Open scene descripion file", "", "Json Files (*.json);;All Files (*)");
    if (!sceneFilename.isEmpty()) {
		stop_rendering();
        // load scene file
		LoadDialog ldial(this);
		SceneLoaderThread slthread(sceneFilename, ldial.getLogger());
		ldial.setWorkerThread(&slthread);
		ldial.exec();
		if (!slthread.isFinished())
			slthread.wait();
		ldial.accept();
		Camera cam = sceneState.clone().camera;
		ui->previewGLWidget->reloadTexture(cam.projection.width, cam.projection.height);
    }
}

void MainWindow::on_fsave_btn_clicked()
{
    QString defaultFilename = ui->fname_lineedit->text();
    QString sceneFilename = QFileDialog::getSaveFileName(this,
        "Save result image", defaultFilename, "PNG (*.png);;JPEG (*.jpg);;All Files (*)");
    if (!sceneFilename.isEmpty()) {
        //@todo: save output image
        
    }
}

void MainWindow::stop_rendering()
{
	if (cuda_thread) {
		QMessageBox msg(this);
		msg.setWindowTitle("Stopping render process");
		msg.setText("Terminating render thread. Please wait...");
		msg.setStandardButtons(QMessageBox::Close);
		msg.setDefaultButton(QMessageBox::Close);
		msg.show();
		QApplication::processEvents();

		cuda_thread->cancel();
		while (cuda_thread->isRunning())
			cuda_thread->wait();
		cuda_thread->resetFlags();
		emit cuda_thread->finishedIteration(0, 0.0);

		msg.close();
	}

}

void MainWindow::on_resume_btn_clicked()
{
	if (cuda_thread)
		cuda_thread->resume();
}

void MainWindow::on_pause_btn_clicked()
{
	if (cuda_thread)
		cuda_thread->pause();
}

void MainWindow::on_restart_btn_clicked()
{
	if (cuda_thread) {
		if (cuda_thread->isRunning()) {
			stop_rendering();
			ui->restart_btn->setText("Start");
		}
		else {
			cuda_thread->start();
			ui->restart_btn->setText("Stop");
		}
	}
}

void MainWindow::on_right_btn_clicked()
{
	if (cuda_thread) {
		cuda_thread->step();
		ui->restart_btn->setText("Stop");
	}
}

void MainWindow::on_maxd_spinbox_valueChanged(int arg1)
{
    //@todo change camera max depth val
}

void MainWindow::on_aajitter_spinbox_valueChanged(double arg1)
{
    //@todo change camera aa jitter
}

void MainWindow::on_gamm_spinbox_valueChanged(double arg1)
{
    //@todo change image gamma correction
}

void MainWindow::on_text_checkbox_stateChanged(int arg1)
{
    //@todo toggle textures
}

void MainWindow::on_aabb_checkbox_stateChanged(int arg1)
{
    //@todo aabb boxes preview mode
}

void MainWindow::on_prev_checkbox_stateChanged(int arg1)
{
    //@todo toggle preview mode
}

void MainWindow::on_hang_spinbox_valueChanged(double arg1)
{

}

void MainWindow::on_vang_spinbox_valueChanged(double arg1)
{

}

void MainWindow::on_x_spinbox_valueChanged(double arg1)
{

}

void MainWindow::on_y_spinbox_valueChanged(double arg1)
{

}

void MainWindow::on_z_spinbox_valueChanged(double arg1)
{

}
