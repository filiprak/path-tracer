#include "mainwindow.h"
#include "loaddialog.h"
#include "sceneloaderthread.h"
#include "scenestate.h"
#include "ui_mainwindow.h"
#include "qtconfig.h"

#include <QCloseEvent>
#include <QFileDialog>
#include <QProgressDialog>
#include <QDateTime>


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
	connect(ui->aabb_checkbox, SIGNAL(toggled(bool)), this, SLOT(on_aabb_checkbox_stateChanged(bool)));
	connect(ui->prev_checkbox, SIGNAL(toggled(bool)), this, SLOT(on_prev_checkbox_stateChanged(bool)));
	connect(ui->norm_checkbox, SIGNAL(toggled(bool)), this, SLOT(on_norm_checkbox_stateChanged(bool)));
	ui->save_iters->setEnabled(ui->autosave_checkbox->isChecked());

	// create output images folder
	if(!QDir(AUTOSAVE_IMG_FOLDER).exists())
		QDir().mkdir(AUTOSAVE_IMG_FOLDER);
}

MainWindow::~MainWindow()
{
	delete cuda_thread;
	delete ui;
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

void MainWindow::load_scene_from_file(QString src_path)
{
	LoadDialog ldial(this);
	SceneLoaderThread slthread(src_path, ldial.getLogger());
	ldial.setWorkerThread(&slthread);
	ldial.exec();
	if (!slthread.isFinished())
		slthread.wait();
	ldial.accept();
	Scene scene = sceneState.clone();
	ui->previewGLWidget->reloadTexture(scene.camera.projection.width, scene.camera.projection.height);

	// load scene settings to gui
	ui->x_spinbox->setValue((double)scene.camera.position.x);
	ui->y_spinbox->setValue((double)scene.camera.position.y);
	ui->z_spinbox->setValue((double)scene.camera.position.z);

	ui->hang_spinbox->setValue((double)scene.camera.h_ang);
	ui->vang_spinbox->setValue((double)scene.camera.v_ang);

	ui->prev_checkbox->setChecked(scene.camera.preview_mode);
	ui->aabb_checkbox->setChecked(scene.camera.preview_mode);
	ui->text_checkbox->setChecked(scene.camera.texture_enabled);

	ui->gamm_spinbox->setValue((double)scene.camera.projection.gamma_corr);
	ui->maxd_spinbox->setValue(scene.camera.max_ray_bounces);
}

void MainWindow::update_stats(int iter, double time)
{
	ui->lcdNumber->display(iter);
	ui->iter_label->setText(QString::asprintf("Iteration(%.3fs)", (float)time));

	if (ui->autosave_checkbox->isChecked() && ((iter % (ui->save_iters->value())) == 0 || iter  == 1) && iter > 0) {
		// autosave image
		QDateTime dt = QDateTime::currentDateTime();
		QString file_name = QString(AUTOSAVE_IMG_FOLDER) + "/" + ui->fname_lineedit->text() + "_" + dt.toString("dd-MM-yyyy") +
			"_" + QString::number(iter) + "spp.png";
		(ui->previewGLWidget->grabFramebuffer()).save(file_name, "png");
	}
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
    QString sceneFilename = QFileDialog::getOpenFileName(this,
        "Open scene descripion file", "", "Json Files (*.json);;All Files (*)");
    if (!sceneFilename.isEmpty()) {
		stop_rendering();
        // load scene file
		load_scene_from_file(sceneFilename);
    }
}

void MainWindow::on_fsave_btn_clicked()
{
    QString defaultFilename = ui->fname_lineedit->text();
    QString sceneFilename = QFileDialog::getSaveFileName(this,
        "Save result image", defaultFilename, "PNG (*.png);;JPEG (*.jpg);;BMP (*.bmp);;All Files (*)");
    if (!sceneFilename.isEmpty()) {
		(ui->previewGLWidget->grabFramebuffer()).save(sceneFilename);
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
		ui->restart_btn->setText("Start");

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
	if (cuda_thread && sceneState.isLoaded()) {
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
	if (cuda_thread && sceneState.isLoaded()) {
		cuda_thread->step();
		ui->restart_btn->setText("Stop");
	}
}

void MainWindow::on_maxd_spinbox_valueChanged(int arg1)
{
	sceneState.updateMaxRayBnc(arg1);
}

void MainWindow::on_aajitter_spinbox_valueChanged(double arg1)
{
	sceneState.updateAajitter((float)arg1);
}

void MainWindow::on_gamm_spinbox_valueChanged(double arg1)
{
	sceneState.updateGamma((float)arg1);
}

void MainWindow::on_text_checkbox_stateChanged(int arg1)
{
	sceneState.toggleTextures((bool)arg1);
}

void MainWindow::on_aabb_checkbox_stateChanged(bool checked)
{
	sceneState.toggleAABBmode(checked);
}

void MainWindow::on_prev_checkbox_stateChanged(bool checked)
{
	sceneState.togglePrevMode(checked);
}

void MainWindow::on_norm_checkbox_stateChanged(bool checked)
{
	if (checked) {
		sceneState.togglePrevMode(false);
		sceneState.toggleAABBmode(false);
	}
}

void MainWindow::on_hang_spinbox_valueChanged(double arg1)
{
	sceneState.setCamHang((float)arg1);
}

void MainWindow::on_vang_spinbox_valueChanged(double arg1)
{
	sceneState.setCamVang((float)arg1);
}

void MainWindow::on_x_spinbox_valueChanged(double arg1)
{
	sceneState.setCamPositionX((float)arg1);
}

void MainWindow::on_y_spinbox_valueChanged(double arg1)
{
	sceneState.setCamPositionY((float)arg1);
}

void MainWindow::on_z_spinbox_valueChanged(double arg1)
{
	sceneState.setCamPositionZ((float)arg1);
}
