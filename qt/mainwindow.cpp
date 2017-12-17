#include "mainwindow.h"
#include "loaddialog.h"
#include "sceneloaderthread.h"
#include "ui_mainwindow.h"

#include <QCloseEvent>
#include <QFileDialog>
#include <QProgressDialog>


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    ui->settings_dockw->setWindowTitle("Settings");
}

void MainWindow::closeEvent(QCloseEvent *Event)
{
    if(!this->askForClosing())
    {
        Event->ignore();
        return;
    }
    Event->accept();
}

MainWindow::~MainWindow()
{
    delete ui;
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
        //@todo: load scene file

    }
}

void MainWindow::on_fsave_btn_clicked()
{
    QString defaultFilename = ui->fname_lineedit->text();
    QString sceneFilename = QFileDialog::getSaveFileName(this,
        "Save result image", defaultFilename, "PNG (*.png);;JPEG (*.jpg);;All Files (*)");
    if (!sceneFilename.isEmpty()) {
        //@todo: save output image
        ui->lcdNumber->display(ui->lcdNumber->value() + 1);
    }
}

void MainWindow::on_stop_btn_clicked()
{
    //@todo stop pathtracing
}

void MainWindow::on_restart_btn_clicked()
{
    //@todo reastart pathtracing
    LoadDialog ldial(this);
    SceneLoaderThread slthread(ldial.getLogger());
    ldial.setWorkerThread(&slthread);
    ldial.exec();
    if (!slthread.isFinished())
        slthread.wait();
    ldial.accept();
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
