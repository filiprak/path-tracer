#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QMessageBox>
#include <QString>
#include "cudarunthread.h"


namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
	void toggle_saveiters(int);

	void update_stats(int, double);

    void on_actionSettings_triggered();

    void on_actionClose_triggered();

    void on_actionAbout_triggered();

    void on_actionLoad_Scene_triggered();

    void on_fsave_btn_clicked();

	void on_resume_btn_clicked();

	void on_pause_btn_clicked();

    void on_restart_btn_clicked();

    void on_maxd_spinbox_valueChanged(int arg1);

    void on_aajitter_spinbox_valueChanged(double arg1);

    void on_gamm_spinbox_valueChanged(double arg1);

    void on_text_checkbox_stateChanged(int arg1);

    void aabb_checkbox_stateChanged(bool checked);

	void prev_checkbox_stateChanged(bool checked);

	void norm_checkbox_stateChanged(bool checked);

    void on_hang_spinbox_valueChanged(double arg1);

    void on_vang_spinbox_valueChanged(double arg1);

    void on_x_spinbox_valueChanged(double arg1);

    void on_y_spinbox_valueChanged(double arg1);

    void on_z_spinbox_valueChanged(double arg1);

	void on_right_btn_clicked();

private:
    Ui::MainWindow *ui;

	CudaRunThread* cuda_thread;

    void closeEvent(QCloseEvent *Event);
    bool askForClosing();
	void stop_rendering();
	void load_scene_from_file(QString src_path);
};

#endif // MAINWINDOW_H
