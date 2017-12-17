#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QMessageBox>

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
    void on_actionSettings_triggered();

    void on_actionClose_triggered();

    void on_actionAbout_triggered();

    void on_actionLoad_Scene_triggered();

    void on_fsave_btn_clicked();

    void on_stop_btn_clicked();

    void on_restart_btn_clicked();

    void on_maxd_spinbox_valueChanged(int arg1);

    void on_aajitter_spinbox_valueChanged(double arg1);

    void on_gamm_spinbox_valueChanged(double arg1);

    void on_text_checkbox_stateChanged(int arg1);

    void on_aabb_checkbox_stateChanged(int arg1);

    void on_prev_checkbox_stateChanged(int arg1);

    void on_hang_spinbox_valueChanged(double arg1);

    void on_vang_spinbox_valueChanged(double arg1);

    void on_x_spinbox_valueChanged(double arg1);

    void on_y_spinbox_valueChanged(double arg1);

    void on_z_spinbox_valueChanged(double arg1);

private:
    Ui::MainWindow *ui;
    void closeEvent(QCloseEvent *Event);
    bool askForClosing();
};

#endif // MAINWINDOW_H
