#ifndef LOADDIALOG_H
#define LOADDIALOG_H

#include "dialoglogger.h"

#include <QDialog>
#include <QThread>

namespace Ui {
class LoadDialog;
}

class LoadDialog : public QDialog
{
    Q_OBJECT

public:
    explicit LoadDialog(QWidget *parent = 0, QThread* workert = 0);
    ~LoadDialog();
    void setWorkerThread(QThread* workert);
    DialogLogger* getLogger();

private slots:
    void progress_slot(int p);
    void msg_slot(const QString& msg);

    void on_pushButton_clicked();

signals:

private:
    Ui::LoadDialog *ui;
    DialogLogger* dlogger;
    QThread* workert;

    void closeEvent(QCloseEvent *) override;
    void showEvent(QShowEvent *) override;
    void keyPressEvent(QKeyEvent *) override;
};

#endif // LOADDIALOG_H
