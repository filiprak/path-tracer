#include "loaddialog.h"
#include "ui_loaddialog.h"

#include <QCloseEvent>

LoadDialog::LoadDialog(QWidget *parent, QThread* workert) :
    QDialog(parent),
    ui(new Ui::LoadDialog),
    dlogger(new DialogLogger()),
    workert(workert)
{
    ui->setupUi(this);
    connect(dlogger, SIGNAL(info_signal(const QString&)),
            this, SLOT(msg_slot(const QString&)),
            Qt::QueuedConnection);
    connect(dlogger, SIGNAL(progress_signal(int)),
            this, SLOT(progress_slot(int)),
            Qt::QueuedConnection);
}

LoadDialog::~LoadDialog()
{
    delete ui;
}

void LoadDialog::setWorkerThread(QThread *workert)
{
    this->workert = workert;
}

DialogLogger *LoadDialog::getLogger()
{
    return this->dlogger;
}

void LoadDialog::msg_slot(const QString& msg)
{
    ui->textBrowser->append(msg);
}

void LoadDialog::closeEvent(QCloseEvent * Event)
{
    if(workert && !this->workert->isFinished())
    {
        Event->ignore();
        return;
    }
    Event->accept();
}

void LoadDialog::showEvent(QShowEvent * Event)
{
    if (workert)
        workert->start();
}

void LoadDialog::keyPressEvent(QKeyEvent * Event)
{
    if(workert && !this->workert->isFinished())
    {
        Event->ignore();
        return;
    }
    Event->accept();
}

void LoadDialog::progress_slot(int p)
{
    ui->progressBar->setValue(p);
}

void LoadDialog::on_pushButton_clicked()
{
    this->close();
}
