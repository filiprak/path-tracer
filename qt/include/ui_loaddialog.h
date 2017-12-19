/********************************************************************************
** Form generated from reading UI file 'loaddialog.ui'
**
** Created by: Qt User Interface Compiler version 5.9.3
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_LOADDIALOG_H
#define UI_LOADDIALOG_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QDialog>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QProgressBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QTextBrowser>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_LoadDialog
{
public:
    QVBoxLayout *verticalLayout;
    QProgressBar *progressBar;
    QTextBrowser *textBrowser;
    QWidget *widget;
    QHBoxLayout *horizontalLayout;
    QSpacerItem *horizontalSpacer;
    QPushButton *pushButton;

    void setupUi(QDialog *LoadDialog)
    {
        if (LoadDialog->objectName().isEmpty())
            LoadDialog->setObjectName(QStringLiteral("LoadDialog"));
        LoadDialog->setWindowModality(Qt::WindowModal);
        LoadDialog->resize(467, 299);
        verticalLayout = new QVBoxLayout(LoadDialog);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        progressBar = new QProgressBar(LoadDialog);
        progressBar->setObjectName(QStringLiteral("progressBar"));
        progressBar->setValue(0);

        verticalLayout->addWidget(progressBar);

        textBrowser = new QTextBrowser(LoadDialog);
        textBrowser->setObjectName(QStringLiteral("textBrowser"));

        verticalLayout->addWidget(textBrowser);

        widget = new QWidget(LoadDialog);
        widget->setObjectName(QStringLiteral("widget"));
        horizontalLayout = new QHBoxLayout(widget);
        horizontalLayout->setSpacing(0);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        horizontalLayout->setContentsMargins(0, 0, 0, 0);
        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer);

        pushButton = new QPushButton(widget);
        pushButton->setObjectName(QStringLiteral("pushButton"));

        horizontalLayout->addWidget(pushButton);


        verticalLayout->addWidget(widget);


        retranslateUi(LoadDialog);

        QMetaObject::connectSlotsByName(LoadDialog);
    } // setupUi

    void retranslateUi(QDialog *LoadDialog)
    {
        LoadDialog->setWindowTitle(QApplication::translate("LoadDialog", "Loading info", Q_NULLPTR));
        pushButton->setText(QApplication::translate("LoadDialog", "OK", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class LoadDialog: public Ui_LoadDialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_LOADDIALOG_H
