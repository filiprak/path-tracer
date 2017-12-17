/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.9.3
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QDockWidget>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QFormLayout>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLCDNumber>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>
#include "previewglwidget.h"

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QAction *actionSettings;
    QAction *actionClose;
    QAction *actionAbout;
    QAction *actionLoad_Scene;
    QWidget *centralWidget;
    QGridLayout *gridLayout;
    PreviewGLWidget *previewGLWidget;
    QMenuBar *menuBar;
    QMenu *menuFile;
    QMenu *menuHelp;
    QStatusBar *statusBar;
    QDockWidget *settings_dockw;
    QWidget *settings_dockw_content;
    QVBoxLayout *verticalLayout;
    QGroupBox *cam_groupbox;
    QFormLayout *formLayout_2;
    QLabel *pos_label;
    QWidget *cpos_xyz;
    QHBoxLayout *horizontalLayout;
    QDoubleSpinBox *x_spinbox;
    QDoubleSpinBox *y_spinbox;
    QDoubleSpinBox *z_spinbox;
    QLabel *hang_label;
    QDoubleSpinBox *hang_spinbox;
    QLabel *vang_label;
    QDoubleSpinBox *vang_spinbox;
    QCheckBox *prev_checkbox;
    QCheckBox *aabb_checkbox;
    QCheckBox *text_checkbox;
    QGroupBox *image_groupbox;
    QFormLayout *formLayout_3;
    QLabel *gamm_label;
    QDoubleSpinBox *gamm_spinbox;
    QLabel *fname_label;
    QLineEdit *fname_lineedit;
    QPushButton *fsave_btn;
    QGroupBox *patht_groupbox;
    QFormLayout *formLayout;
    QPushButton *stop_btn;
    QPushButton *restart_btn;
    QLabel *maxd_label;
    QSpinBox *maxd_spinbox;
    QLabel *aajitter_label;
    QDoubleSpinBox *aajitter_spinbox;
    QLabel *iter_label;
    QLCDNumber *lcdNumber;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(927, 593);
        MainWindow->setLayoutDirection(Qt::LeftToRight);
        actionSettings = new QAction(MainWindow);
        actionSettings->setObjectName(QStringLiteral("actionSettings"));
        actionClose = new QAction(MainWindow);
        actionClose->setObjectName(QStringLiteral("actionClose"));
        actionAbout = new QAction(MainWindow);
        actionAbout->setObjectName(QStringLiteral("actionAbout"));
        actionLoad_Scene = new QAction(MainWindow);
        actionLoad_Scene->setObjectName(QStringLiteral("actionLoad_Scene"));
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        gridLayout = new QGridLayout(centralWidget);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        previewGLWidget = new PreviewGLWidget(centralWidget);
        previewGLWidget->setObjectName(QStringLiteral("previewGLWidget"));
        QPalette palette;
        QBrush brush(QColor(0, 0, 0, 255));
        brush.setStyle(Qt::SolidPattern);
        palette.setBrush(QPalette::Active, QPalette::WindowText, brush);
        palette.setBrush(QPalette::Inactive, QPalette::WindowText, brush);
        QBrush brush1(QColor(255, 255, 255, 127));
        brush1.setStyle(Qt::SolidPattern);
        palette.setBrush(QPalette::Disabled, QPalette::WindowText, brush1);
        previewGLWidget->setPalette(palette);
        previewGLWidget->setLayoutDirection(Qt::RightToLeft);

        gridLayout->addWidget(previewGLWidget, 1, 0, 1, 1);

        MainWindow->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(MainWindow);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 927, 20));
        menuFile = new QMenu(menuBar);
        menuFile->setObjectName(QStringLiteral("menuFile"));
        menuHelp = new QMenu(menuBar);
        menuHelp->setObjectName(QStringLiteral("menuHelp"));
        MainWindow->setMenuBar(menuBar);
        statusBar = new QStatusBar(MainWindow);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        MainWindow->setStatusBar(statusBar);
        settings_dockw = new QDockWidget(MainWindow);
        settings_dockw->setObjectName(QStringLiteral("settings_dockw"));
        settings_dockw_content = new QWidget();
        settings_dockw_content->setObjectName(QStringLiteral("settings_dockw_content"));
        verticalLayout = new QVBoxLayout(settings_dockw_content);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        cam_groupbox = new QGroupBox(settings_dockw_content);
        cam_groupbox->setObjectName(QStringLiteral("cam_groupbox"));
        formLayout_2 = new QFormLayout(cam_groupbox);
        formLayout_2->setSpacing(6);
        formLayout_2->setContentsMargins(11, 11, 11, 11);
        formLayout_2->setObjectName(QStringLiteral("formLayout_2"));
        pos_label = new QLabel(cam_groupbox);
        pos_label->setObjectName(QStringLiteral("pos_label"));

        formLayout_2->setWidget(1, QFormLayout::LabelRole, pos_label);

        cpos_xyz = new QWidget(cam_groupbox);
        cpos_xyz->setObjectName(QStringLiteral("cpos_xyz"));
        horizontalLayout = new QHBoxLayout(cpos_xyz);
        horizontalLayout->setSpacing(3);
        horizontalLayout->setContentsMargins(11, 11, 11, 11);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        horizontalLayout->setContentsMargins(0, 0, 0, 0);
        x_spinbox = new QDoubleSpinBox(cpos_xyz);
        x_spinbox->setObjectName(QStringLiteral("x_spinbox"));

        horizontalLayout->addWidget(x_spinbox);

        y_spinbox = new QDoubleSpinBox(cpos_xyz);
        y_spinbox->setObjectName(QStringLiteral("y_spinbox"));

        horizontalLayout->addWidget(y_spinbox);

        z_spinbox = new QDoubleSpinBox(cpos_xyz);
        z_spinbox->setObjectName(QStringLiteral("z_spinbox"));

        horizontalLayout->addWidget(z_spinbox);


        formLayout_2->setWidget(1, QFormLayout::FieldRole, cpos_xyz);

        hang_label = new QLabel(cam_groupbox);
        hang_label->setObjectName(QStringLiteral("hang_label"));

        formLayout_2->setWidget(3, QFormLayout::LabelRole, hang_label);

        hang_spinbox = new QDoubleSpinBox(cam_groupbox);
        hang_spinbox->setObjectName(QStringLiteral("hang_spinbox"));

        formLayout_2->setWidget(3, QFormLayout::FieldRole, hang_spinbox);

        vang_label = new QLabel(cam_groupbox);
        vang_label->setObjectName(QStringLiteral("vang_label"));

        formLayout_2->setWidget(4, QFormLayout::LabelRole, vang_label);

        vang_spinbox = new QDoubleSpinBox(cam_groupbox);
        vang_spinbox->setObjectName(QStringLiteral("vang_spinbox"));

        formLayout_2->setWidget(4, QFormLayout::FieldRole, vang_spinbox);

        prev_checkbox = new QCheckBox(cam_groupbox);
        prev_checkbox->setObjectName(QStringLiteral("prev_checkbox"));

        formLayout_2->setWidget(5, QFormLayout::FieldRole, prev_checkbox);

        aabb_checkbox = new QCheckBox(cam_groupbox);
        aabb_checkbox->setObjectName(QStringLiteral("aabb_checkbox"));

        formLayout_2->setWidget(6, QFormLayout::FieldRole, aabb_checkbox);

        text_checkbox = new QCheckBox(cam_groupbox);
        text_checkbox->setObjectName(QStringLiteral("text_checkbox"));
        text_checkbox->setChecked(true);

        formLayout_2->setWidget(7, QFormLayout::FieldRole, text_checkbox);


        verticalLayout->addWidget(cam_groupbox);

        image_groupbox = new QGroupBox(settings_dockw_content);
        image_groupbox->setObjectName(QStringLiteral("image_groupbox"));
        formLayout_3 = new QFormLayout(image_groupbox);
        formLayout_3->setSpacing(6);
        formLayout_3->setContentsMargins(11, 11, 11, 11);
        formLayout_3->setObjectName(QStringLiteral("formLayout_3"));
        gamm_label = new QLabel(image_groupbox);
        gamm_label->setObjectName(QStringLiteral("gamm_label"));

        formLayout_3->setWidget(0, QFormLayout::LabelRole, gamm_label);

        gamm_spinbox = new QDoubleSpinBox(image_groupbox);
        gamm_spinbox->setObjectName(QStringLiteral("gamm_spinbox"));
        gamm_spinbox->setValue(1.5);

        formLayout_3->setWidget(0, QFormLayout::FieldRole, gamm_spinbox);

        fname_label = new QLabel(image_groupbox);
        fname_label->setObjectName(QStringLiteral("fname_label"));

        formLayout_3->setWidget(2, QFormLayout::LabelRole, fname_label);

        fname_lineedit = new QLineEdit(image_groupbox);
        fname_lineedit->setObjectName(QStringLiteral("fname_lineedit"));

        formLayout_3->setWidget(3, QFormLayout::SpanningRole, fname_lineedit);

        fsave_btn = new QPushButton(image_groupbox);
        fsave_btn->setObjectName(QStringLiteral("fsave_btn"));

        formLayout_3->setWidget(4, QFormLayout::LabelRole, fsave_btn);


        verticalLayout->addWidget(image_groupbox);

        patht_groupbox = new QGroupBox(settings_dockw_content);
        patht_groupbox->setObjectName(QStringLiteral("patht_groupbox"));
        formLayout = new QFormLayout(patht_groupbox);
        formLayout->setSpacing(6);
        formLayout->setContentsMargins(11, 11, 11, 11);
        formLayout->setObjectName(QStringLiteral("formLayout"));
        stop_btn = new QPushButton(patht_groupbox);
        stop_btn->setObjectName(QStringLiteral("stop_btn"));

        formLayout->setWidget(1, QFormLayout::LabelRole, stop_btn);

        restart_btn = new QPushButton(patht_groupbox);
        restart_btn->setObjectName(QStringLiteral("restart_btn"));

        formLayout->setWidget(1, QFormLayout::FieldRole, restart_btn);

        maxd_label = new QLabel(patht_groupbox);
        maxd_label->setObjectName(QStringLiteral("maxd_label"));

        formLayout->setWidget(2, QFormLayout::LabelRole, maxd_label);

        maxd_spinbox = new QSpinBox(patht_groupbox);
        maxd_spinbox->setObjectName(QStringLiteral("maxd_spinbox"));
        maxd_spinbox->setValue(5);
        maxd_spinbox->setDisplayIntegerBase(10);

        formLayout->setWidget(2, QFormLayout::FieldRole, maxd_spinbox);

        aajitter_label = new QLabel(patht_groupbox);
        aajitter_label->setObjectName(QStringLiteral("aajitter_label"));

        formLayout->setWidget(3, QFormLayout::LabelRole, aajitter_label);

        aajitter_spinbox = new QDoubleSpinBox(patht_groupbox);
        aajitter_spinbox->setObjectName(QStringLiteral("aajitter_spinbox"));
        aajitter_spinbox->setValue(1.5);

        formLayout->setWidget(3, QFormLayout::FieldRole, aajitter_spinbox);

        iter_label = new QLabel(patht_groupbox);
        iter_label->setObjectName(QStringLiteral("iter_label"));

        formLayout->setWidget(0, QFormLayout::LabelRole, iter_label);

        lcdNumber = new QLCDNumber(patht_groupbox);
        lcdNumber->setObjectName(QStringLiteral("lcdNumber"));
        lcdNumber->setFrameShape(QFrame::NoFrame);
        lcdNumber->setFrameShadow(QFrame::Raised);
        lcdNumber->setLineWidth(0);
        lcdNumber->setMode(QLCDNumber::Dec);
        lcdNumber->setSegmentStyle(QLCDNumber::Flat);
        lcdNumber->setProperty("intValue", QVariant(30223));

        formLayout->setWidget(0, QFormLayout::FieldRole, lcdNumber);


        verticalLayout->addWidget(patht_groupbox);

        settings_dockw->setWidget(settings_dockw_content);
        MainWindow->addDockWidget(static_cast<Qt::DockWidgetArea>(1), settings_dockw);

        menuBar->addAction(menuFile->menuAction());
        menuBar->addAction(menuHelp->menuAction());
        menuFile->addAction(actionSettings);
        menuFile->addAction(actionLoad_Scene);
        menuFile->addAction(actionClose);
        menuFile->addSeparator();
        menuHelp->addAction(actionAbout);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "Path Tracing Preview", Q_NULLPTR));
        actionSettings->setText(QApplication::translate("MainWindow", "Settings", Q_NULLPTR));
        actionClose->setText(QApplication::translate("MainWindow", "Close", Q_NULLPTR));
        actionAbout->setText(QApplication::translate("MainWindow", "About", Q_NULLPTR));
        actionLoad_Scene->setText(QApplication::translate("MainWindow", "Load Scene", Q_NULLPTR));
        menuFile->setTitle(QApplication::translate("MainWindow", "File", Q_NULLPTR));
        menuHelp->setTitle(QApplication::translate("MainWindow", "Help", Q_NULLPTR));
        cam_groupbox->setTitle(QApplication::translate("MainWindow", "Camera", Q_NULLPTR));
        pos_label->setText(QApplication::translate("MainWindow", "Position", Q_NULLPTR));
        hang_label->setText(QApplication::translate("MainWindow", "HAngle", Q_NULLPTR));
        vang_label->setText(QApplication::translate("MainWindow", "VAngle", Q_NULLPTR));
        prev_checkbox->setText(QApplication::translate("MainWindow", "Simple Preview Mode", Q_NULLPTR));
        aabb_checkbox->setText(QApplication::translate("MainWindow", "AABBoxes Preview mode", Q_NULLPTR));
        text_checkbox->setText(QApplication::translate("MainWindow", "Render textures", Q_NULLPTR));
        image_groupbox->setTitle(QApplication::translate("MainWindow", "Image", Q_NULLPTR));
        gamm_label->setText(QApplication::translate("MainWindow", "Gamma Correction Exp", Q_NULLPTR));
        fname_label->setText(QApplication::translate("MainWindow", "File name", Q_NULLPTR));
        fname_lineedit->setText(QApplication::translate("MainWindow", "image.png", Q_NULLPTR));
        fsave_btn->setText(QApplication::translate("MainWindow", "Save Image", Q_NULLPTR));
        patht_groupbox->setTitle(QApplication::translate("MainWindow", "Path Tracing", Q_NULLPTR));
        stop_btn->setText(QApplication::translate("MainWindow", "Stop", Q_NULLPTR));
        restart_btn->setText(QApplication::translate("MainWindow", "Restart", Q_NULLPTR));
        maxd_label->setText(QApplication::translate("MainWindow", "Max depth", Q_NULLPTR));
        aajitter_label->setText(QApplication::translate("MainWindow", "Antialiasing ray jitter", Q_NULLPTR));
        iter_label->setText(QApplication::translate("MainWindow", "Iteration(<time> ms)", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
