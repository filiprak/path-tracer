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
    QLabel *hang_label;
    QDoubleSpinBox *hang_spinbox;
    QLabel *vang_label;
    QDoubleSpinBox *vang_spinbox;
    QCheckBox *aabb_checkbox;
    QCheckBox *text_checkbox;
    QWidget *cpos_xyz;
    QHBoxLayout *horizontalLayout;
    QDoubleSpinBox *x_spinbox;
    QDoubleSpinBox *z_spinbox;
    QDoubleSpinBox *y_spinbox;
    QLabel *aajitter_label;
    QCheckBox *prev_checkbox;
    QDoubleSpinBox *aajitter_spinbox;
    QGroupBox *image_groupbox;
    QFormLayout *formLayout_3;
    QLabel *gamm_label;
    QDoubleSpinBox *gamm_spinbox;
    QLabel *fname_label;
    QLineEdit *fname_lineedit;
    QCheckBox *autosave_checkbox;
    QWidget *widget_3;
    QHBoxLayout *horizontalLayout_4;
    QLabel *every_label;
    QSpinBox *save_iters;
    QPushButton *fsave_btn;
    QGroupBox *patht_groupbox;
    QFormLayout *formLayout;
    QLabel *iter_label;
    QLCDNumber *lcdNumber;
    QLabel *maxd_label;
    QSpinBox *maxd_spinbox;
    QWidget *widget;
    QHBoxLayout *horizontalLayout_2;
    QPushButton *resume_btn;
    QPushButton *pause_btn;
    QPushButton *restart_btn;
    QWidget *widget_2;
    QHBoxLayout *horizontalLayout_3;
    QPushButton *right_btn;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(1078, 631);
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
        previewGLWidget->setCursor(QCursor(Qt::CrossCursor));
        previewGLWidget->setLayoutDirection(Qt::RightToLeft);

        gridLayout->addWidget(previewGLWidget, 1, 0, 1, 1);

        MainWindow->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(MainWindow);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 1078, 20));
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

        hang_label = new QLabel(cam_groupbox);
        hang_label->setObjectName(QStringLiteral("hang_label"));

        formLayout_2->setWidget(3, QFormLayout::LabelRole, hang_label);

        hang_spinbox = new QDoubleSpinBox(cam_groupbox);
        hang_spinbox->setObjectName(QStringLiteral("hang_spinbox"));
        hang_spinbox->setCursor(QCursor(Qt::PointingHandCursor));
        hang_spinbox->setMinimum(-360);
        hang_spinbox->setMaximum(360);
        hang_spinbox->setSingleStep(5);

        formLayout_2->setWidget(3, QFormLayout::FieldRole, hang_spinbox);

        vang_label = new QLabel(cam_groupbox);
        vang_label->setObjectName(QStringLiteral("vang_label"));

        formLayout_2->setWidget(4, QFormLayout::LabelRole, vang_label);

        vang_spinbox = new QDoubleSpinBox(cam_groupbox);
        vang_spinbox->setObjectName(QStringLiteral("vang_spinbox"));
        vang_spinbox->setCursor(QCursor(Qt::PointingHandCursor));
        vang_spinbox->setDecimals(1);
        vang_spinbox->setMinimum(-360);
        vang_spinbox->setMaximum(360);
        vang_spinbox->setSingleStep(5);
        vang_spinbox->setValue(0);

        formLayout_2->setWidget(4, QFormLayout::FieldRole, vang_spinbox);

        aabb_checkbox = new QCheckBox(cam_groupbox);
        aabb_checkbox->setObjectName(QStringLiteral("aabb_checkbox"));
        aabb_checkbox->setCursor(QCursor(Qt::PointingHandCursor));

        formLayout_2->setWidget(7, QFormLayout::FieldRole, aabb_checkbox);

        text_checkbox = new QCheckBox(cam_groupbox);
        text_checkbox->setObjectName(QStringLiteral("text_checkbox"));
        text_checkbox->setCursor(QCursor(Qt::PointingHandCursor));
        text_checkbox->setChecked(true);

        formLayout_2->setWidget(8, QFormLayout::FieldRole, text_checkbox);

        cpos_xyz = new QWidget(cam_groupbox);
        cpos_xyz->setObjectName(QStringLiteral("cpos_xyz"));
        horizontalLayout = new QHBoxLayout(cpos_xyz);
        horizontalLayout->setSpacing(3);
        horizontalLayout->setContentsMargins(11, 11, 11, 11);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        horizontalLayout->setContentsMargins(0, 0, 0, 0);
        x_spinbox = new QDoubleSpinBox(cpos_xyz);
        x_spinbox->setObjectName(QStringLiteral("x_spinbox"));
        x_spinbox->setCursor(QCursor(Qt::PointingHandCursor));
        x_spinbox->setInputMethodHints(Qt::ImhNone);
        x_spinbox->setReadOnly(false);
        x_spinbox->setButtonSymbols(QAbstractSpinBox::UpDownArrows);
        x_spinbox->setAccelerated(false);
        x_spinbox->setProperty("showGroupSeparator", QVariant(false));
        x_spinbox->setDecimals(2);
        x_spinbox->setMinimum(-999.99);
        x_spinbox->setMaximum(999.99);
        x_spinbox->setValue(0);

        horizontalLayout->addWidget(x_spinbox);

        z_spinbox = new QDoubleSpinBox(cpos_xyz);
        z_spinbox->setObjectName(QStringLiteral("z_spinbox"));
        z_spinbox->setCursor(QCursor(Qt::PointingHandCursor));
        z_spinbox->setMinimum(-999.99);
        z_spinbox->setMaximum(999.99);
        z_spinbox->setValue(0);

        horizontalLayout->addWidget(z_spinbox);

        y_spinbox = new QDoubleSpinBox(cpos_xyz);
        y_spinbox->setObjectName(QStringLiteral("y_spinbox"));
        y_spinbox->setCursor(QCursor(Qt::PointingHandCursor));
        y_spinbox->setMinimum(-999.99);
        y_spinbox->setMaximum(999.99);

        horizontalLayout->addWidget(y_spinbox);


        formLayout_2->setWidget(1, QFormLayout::FieldRole, cpos_xyz);

        aajitter_label = new QLabel(cam_groupbox);
        aajitter_label->setObjectName(QStringLiteral("aajitter_label"));

        formLayout_2->setWidget(5, QFormLayout::LabelRole, aajitter_label);

        prev_checkbox = new QCheckBox(cam_groupbox);
        prev_checkbox->setObjectName(QStringLiteral("prev_checkbox"));
        prev_checkbox->setCursor(QCursor(Qt::PointingHandCursor));

        formLayout_2->setWidget(6, QFormLayout::FieldRole, prev_checkbox);

        aajitter_spinbox = new QDoubleSpinBox(cam_groupbox);
        aajitter_spinbox->setObjectName(QStringLiteral("aajitter_spinbox"));
        aajitter_spinbox->setCursor(QCursor(Qt::PointingHandCursor));
        aajitter_spinbox->setSingleStep(0.1);
        aajitter_spinbox->setValue(1.5);

        formLayout_2->setWidget(5, QFormLayout::FieldRole, aajitter_spinbox);


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
        gamm_spinbox->setCursor(QCursor(Qt::PointingHandCursor));
        gamm_spinbox->setDecimals(4);
        gamm_spinbox->setMinimum(0.1);
        gamm_spinbox->setSingleStep(0.1);
        gamm_spinbox->setValue(1.5);

        formLayout_3->setWidget(0, QFormLayout::FieldRole, gamm_spinbox);

        fname_label = new QLabel(image_groupbox);
        fname_label->setObjectName(QStringLiteral("fname_label"));

        formLayout_3->setWidget(2, QFormLayout::LabelRole, fname_label);

        fname_lineedit = new QLineEdit(image_groupbox);
        fname_lineedit->setObjectName(QStringLiteral("fname_lineedit"));

        formLayout_3->setWidget(2, QFormLayout::FieldRole, fname_lineedit);

        autosave_checkbox = new QCheckBox(image_groupbox);
        autosave_checkbox->setObjectName(QStringLiteral("autosave_checkbox"));
        autosave_checkbox->setCursor(QCursor(Qt::PointingHandCursor));

        formLayout_3->setWidget(3, QFormLayout::LabelRole, autosave_checkbox);

        widget_3 = new QWidget(image_groupbox);
        widget_3->setObjectName(QStringLiteral("widget_3"));
        horizontalLayout_4 = new QHBoxLayout(widget_3);
        horizontalLayout_4->setSpacing(6);
        horizontalLayout_4->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_4->setObjectName(QStringLiteral("horizontalLayout_4"));
        horizontalLayout_4->setContentsMargins(0, 0, 0, 0);
        every_label = new QLabel(widget_3);
        every_label->setObjectName(QStringLiteral("every_label"));

        horizontalLayout_4->addWidget(every_label);

        save_iters = new QSpinBox(widget_3);
        save_iters->setObjectName(QStringLiteral("save_iters"));
        save_iters->setCursor(QCursor(Qt::PointingHandCursor));
        save_iters->setWrapping(false);
        save_iters->setFrame(true);
        save_iters->setReadOnly(false);
        save_iters->setProperty("showGroupSeparator", QVariant(true));
        save_iters->setMinimum(1);
        save_iters->setMaximum(99999999);
        save_iters->setValue(100);

        horizontalLayout_4->addWidget(save_iters);


        formLayout_3->setWidget(3, QFormLayout::FieldRole, widget_3);

        fsave_btn = new QPushButton(image_groupbox);
        fsave_btn->setObjectName(QStringLiteral("fsave_btn"));
        fsave_btn->setCursor(QCursor(Qt::PointingHandCursor));

        formLayout_3->setWidget(5, QFormLayout::FieldRole, fsave_btn);


        verticalLayout->addWidget(image_groupbox);

        patht_groupbox = new QGroupBox(settings_dockw_content);
        patht_groupbox->setObjectName(QStringLiteral("patht_groupbox"));
        formLayout = new QFormLayout(patht_groupbox);
        formLayout->setSpacing(6);
        formLayout->setContentsMargins(11, 11, 11, 11);
        formLayout->setObjectName(QStringLiteral("formLayout"));
        iter_label = new QLabel(patht_groupbox);
        iter_label->setObjectName(QStringLiteral("iter_label"));
        iter_label->setCursor(QCursor(Qt::ArrowCursor));

        formLayout->setWidget(0, QFormLayout::LabelRole, iter_label);

        lcdNumber = new QLCDNumber(patht_groupbox);
        lcdNumber->setObjectName(QStringLiteral("lcdNumber"));
        lcdNumber->setFrameShape(QFrame::NoFrame);
        lcdNumber->setFrameShadow(QFrame::Raised);
        lcdNumber->setLineWidth(0);
        lcdNumber->setMode(QLCDNumber::Dec);
        lcdNumber->setSegmentStyle(QLCDNumber::Flat);
        lcdNumber->setProperty("intValue", QVariant(0));

        formLayout->setWidget(0, QFormLayout::FieldRole, lcdNumber);

        maxd_label = new QLabel(patht_groupbox);
        maxd_label->setObjectName(QStringLiteral("maxd_label"));

        formLayout->setWidget(3, QFormLayout::LabelRole, maxd_label);

        maxd_spinbox = new QSpinBox(patht_groupbox);
        maxd_spinbox->setObjectName(QStringLiteral("maxd_spinbox"));
        maxd_spinbox->setCursor(QCursor(Qt::PointingHandCursor));
        maxd_spinbox->setValue(5);
        maxd_spinbox->setDisplayIntegerBase(10);

        formLayout->setWidget(3, QFormLayout::FieldRole, maxd_spinbox);

        widget = new QWidget(patht_groupbox);
        widget->setObjectName(QStringLiteral("widget"));
        horizontalLayout_2 = new QHBoxLayout(widget);
        horizontalLayout_2->setSpacing(6);
        horizontalLayout_2->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_2->setObjectName(QStringLiteral("horizontalLayout_2"));
        horizontalLayout_2->setContentsMargins(0, 0, 0, 0);
        resume_btn = new QPushButton(widget);
        resume_btn->setObjectName(QStringLiteral("resume_btn"));
        resume_btn->setCursor(QCursor(Qt::PointingHandCursor));

        horizontalLayout_2->addWidget(resume_btn);

        pause_btn = new QPushButton(widget);
        pause_btn->setObjectName(QStringLiteral("pause_btn"));
        pause_btn->setCursor(QCursor(Qt::PointingHandCursor));

        horizontalLayout_2->addWidget(pause_btn);

        restart_btn = new QPushButton(widget);
        restart_btn->setObjectName(QStringLiteral("restart_btn"));
        restart_btn->setCursor(QCursor(Qt::PointingHandCursor));

        horizontalLayout_2->addWidget(restart_btn);


        formLayout->setWidget(5, QFormLayout::SpanningRole, widget);

        widget_2 = new QWidget(patht_groupbox);
        widget_2->setObjectName(QStringLiteral("widget_2"));
        horizontalLayout_3 = new QHBoxLayout(widget_2);
        horizontalLayout_3->setSpacing(6);
        horizontalLayout_3->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_3->setObjectName(QStringLiteral("horizontalLayout_3"));
        horizontalLayout_3->setContentsMargins(0, 0, 0, 0);
        right_btn = new QPushButton(widget_2);
        right_btn->setObjectName(QStringLiteral("right_btn"));
        right_btn->setMaximumSize(QSize(16777215, 16777215));
        right_btn->setCursor(QCursor(Qt::PointingHandCursor));

        horizontalLayout_3->addWidget(right_btn);


        formLayout->setWidget(6, QFormLayout::SpanningRole, widget_2);


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
        aabb_checkbox->setText(QApplication::translate("MainWindow", "AABBoxes Preview mode", Q_NULLPTR));
        text_checkbox->setText(QApplication::translate("MainWindow", "Render textures", Q_NULLPTR));
        x_spinbox->setPrefix(QString());
        aajitter_label->setText(QApplication::translate("MainWindow", "Ray jittter", Q_NULLPTR));
        prev_checkbox->setText(QApplication::translate("MainWindow", "Simple Preview Mode", Q_NULLPTR));
        image_groupbox->setTitle(QApplication::translate("MainWindow", "Image", Q_NULLPTR));
        gamm_label->setText(QApplication::translate("MainWindow", "Gamma Corr. Exp", Q_NULLPTR));
        fname_label->setText(QApplication::translate("MainWindow", "File name", Q_NULLPTR));
        fname_lineedit->setText(QApplication::translate("MainWindow", "image.png", Q_NULLPTR));
        autosave_checkbox->setText(QApplication::translate("MainWindow", "Autosave", Q_NULLPTR));
        every_label->setText(QApplication::translate("MainWindow", "Every", Q_NULLPTR));
        save_iters->setSuffix(QApplication::translate("MainWindow", " iterations", Q_NULLPTR));
        fsave_btn->setText(QApplication::translate("MainWindow", "Save Image", Q_NULLPTR));
        patht_groupbox->setTitle(QApplication::translate("MainWindow", "Path Tracing", Q_NULLPTR));
        iter_label->setText(QApplication::translate("MainWindow", "Iteration(<time>s)", Q_NULLPTR));
        maxd_label->setText(QApplication::translate("MainWindow", "Max depth", Q_NULLPTR));
        resume_btn->setText(QApplication::translate("MainWindow", "Resume", Q_NULLPTR));
        pause_btn->setText(QApplication::translate("MainWindow", "Pause", Q_NULLPTR));
        restart_btn->setText(QApplication::translate("MainWindow", "Start", Q_NULLPTR));
        right_btn->setText(QApplication::translate("MainWindow", "Step >", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
