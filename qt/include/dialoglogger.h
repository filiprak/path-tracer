#ifndef DIALOGLOGGER_H
#define DIALOGLOGGER_H

#include <QObject>

#define MAX_LOG_SIZE 1024


class DialogLogger : public QObject
{
    Q_OBJECT
public:
    explicit DialogLogger(QObject *parent = nullptr);

    void info(const char* format, ...);
    void error(const char* format, ...);
    void warning(const char* format, ...);

    void progress(int p);

signals:
    void info_signal(const QString& msg);
    void progress_signal(int);

public slots:
};

#endif // DIALOGLOGGER_H
