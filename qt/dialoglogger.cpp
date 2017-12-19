#include "dialoglogger.h"

DialogLogger::DialogLogger(QObject *parent) : QObject(parent)
{

}

void DialogLogger::info(const char *format, ...)
{
    char buffer[MAX_LOG_SIZE];
    va_list argptr;
    va_start(argptr, format);
    vsnprintf_s(buffer, MAX_LOG_SIZE, format, argptr);
    emit info_signal(QString::fromLocal8Bit(buffer));
    va_end(argptr);
}

void DialogLogger::error(const char *format, ...)
{
    char buffer[MAX_LOG_SIZE];
    va_list argptr;
    va_start(argptr, format);
	vsnprintf_s(buffer, MAX_LOG_SIZE, format, argptr);
    emit info_signal("<span style='color: red; font-weight: bold'>" + QString::fromLocal8Bit(buffer) + "</span>");
    va_end(argptr);
}

void DialogLogger::warning(const char *format, ...)
{
    char buffer[MAX_LOG_SIZE];
    va_list argptr;
    va_start(argptr, format);
	vsnprintf_s(buffer, MAX_LOG_SIZE, format, argptr);
    emit info_signal("<span style='color: orange;'>" + QString::fromLocal8Bit(buffer) + "</span>");
    va_end(argptr);
}

void DialogLogger::progress(int p)
{
    emit progress_signal(p);
}
