#ifndef PREVIEWGLWIDGET_H
#define PREVIEWGLWIDGET_H

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLBuffer>
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>


class PreviewGLWidget : public QOpenGLWidget, protected QOpenGLFunctions
{
	Q_OBJECT
public:
    PreviewGLWidget(QWidget* parent);
    ~PreviewGLWidget();

	QString getGLinfo();
	void reloadTexture(int, int);

protected:
    void initializeGL() override;
    void resizeGL(int, int) override;
    void paintGL() override;

private:
	QOpenGLBuffer* m_vertices;
	QOpenGLBuffer* m_texcoords;
	QOpenGLBuffer* m_indices;

	QOpenGLBuffer* m_pbo;
    QOpenGLShaderProgram m_program;
    QOpenGLTexture *m_texture;

	void initShader();
	void initPBO(int, int);
	void deletePBO();
	void initVAO();
	void imageTextureInit(int, int);
	void runCUDApbotest();

	void initNativeGL();
	void cleanupNativeGL();

private slots:
	void aboutToResize_slot();

public slots:
	void refresh(int, double);
};

#endif // PREVIEWGLWIDGET_H
