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
public:
    PreviewGLWidget(QWidget* parent);
    ~PreviewGLWidget();

	QString getGLinfo();
	

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
	void initPBO();
	void deletePBO();
	void initVAO();
	void imageTextureInit();

};

#endif // PREVIEWGLWIDGET_H
