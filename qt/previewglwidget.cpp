#include "previewglwidget.h"
#include "world.h"
#include "cudaUtility.h"
#include "pbotest.h"
#include <cuda_gl_interop.h>

#include "kernel.h"

// default shaders paths
const char* shaderVertDefault = "shaders/default.vert";
const char* shaderFragDefault = "shaders/default.frag";


PreviewGLWidget::PreviewGLWidget(QWidget* parent)
	: m_pbo(0), m_texcoords(0), m_indices(0),
	m_vertices(0), m_texture(0), QOpenGLWidget(parent) {}

PreviewGLWidget::~PreviewGLWidget()
{
	makeCurrent();

	if (m_vertices)
		m_vertices->destroy();
	if (m_indices)
		m_indices->destroy();
	if (m_texcoords)
		m_texcoords->destroy();

	delete m_texture;
	delete m_vertices;
	delete m_indices;
	delete m_texcoords;

	deletePBO();

	doneCurrent();
}

QString PreviewGLWidget::getGLinfo() {
	// get GL info
	const GLubyte* version = glGetString(GL_VERSION);
	const GLubyte* renderer = glGetString(GL_RENDERER);
	char buffer[1024];
	sprintf_s(buffer, "OpenGL version: %s\nPreview image renderer: %s\n", (const char*)version, (const char*)renderer);
	return QString::fromLocal8Bit(buffer);
}

void PreviewGLWidget::initShader() {
	
	// Compile vertex shader
	if (!m_program.addShaderFromSourceFile(QOpenGLShader::Vertex, shaderVertDefault)) {
		printf("PreviewGLWidget::initShader(): compile vert: %s\n", m_program.log().toStdString().c_str());
		close();
	}
	// Compile fragment shader
	if (!m_program.addShaderFromSourceFile(QOpenGLShader::Fragment, shaderFragDefault)) {
		printf("PreviewGLWidget::initShader(): compile frag: %s\n", m_program.log().toStdString().c_str());
		close();
	}
	// Link shader pipeline
	if (!m_program.link()) {
		printf("PreviewGLWidget::initShader() link: %s\n", m_program.log().toStdString().c_str());
		close();
	}
	// Bind shader pipeline for use
	if (!m_program.bind()) {
		printf("PreviewGLWidget::initShader() bind: %s\n", m_program.log().toStdString().c_str());
		close();
	}
}

void PreviewGLWidget::initPBO(int w, int h) {
	int sizeof_pbo = 4 * w * h * sizeof(GLubyte);
	assert(sizeof_pbo > 0);
	m_pbo = new QOpenGLBuffer(QOpenGLBuffer::PixelUnpackBuffer);
	m_pbo->setUsagePattern(QOpenGLBuffer::StaticDraw);
	m_pbo->create();
	m_pbo->bind();
	m_pbo->allocate(sizeof_pbo);
	cudaOk(cudaGraphicsGLRegisterBuffer(&viewPBO_cuda, m_pbo->bufferId(), cudaGraphicsMapFlagsWriteDiscard));

	size_t num_bytes;
	// map buffer object
	cudaOk(cudaGraphicsMapResources(1, &viewPBO_cuda));
	cudaOk(cudaGraphicsResourceGetMappedPointer((void**)&pbo_dptr, &num_bytes, viewPBO_cuda));
	assert(pbo_dptr);
}

void PreviewGLWidget::deletePBO() {
	if (m_pbo) {
		cudaOk(cudaGraphicsUnmapResources(1, &viewPBO_cuda));
		cudaOk(cudaGraphicsUnregisterResource(viewPBO_cuda));

		m_pbo->bind();
		m_pbo->destroy();

		delete m_pbo;
		m_pbo = NULL;
	}
}

void PreviewGLWidget::initVAO() {

	GLfloat vertices[] = {
		-1.0f, -1.0f,
		1.0f, -1.0f,
		1.0f, 1.0f,
		-1.0f, 1.0f
	};

	GLfloat texcoords[] = {
		1.0f, 1.0f,
		0.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f
	};

	GLushort indices[] = { 0, 1, 3, 3, 1, 2 };

	m_vertices = new QOpenGLBuffer(QOpenGLBuffer::VertexBuffer);
	m_texcoords = new QOpenGLBuffer(QOpenGLBuffer::VertexBuffer);
	m_indices = new QOpenGLBuffer(QOpenGLBuffer::IndexBuffer);

	m_vertices->create();
	m_indices->create();
	m_texcoords->create();

	m_vertices->bind();
	m_vertices->allocate(vertices, sizeof(vertices));

	m_indices->bind();
	m_indices->allocate(indices, sizeof(indices));

	m_texcoords->bind();
	m_texcoords->allocate(texcoords, sizeof(texcoords));

	m_vertices->bind();
	int vloc = m_program.attributeLocation("Position");
	m_program.enableAttributeArray(vloc);
	m_program.setAttributeBuffer(vloc, GL_FLOAT, 0, 2);

	m_texcoords->bind();
	int tcloc = m_program.attributeLocation("Texcoords");
	m_program.enableAttributeArray(tcloc);
	m_program.setAttributeBuffer(tcloc, GL_FLOAT, 0, 2);
}

void PreviewGLWidget::imageTextureInit(int w, int h) {
	if (m_texture) {
		m_texture->destroy();
		m_texture = NULL;
	}
	m_texture = new QOpenGLTexture(QOpenGLTexture::Target2D);
	m_texture->create();
	assert(w > 0 && h > 0);
	m_texture->setSize(w, h);
	m_texture->setMinMagFilters(QOpenGLTexture::Linear, QOpenGLTexture::Linear);
	m_texture->setFormat(QOpenGLTexture::RGBA8_UNorm);
	m_texture->allocateStorage(QOpenGLTexture::BGRA, QOpenGLTexture::UInt8);
}

void PreviewGLWidget::initializeGL()
{
	initializeOpenGLFunctions();

	// compile and link shader
	initShader();

	//default maximum resolution
	initPBO(1280, 1024);
	initVAO();
	imageTextureInit(32, 32);
}

void PreviewGLWidget::resizeGL(int w, int h)
{
	printf("PreviewGLWidget::resizeGL(): %dx%d\n", w, h);
}

void PreviewGLWidget::paintGL()
{
	m_pbo->bind();
	m_texture->bind();

	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_texture->width(), m_texture->height(), GL_RGBA, GL_UNSIGNED_BYTE, NULL);

	glClear(GL_COLOR_BUFFER_BIT);
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);
	m_texture->release();
	m_pbo->release();
}

void PreviewGLWidget::refresh(int iter, double time) {
	update();
}

void PreviewGLWidget::reloadTexture(int w, int h) {
	imageTextureInit(w, h);
}