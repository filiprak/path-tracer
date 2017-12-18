#include "previewglwidget.h"
#include "world.h"
#include "pbotest.h"
#include <cuda_gl_interop.h>

#include "kernel.h"

// default shaders paths
const char* shaderVertDefault = "shaders/default.vert";
const char* shaderFragDefault = "shaders/default.frag";


PreviewGLWidget::PreviewGLWidget(QWidget* parent)
	: m_pbo(0), m_texcoords(0), m_indices(0),
	m_vertices(0), m_texture(0), QOpenGLWidget(parent)
{
	connect(this, SIGNAL(aboutToResize()), this, SLOT(aboutToResize_slot()), Qt::QueuedConnection);
	//testing parameters
	scene.camera.projection.width = 4;
	scene.camera.projection.height = 8;
	scene.camera.projection.num_pixels = 4 * 8;
	printf("GLsize: %d, %d\n", this->width(), this->height());
}

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
	if (!this->m_program.addShaderFromSourceFile(QOpenGLShader::Vertex, shaderVertDefault)) {
		printf("PreviewGLWidget::initShader(): compile vert: %s\n", this->m_program.log().toStdString().c_str());
		close();
	}

	// Compile fragment shader
	if (!this->m_program.addShaderFromSourceFile(QOpenGLShader::Fragment, shaderFragDefault)) {
		printf("PreviewGLWidget::initShader(): compile frag: %s\n", this->m_program.log().toStdString().c_str());
		close();
	}

	// Link shader pipeline
	if (!this->m_program.link()) {
		printf("PreviewGLWidget::initShader() link: %s\n", this->m_program.log().toStdString().c_str());
		close();
	}

	// Bind shader pipeline for use
	if (!this->m_program.bind()) {
		printf("PreviewGLWidget::initShader() bind: %s\n", this->m_program.log().toStdString().c_str());
		close();
	}
}

void PreviewGLWidget::initPBO(int w, int h) {
	int sizeof_pbo = 4 * w * h * sizeof(GLubyte);
	assert(sizeof_pbo > 0);
	this->m_pbo = new QOpenGLBuffer(QOpenGLBuffer::PixelUnpackBuffer);
	this->m_pbo->setUsagePattern(QOpenGLBuffer::StaticDraw);
	this->m_pbo->create();
	this->m_pbo->bind();
	this->m_pbo->allocate(sizeof_pbo);
	cudaGraphicsGLRegisterBuffer(&viewPBO_cuda, this->m_pbo->bufferId(), cudaGraphicsMapFlagsWriteDiscard);
}

void PreviewGLWidget::deletePBO() {
	if (this->m_pbo) {
		cudaGraphicsUnregisterResource(viewPBO_cuda);

		this->m_pbo->bind();
		this->m_pbo->destroy();

		delete this->m_pbo;
		this->m_pbo = NULL;
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
	int vloc = this->m_program.attributeLocation("Position");
	this->m_program.enableAttributeArray(vloc);
	this->m_program.setAttributeBuffer(vloc, GL_FLOAT, 0, 2);

	m_texcoords->bind();
	int tcloc = this->m_program.attributeLocation("Texcoords");
	this->m_program.enableAttributeArray(tcloc);
	this->m_program.setAttributeBuffer(tcloc, GL_FLOAT, 0, 2);
}

void PreviewGLWidget::imageTextureInit() {
	m_texture = new QOpenGLTexture(QOpenGLTexture::Target2D);
	m_texture->create();
	assert(scene.camera.projection.width > 0);
	assert(scene.camera.projection.height > 0);
	m_texture->setSize(scene.camera.projection.width, scene.camera.projection.height);
	m_texture->setMinMagFilters(QOpenGLTexture::Nearest, QOpenGLTexture::Nearest);
	m_texture->setFormat(QOpenGLTexture::RGBA8_UNorm);
	m_texture->allocateStorage(QOpenGLTexture::RGBA, QOpenGLTexture::UInt8);
}

void PreviewGLWidget::initializeGL()
{
	initializeOpenGLFunctions();

	// Setup view rendering
	printf("GLsize: %d, %d\n", this->width(), this->height());
	// compile and link shader
	initShader();

	//default maximum resolution
	initPBO(1280, 1024);
	initVAO();
	imageTextureInit();
}

//debug - testing
void PreviewGLWidget::aboutToResize_slot()
{
	//deletePBO();
	//initPBO(this->width(), this->height());
	printf("aboutToResize_slot: %d, %d\n", this->width(), this->height());
}

void PreviewGLWidget::resizeGL(int w, int h)
{
	printf("resizeGL: %d, %d\n", w, h);
}

//debug
void PreviewGLWidget::runCUDApbotest() {
	uchar4 *pbo_dptr = NULL;
	size_t num_bytes;

	// map buffer object
	cudaGraphicsMapResources(1, &viewPBO_cuda, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&pbo_dptr, &num_bytes, viewPBO_cuda);

	pbotestRun(pbo_dptr,
		scene.camera.projection.width,
		scene.camera.projection.height);

	// unmap buffer object
	cudaGraphicsUnmapResources(1, &viewPBO_cuda, 0);
}

void PreviewGLWidget::paintGL()
{
	this->m_pbo->bind();
	this->m_texture->bind();

	printf("GL error: %d\n", glGetError());
	printf("GLsize: %d, %d\n", this->width(), this->height());

	runCUDApbotest();

	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
		scene.camera.projection.width,
		scene.camera.projection.height,
		GL_RGBA, GL_UNSIGNED_BYTE, NULL);

	glClear(GL_COLOR_BUFFER_BIT);
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);
}