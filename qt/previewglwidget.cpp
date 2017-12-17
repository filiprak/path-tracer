#include "previewglwidget.h"
#include "world.h"
#include <cuda_gl_interop.h>


// default shaders paths
const char* shaderVertDefault = "shaders/default.vert";
const char* shaderFragDefault = "shaders/default.frag";

GLuint viewPBO_id;
struct cudaGraphicsResource* viewPBO_cuda;

GLuint viewTexture_id;


PreviewGLWidget::PreviewGLWidget(QWidget* parent)
	: m_pbo(0), m_texcoords(0), m_indices(0),
	m_vertices(0), m_texture(0), QOpenGLWidget(parent)
{

}

PreviewGLWidget::~PreviewGLWidget()
{
	makeCurrent();

	m_vertices->destroy();
	m_indices->destroy();
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
	if (!this->m_program.addShaderFromSourceFile(QOpenGLShader::Vertex, shaderVertDefault))
		close();

	// Compile fragment shader
	if (!this->m_program.addShaderFromSourceFile(QOpenGLShader::Fragment, shaderFragDefault))
		close();

	// Link shader pipeline
	if (!this->m_program.link())
		close();

	// Bind shader pipeline for use
	if (!this->m_program.bind())
		close();

	this->m_program.bindAttributeLocation("Position", 0);
	this->m_program.bindAttributeLocation("Texcoords", 1);
}

void PreviewGLWidget::initPBO() {
	// set up vertex data parameter
	Camera& cam = scene.camera;
	int sizeof_pbo = 4 * cam.projection.num_pixels * sizeof(GLubyte);
	this->m_pbo = new QOpenGLBuffer(QOpenGLBuffer::PixelUnpackBuffer);
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
		-1.0f, 1.0f,
	};

	GLfloat texcoords[] = {
		1.0f, 1.0f,
		0.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f
	};

	GLushort indices[] = { 0, 1, 3, 3, 1, 2 };
	GLuint posLocation = 0, texLocation = 1;

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
}

void PreviewGLWidget::imageTextureInit() {
	m_texture = new QOpenGLTexture(QOpenGLTexture::Target2D);
	m_texture->create();
	m_texture->bind();
	m_texture->setSize(scene.camera.projection.width, scene.camera.projection.height);
	m_texture->setMinMagFilters(QOpenGLTexture::Nearest, QOpenGLTexture::Nearest);
	m_texture->allocateStorage(QOpenGLTexture::BGRA, QOpenGLTexture::UInt8);
}

void PreviewGLWidget::initializeGL()
{
	initializeOpenGLFunctions();

	// Setup view rendering
	initPBO();
	initVAO();
	imageTextureInit();

	// compile and link shader
	initShader();
}

void PreviewGLWidget::resizeGL(int, int)
{

}

void PreviewGLWidget::paintGL()
{
	this->m_pbo->bind();
	this->m_texture->bind();
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
		scene.camera.projection.width,
		scene.camera.projection.height,
		GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glClear(GL_COLOR_BUFFER_BIT);
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);
}
