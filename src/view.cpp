#include "main.h"
#include "view.h"
#include "world.h"
#include "camera.h"
#include "config.h"
#include "shaderUtility.h"
#include "cudaUtility.h"
#include "kernel.h"
#include <time.h>

// default shaders paths
const char* shaderVertDefault = "shaders/default.vert";
const char* shaderFragDefault = "shaders/default.frag";

// main rendering view window
GLFWwindow *window;

GLuint viewPBO_id;
struct cudaGraphicsResource* viewPBO_cuda;

GLuint viewTexture_id;


void viewPrintGLinfo() {
	// get GL info
	const GLubyte* version = glGetString(GL_VERSION);
	const GLubyte* renderer = glGetString(GL_RENDERER);
	printSep();
	printf("OpenGL version: %s\nPreview image renderer: %s\n", (const char*)version, (const char*)renderer);
	printSep();
}

GLuint viewInitShader() {
	GLuint program = glCreateProgram();

	glBindAttribLocation(program, 0, "Position");
	glBindAttribLocation(program, 1, "Texcoords");

	ShaderUtils::ShaderId ids = ShaderUtils::loadAndCompileShaders(shaderVertDefault, shaderFragDefault);
	ShaderUtils::linkShaders(program, ids);

	GLint location;
	if ((location = glGetUniformLocation(program, "u_image")) != -1) {
		glUniform1i(location, 0);
	}

	return program;
}

void viewPBOinit(GLuint* pbo) {
	// set up vertex data parameter
	Camera& cam = scene.camera;
	int sizeof_pbo = 4 * cam.projection.num_pixels * sizeof(GLubyte);

	glGenBuffers(1, pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, sizeof_pbo, NULL, GL_DYNAMIC_COPY);
	cudaGraphicsGLRegisterBuffer(&viewPBO_cuda, *pbo, cudaGraphicsMapFlagsWriteDiscard);
}

void viewPBOdelete(GLuint* pbo) {
	if (pbo) {
		// unregister this buffer object with CUDA
		cudaGraphicsUnregisterResource(viewPBO_cuda);

		glBindBuffer(GL_ARRAY_BUFFER, *pbo);
		glDeleteBuffers(1, pbo);

		*pbo = (GLuint)NULL;
	}
}

void viewVAOinit(void) {
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

	GLuint VAOid[3];
	glGenBuffers(3, VAOid);

	glBindBuffer(GL_ARRAY_BUFFER, VAOid[0]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
	glVertexAttribPointer((GLuint)posLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(posLocation);

	glBindBuffer(GL_ARRAY_BUFFER, VAOid[1]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
	glVertexAttribPointer((GLuint)texLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(texLocation);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, VAOid[2]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}

void viewTextureInit(GLuint* id) {
	glGenTextures(1, id);
	glBindTexture(GL_TEXTURE_2D, *id);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,
		scene.camera.projection.width,
		scene.camera.projection.height,
		0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
}

void viewTextureDelete(GLuint* id) {
	if (id) {
		glDeleteTextures(1, id);
		*id = (GLuint) NULL;
	}
}

void viewCleanCUDA() {
	viewPBOdelete(&viewPBO_id);
	viewTextureDelete(&viewTexture_id);
}

void viewInitCUDA() {
	atexit(viewCleanCUDA);
}


bool viewInit(const Json::Value& jcam) {
	Camera& cam = scene.camera;
	glfwSetErrorCallback(viewErrCallback);

	// Initialize glfw library
	if (!glfwInit()) {
		exit(EXIT_FAILURE);
	}

	window = glfwCreateWindow(cam.projection.width, cam.projection.height, "path-tracer", NULL, NULL);
	if (!window) {
		glfwTerminate();
		return false;
	}
	// set context
	glfwMakeContextCurrent(window);
	glfwSetKeyCallback(window, viewKeyCallback);

	// Initialize glew library
	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK) {
		return false;
	}

	// Setup view rendering
	viewVAOinit();
	viewTextureInit(&viewTexture_id);
	viewInitCUDA();
	viewPBOinit(&viewPBO_id);

	// compile and link shader
	GLuint shader = viewInitShader();
	glUseProgram(shader);

	// use view texture
	glActiveTexture(GL_TEXTURE0);

	viewPrintGLinfo();
	return true;
}

void viewLoop() {
	char title[48];
	int2 screen_size = make_int2(scene.camera.projection.width, scene.camera.projection.height);

	// init cuda kernel
	kernelInit();

	printProgramConfig();

	int iter = 1;
	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();
		if (scene.camera.changed) {
			iter = 1;
			printCamInfo();
		}
		clock_t begin = clock();
		runCUDA(iter);
		clock_t end = clock();
		if (checkCudaError("runCuda()"))
			break;
		
		sprintf_s(title, "Iter: %d [ %.3f s ]", iter, (double)(end - begin) / CLOCKS_PER_SEC);
		glfwSetWindowTitle(window, title);

		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, viewPBO_id);
		glBindTexture(GL_TEXTURE_2D, viewTexture_id);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, screen_size.x, screen_size.y, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
		glClear(GL_COLOR_BUFFER_BIT);

		// VAO, shader program, and texture already bound
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);
		glfwSwapBuffers(window);

		iter++;
		if (iter > 50000)
			break;
	}
	// clean cuda kernel
	kernelCleanUp();

	// clean glfw
	glfwDestroyWindow(window);
	glfwTerminate();
}

void viewErrCallback(int error, const char* description)
{
	fprintf(stderr, "GLFW ERROR (code %d): %s\n", error, description);
}

static void viewKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	printf("viewKeyCallback();\n");
	int rcontrol_pressed = glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS;
	int lcontrol_pressed = glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS;
	float scale = rcontrol_pressed ? 3.0f : 1.0f;
	scale *= lcontrol_pressed ? 10.0f : 1.0f;

	Camera& cam = scene.camera;

	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, GL_TRUE);
		return;
	}
	if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
		resetCamera();
		return;
	}
	if (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS) {
		togglePrevMode();
		return;
	}
	if (glfwGetKey(window, GLFW_KEY_T) == GLFW_PRESS) {
		toggleTextures();
		return;
	}

	if (action == GLFW_PRESS) {
		key == GLFW_KEY_COMMA ? updateMaxRayBnc(-1) :
			(key == GLFW_KEY_PERIOD ? updateMaxRayBnc(1) : NULL);

		key == GLFW_KEY_LEFT_BRACKET ? updateAajitter(-0.1f * scale) :
			(key == GLFW_KEY_RIGHT_BRACKET ? updateAajitter(0.1f * scale) : NULL);

		key == GLFW_KEY_9 ? updateGamma(-0.05f * scale) :
			(key == GLFW_KEY_0 ? updateGamma(0.05f * scale) : NULL);
	}

	if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
		rotateVCamera(-CAM_ROTATE_ANGLE_DELTA * scale);
	if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
		rotateVCamera(CAM_ROTATE_ANGLE_DELTA * scale);
	if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
		rotateHCamera(-CAM_ROTATE_ANGLE_DELTA * scale);
	if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
		rotateHCamera(CAM_ROTATE_ANGLE_DELTA * scale);

	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		moveCamera(make_float3(-CAM_MOVE_DISTANCE_DELTA * scale, .0, .0));
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		moveCamera(make_float3(CAM_MOVE_DISTANCE_DELTA * scale, .0, .0));

	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		moveCamera(make_float3(.0, CAM_MOVE_DISTANCE_DELTA * scale, .0));
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		moveCamera(make_float3(.0, -CAM_MOVE_DISTANCE_DELTA * scale, .0));

	if (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS)
		moveCamera(make_float3(.0, .0, CAM_MOVE_DISTANCE_DELTA * scale));
	if (glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS)
		moveCamera(make_float3(.0, .0, -CAM_MOVE_DISTANCE_DELTA * scale));
}

void runCUDA(int iter) {
	uchar4 *pbo_dptr = NULL;
	size_t num_bytes;

	// map buffer object
	cudaGraphicsMapResources(1, &viewPBO_cuda, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&pbo_dptr, &num_bytes, viewPBO_cuda);

	// execute the kernel - start device code execution
	kernelMain(pbo_dptr, iter);

	// unmap buffer object
	cudaGraphicsUnmapResources(1, &viewPBO_cuda, 0);
}