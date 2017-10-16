#pragma once

#include <GL/glew.h>
#include <GLFW\glfw3.h>

#include <cuda_gl_interop.h>

extern GLuint viewPBO_id;
extern GLuint viewWidth, viewHeight;

bool viewInit();

void viewLoop();

void runCUDA(int iter);

void viewErrCallback(int error, const char* description);

static void viewKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);