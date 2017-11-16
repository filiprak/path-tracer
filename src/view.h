#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h>

extern GLuint viewPBO_id;
extern GLuint viewWidth, viewHeight;


bool viewInit();

void viewLoop();

void runCUDA(int);

void viewErrCallback(int, const char*);

static void viewKeyCallback(GLFWwindow*, int, int, int, int);