#pragma once

#include <GLFW\glfw3.h>


bool viewInit(int width, int height);

void viewLoop();

void viewErrCallback(int error, const char* description);