#include "main.h"
#include "view.h"

// main rendering view window
GLFWwindow *window;


bool viewInit(int width, int height) {
	glfwSetErrorCallback(viewErrCallback);

	/* Initialize glfw library */
	if (!glfwInit())
		return false;

	window = glfwCreateWindow(width, height, "path-tracer", NULL, NULL);
	if (!window) {
		glfwTerminate();
		return false;
	}

	glfwMakeContextCurrent(window);
	glfwSetKeyCallback(window, viewKeyCallback);
}

void viewLoop() {
	while (!glfwWindowShouldClose(window))
	{
		/* Render here */
		glClear(GL_COLOR_BUFFER_BIT);

		/* Swap front and back buffers */
		glfwSwapBuffers(window);

		/* Poll for and process events */
		glfwPollEvents();
	}
	// clean glfw
	glfwDestroyWindow(window);
	glfwTerminate();
}

void viewErrCallback(int error, const char* description)
{
	fprintf(stderr, "Error: %s\n", description);
}

static void viewKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GL_TRUE);
}