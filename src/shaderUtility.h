#pragma once

#include <GL/glew.h>
#include <string>
#include <vector>
#include <iostream>


namespace ShaderUtils {

	// struct for storing shader ids
	typedef struct {
		GLuint vertId;
		GLuint fragId;
		GLint geomId;

	} ShaderId;

	ShaderId loadAndCompileShaders(const char* vertexPath, const char* fragmentPath, const char* geometryPath = NULL);
	bool linkShaders(GLuint program, ShaderId shaders);
	GLint compileShader(const char* name, const char* csource, GLenum type);
	std::string loadFile(const char *fname);
}