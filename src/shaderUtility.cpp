#include "shaderUtility.h"
#include <fstream>

namespace ShaderUtils {

	void printShaderInfoLog(GLuint shader) {
		GLint maxLength = 0;
		glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &maxLength);
		if (maxLength < 1) {
			std::cout << "empty info log" << std::endl;
			return;
		}

		std::vector<GLchar> infoLog(maxLength);
		glGetShaderInfoLog(shader, maxLength, &maxLength, &infoLog[0]);
		std::string infoLogString(infoLog.begin(), infoLog.end());
		std::cout << "InfoLog: " << std::endl << infoLogString << std::endl;
	}

	void printLinkInfoLog(GLuint program) {
		GLint maxLength = 0;
		glGetProgramiv(program, GL_INFO_LOG_LENGTH, &maxLength);
		if (maxLength < 1) {
			std::cout << "empty info log" << std::endl;
			return;
		}

		std::vector<GLchar> infoLog(maxLength);
		glGetProgramInfoLog(program, maxLength, &maxLength, &infoLog[0]);
		std::string infoLogString(infoLog.begin(), infoLog.end());
		std::cout << "InfoLog: " << std::endl << infoLogString << std::endl;
	}

	ShaderId loadAndCompileShaders(const char* vertexPath, const char* fragmentPath, const char* geometryPath) {

		ShaderId result;

		// compile shaders
		std::string vertSrc = loadFile(vertexPath);
		result.vertId = compileShader(vertexPath, vertSrc.c_str(), GL_VERTEX_SHADER);

		std::string fragSrc = loadFile(fragmentPath).c_str();
		result.fragId = compileShader(fragmentPath, fragSrc.c_str(), GL_FRAGMENT_SHADER);

		if (geometryPath) {
			std::string geomSrc = loadFile(geometryPath).c_str();
			result.geomId = compileShader(geometryPath, geomSrc.c_str(), GL_GEOMETRY_SHADER);
		}
		else {
			result.geomId = -1;
		}
		return result;
	}

	bool linkShaders(GLuint program, ShaderId shaders) {
		bool geom = shaders.geomId != -1;

		//Attach shaders to program
		glAttachShader(program, shaders.vertId);
		glAttachShader(program, shaders.fragId);
		if (geom)
			glAttachShader(program, shaders.geomId);

		//Link program
		glLinkProgram(program);

		GLint isLinked = 0;
		glGetProgramiv(program, GL_LINK_STATUS, (int *)&isLinked);
		if (isLinked == GL_FALSE)
		{
			std::cout << "Error: Linking shaders: ";
			printLinkInfoLog(program);

			glDeleteProgram(program);

			glDeleteShader(shaders.vertId);
			glDeleteShader(shaders.fragId);
			if (geom)
				glDeleteShader(shaders.geomId);

			return false;
		}

		// detach shaders
		glDetachShader(program, shaders.vertId);
		glDetachShader(program, shaders.fragId);
		if (geom)
			glDetachShader(program, shaders.geomId);

		return true;
	}

	GLint compileShader(const char* name, const char* csource, GLenum type) {
		GLuint shaderId = glCreateShader(type);

		glShaderSource(shaderId, 1, &csource, 0);
		glCompileShader(shaderId);

		GLint isCompiled = 0;
		glGetShaderiv(shaderId, GL_COMPILE_STATUS, &isCompiled);

		if (isCompiled == GL_FALSE) {
			std::cout << "Error: Compiling shader: " << name << ": ";
			ShaderUtils::printShaderInfoLog(shaderId);
			glDeleteShader(shaderId);
		}
		return shaderId;
	}

	std::string loadFile(const char *filePath) {
		std::ifstream ifs(filePath);
		std::string content((std::istreambuf_iterator<char>(ifs)),
			(std::istreambuf_iterator<char>()));
		return content;
	}
}