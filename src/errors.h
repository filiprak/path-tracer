#pragma once

#include <exception>
#include <string>


class scene_file_error : public std::runtime_error {
	public:
		explicit scene_file_error(const std::string& what_arg)
			: std::runtime_error(what_arg) {};
		explicit scene_file_error(const char* what_arg)
			: std::runtime_error(what_arg) {};
};