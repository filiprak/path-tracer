cmake_minimum_required(VERSION 3.0)

project(path-tracer)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# Set up include and lib paths
set(EXTERNAL "external")
include_directories("${EXTERNAL}")
include_directories("${EXTERNAL}/include/")

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    set(EXTERNAL_LIB_PATH "${EXTERNAL}/lib/linux" "/usr/lib64")
elseif(WIN32)
    set(EXTERNAL_LIB_PATH "${EXTERNAL}/lib/win")
endif()
link_directories(${EXTERNAL_LIB_PATH})
list(APPEND CMAKE_LIBRARY_PATH "${EXTERNAL_LIB_PATH}")


# Find up and set up core dependency libs

# GLFW
set(GLFW_INCLUDE_DIR "${EXTERNAL}/include")
set(GLFW_LIBRARY_DIR "${CMAKE_LIBRARY_PATH}")
find_library(GLFW_LIBRARY "glfw3" HINTS "${GLFW_LIBRARY_DIR}")

# GLEW
set(GLEW_INCLUDE_DIR "${EXTERNAL}/include")
set(GLEW_LIBRARY_DIR "${CMAKE_LIBRARY_PATH}")
add_definitions(-DGLEW_STATIC)
find_package(GLEW REQUIRED)

# ASSIMP
set(ASSIMP_INCLUDE_DIRS "${EXTERNAL}/include")
set(ASSIMP_LIBRARY_DIRS "${CMAKE_LIBRARY_PATH}")
find_package(ASSIMP REQUIRED)

# OPENGL
find_package(OpenGL REQUIRED)


set(CORELIBS
    "${GLFW_LIBRARY}"
    "${OPENGL_LIBRARY}"
    "${GLEW_LIBRARY}"
	"${ASSIMP_LIBRARY}"
    )

# Enable C++11 for host code
set(CMAKE_CXX_STANDARD 11)

# Enable CUDA debug info in debug mode builds
list(APPEND CUDA_NVCC_FLAGS_DEBUG -G -g)

# Specify CUDA nvcc flags
list(APPEND CUDA_NVCC_FLAGS -arch=sm_35)

# Linux-specific hacks/fixes
if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    list(APPEND CMAKE_EXE_LINKER_FLAGS "-lX11 -lXxf86vm -lXrandr -lXi")
endif()

# CUDA linking
find_package(Threads REQUIRED)
find_package(CUDA REQUIRED)

set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
set(CUDA_SEPARABLE_COMPILATION ON)


set(SRC_PATH "src")
# set up source files
file(GLOB_RECURSE SOURCES
    "${SRC_PATH}/*.h"
	"${SRC_PATH}/*.hpp"
    "${SRC_PATH}/*.cpp"
	"${SRC_PATH}/*.cu"
	"${SRC_PATH}/*.cuh"
)

cuda_add_executable(${CMAKE_PROJECT_NAME}
    ${SOURCES}
    )

target_link_libraries(${CMAKE_PROJECT_NAME}
    ${CORELIBS}
    )

add_custom_command(
    TARGET ${CMAKE_PROJECT_NAME}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/shaders
        ${CMAKE_BINARY_DIR}/shaders
    )
